from types import SimpleNamespace
from collections import deque
import os
import argparse

import dlib
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
import torchvision.transforms as transforms
import wandb
import timm
from timm.data.config import resolve_model_data_config

from utils import *
from detector import Detector
from dataloader import ALPRDataset



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_detection(
    model,
    train_dataloader,
    val_dataloader,
    device,
    vocab=None,
    start_epoch=0,
    num_epochs=50,
    version='',
    head_only_epochs=10,
    start_lr=1e-3,
    min_lr=1e-5,
    backbone_lr_factor=0.1,
    weight_decay=0.01,
    plateau_patience=2500,
    temp_model_path=None,
    use_wandb=False,
    compile=False
):
    """
    Training function with initial head-only training followed by full model training
    with differential learning rates for backbone and other components.
    """
    def create_optimizer(model, lr, backbone_lr_factor=1.0):
        backbone_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': lr * backbone_lr_factor},
            {'params': other_params, 'lr': lr}
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    def forward_pass(model, batch, device):
        images, gt_bboxes, gt_labels, gt_kps, _ = batch
        batch_size = images.size(0)
        
        gt_instances = [
            SimpleNamespace(**{
                'bboxes': gt_bboxes[i].to(device),
                'labels': gt_labels[i].to(device),
                'kps': gt_kps[i].to(device)
            }) for i in range(batch_size)
        ]
        
        img_metas = [{'pad_shape': (224, 224, 3), 'img_shape': (224, 224, 3)}] * batch_size
        
        cls_score, bbox_pred, kps_pred = model(images.to(device))
        losses = model.head.loss_by_feat(cls_score, bbox_pred, kps_pred, gt_instances, img_metas)
        
        total_loss = sum(
            loss.item() * (1 if 'kps' not in name else 0.01)
            for name, loss_group in losses.items()
            for loss in loss_group
        )
        
        return losses, total_loss
    
    def train_epoch(model, dataloader, optimizer, loss_history, curr_lr):
        model.train()
        epoch_losses = []
        
        for batch in dataloader:
            optimizer.zero_grad()
            losses, total_loss = forward_pass(model, batch, device)
            log = {}
            
            for loss_name, loss_group in losses.items():
                for loss_idx, loss in enumerate(loss_group):
                    loss.backward(retain_graph=True)
                    log[f'{loss_name}-{loss_idx}'] = float(loss.detach())
            
            optimizer.step()
            epoch_losses.append(total_loss)
            loss_history.append(total_loss)

            if use_wandb:
                log['train_loss'] = total_loss
                log['learning_rate'] = curr_lr
                wandb.log(log)
            
            # Check for plateau and adjust learning rate if needed
            if is_in_plateau(loss_history, threshold=plateau_patience):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    curr_lr = param_group['lr']
                loss_history = []  # Reset loss history after lr change
                print(f"Learning rate reduced to {curr_lr}")
                
                if curr_lr < min_lr:
                    print("Minimum learning rate reached")
                    return np.mean(epoch_losses), loss_history, curr_lr, True
            
        return np.mean(epoch_losses), loss_history, curr_lr, False
    
    def validate(model, dataloader):
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                _, total_loss = forward_pass(model, batch, device)
                val_losses.append(total_loss)
                
        return np.mean(val_losses)

    # Load checkpoint if resuming training
    curr_lr = start_lr
    loss_history = []
    training_phase = 1
    
    if temp_model_path and os.path.exists(temp_model_path):
        print(f"Loading checkpoint from {temp_model_path}")
        checkpoint = torch.load(temp_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        curr_lr = checkpoint.get('learning_rate', start_lr)
        loss_history = checkpoint.get('loss_history', [])
        start_epoch = checkpoint.get('epoch', 0) + 1
        training_phase = checkpoint.get('phase', 1)
        print(f"Resuming from epoch {start_epoch}, phase {training_phase}")

    # Compile model if requested
    if compile:
        model = torch.compile(model)
    
    # Phase 1: Train head only (if not already completed)
    if training_phase == 1:
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        head_optimizer = create_optimizer(model, curr_lr)
        
        if temp_model_path and os.path.exists(temp_model_path):
            head_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print("Phase 1: Training head only")
        for epoch in range(start_epoch, head_only_epochs):
            train_loss, loss_history, curr_lr, stop_training = train_epoch(
                model, train_dataloader, head_optimizer, loss_history, curr_lr)
            val_loss = validate(model, val_dataloader)
            
            if use_wandb:
                wandb.log({
                    'phase': 1,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': curr_lr
                })
            
            print(f'Phase 1 - Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            # Save checkpoint
            if temp_model_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': head_optimizer.state_dict(),
                    'epoch': epoch,
                    'phase': 1,
                    'learning_rate': curr_lr,
                    'loss_history': loss_history
                }, temp_model_path)
            
            if stop_training:
                break
        
        training_phase = 2
        start_epoch = 0
        curr_lr = start_lr
        loss_history = []
    
    # Phase 2: Train full model with different learning rates
    if training_phase == 2:
        for param in model.parameters():
            param.requires_grad = True
        full_optimizer = create_optimizer(model, curr_lr, backbone_lr_factor)
        
        print("Phase 2: Training full model")
        for epoch in range(start_epoch, num_epochs):
            train_loss, loss_history, curr_lr, stop_training = train_epoch(
                model, train_dataloader, full_optimizer, loss_history, curr_lr)
            val_loss = validate(model, val_dataloader)
            
            if use_wandb:
                wandb.log({
                    'phase': 2,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': curr_lr
                })
            
            print(f'Phase 2 - Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            # Save checkpoint
            if temp_model_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': full_optimizer.state_dict(),
                    'epoch': epoch,
                    'phase': 2,
                    'learning_rate': curr_lr,
                    'loss_history': loss_history
                }, temp_model_path)
            
            if stop_training:
                break


def is_in_plateau(vec, threshold):
    dlib_simple = dlib.count_steps_without_decrease(vec)
    dlib_robust = dlib.count_steps_without_decrease_robust(vec)
    return dlib_simple > threshold and dlib_robust > threshold

def dummy_dataloader(dataset_size: int, batch_size: int):
    img_meta = {
        'pad_shape': (224,224,3),
        'img_shape': (224,224,3),
    }

    for n in range(0, dataset_size, batch_size):
        l, t, r, b = 10, 20, 130, 140
        images = torch.zeros(batch_size,3,224,224, dtype=torch.float32).to(DEVICE)
        images[:,0,l:r,t] = 0.5
        images[:,0,l:r,b] = 0.5
        images[:,0,l,t:b] = 0.5
        images[:,0,r,t:b] = 0.5

        gt_instances_sn = []
        for i in range(batch_size):
            gt_bboxes = torch.Tensor(
                [[l, t, r, b]])
            gt_labels = torch.LongTensor([0])
            gt_kps = torch.Tensor(
                [[l,t,r,t,r,b,l,b]])

            gt_instances = {
                'bboxes': gt_bboxes.to(DEVICE),
                'labels': gt_labels.to(DEVICE),
                'kps': gt_kps.to(DEVICE)
            }
            gt_instances_sn.append(SimpleNamespace(**gt_instances))
        yield images, [img_meta]*batch_size, gt_instances_sn


def get_args():
    parser = argparse.ArgumentParser(description='Trains the network')
    parser.add_argument('--load', required=False, help='Path to checkpoint model')
    parser.add_argument('--version', type=str, required=True, help='Training Version')
    parser.add_argument('--dataset_path', required=True, help='Dataset path. Ex: /path/ccpd2019/')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--start_lr', type=float, default=1e-3, help='Starting learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Starting learning rate')
    parser.add_argument('--num_workers', default=8, type=int, help='Dataloader workers')
    parser.add_argument('--start_epoch', default=0, type=int, help='Epoch to start training')
    parser.add_argument('--end_epoch', default=50, type=int, help='Epoch to end training')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--wandb', action='store_true', help='Log to wandb')
    parser.add_argument('--debug', action='store_true', help='Use debug dataloader')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    test_cfg = {
            'score_thr': 0.5,
            'nms': {
                'type': 'nms',
                'iou_threshold': 0.5
            },
    }

    model = Detector('mobilenetv4_hybrid_medium.e500_r224_in1k', 1, test_cfg, use_kps=True)
    if args.load:
        state_dict = torch.load(args.load, map_location=torch.device(DEVICE))
        # state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    summary(model, input_size=(2, 3, 224, 224))
    
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.start_lr, weight_decay=0.01)
    batch_size = args.batch_size if not args.debug else 4

    model_data_config = resolve_model_data_config(model.backbone)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=model_data_config['mean'],
            std=model_data_config['std']
        )
    ])

    if not args.debug:
        train_dataset = ALPRDataset(
            os.path.join(args.dataset_path, 'ccpd_base', 'alpr_annotation.csv'),
            os.path.join(args.dataset_path, 'ccpd_base'),
            data_transform,
            detection=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, num_workers=args.num_workers, drop_last=True)
        
        val_dataset = ALPRDataset(
            os.path.join(args.dataset_path, 'ccpd_weather', 'alpr_annotation.csv'),
            os.path.join(args.dataset_path, 'ccpd_weather'),
            data_transform,
            detection=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, num_workers=args.num_workers, drop_last=True)
    else:
        train_dataloader = dummy_dataloader
        val_dataloader = dummy_dataloader

    os.makedirs('model_bin', exist_ok=True)


    # Train the model using the external train_model function
    train_model(
        project='alpr_detection_v4',
        model=model,
        train_function=train_detection,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=DEVICE,
        vocab=None,  # Not needed for detection but required by function signature
        model_name=f'detection_{args.version}',
        num_epochs=args.end_epoch,
        use_wandb=args.wandb,
        config=args,
        version=args.version,
        start_lr=args.start_lr,
        min_lr=args.min_lr,
        compile=args.compile
    )
