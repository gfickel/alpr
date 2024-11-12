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
import wandb
import timm
from timm.data.config import resolve_model_data_config

from utils import *
from detector import Detector
from dataloader import ALPRDataset



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_detection(model, train_dataloader, val_dataloader, device, vocab=None, 
                   num_epochs=50, start_epoch=0, temp_model_path=None,
                   use_wandb=False, start_lr=1e-3, min_lr=1e-5, version=None, compile=False, **kwargs):
    """
    Main training function for detector model
    """
    if compile:
        model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=0.01)
    
    # Load optimizer state if resuming training
    if temp_model_path and os.path.exists(temp_model_path):
        checkpoint = torch.load(temp_model_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_lr = checkpoint['learning_rate']
        loss_history = checkpoint.get('loss_history', [])
    else:
        curr_lr = start_lr
        loss_history = []

    LOSS_LEN = 400
    loss_deque = deque(maxlen=LOSS_LEN)
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for i, data in enumerate(train_dataloader):
            sum_loss = 0
            begin = time.time()
            log = {}
            
            images, gt_bboxes, gt_labels, gt_kps, _ = data
            batch_size = images.size(0)
            
            gt_instances_sn = []
            for j in range(batch_size):
                gt_instances = {
                    'bboxes': gt_bboxes[j].to(device),
                    'labels': gt_labels[j].to(device),
                    'kps': gt_kps[j].to(device)
                }
                gt_instances_sn.append(SimpleNamespace(**gt_instances))

            img_metas = [{
                'pad_shape': (224, 224, 3),
                'img_shape': (224, 224, 3),
            }]*batch_size

            optimizer.zero_grad()

            cls_quality_score, bbox_pred, kps_pred = model(images.to(device))
            
            loss = model.head.loss_by_feat(
                cls_quality_score,
                bbox_pred,
                kps_pred,
                gt_instances_sn,
                img_metas)

            for loss_name, loss_val in loss.items():
                for loss_idx, lv in enumerate(loss_val):
                    lv.backward(retain_graph=True)
                    log[f'{loss_name}-{loss_idx}'] = float(lv.detach())
                    mult = 1 if 'kps' not in loss_name else 1/100
                    sum_loss += log[f'{loss_name}-{loss_idx}']*mult

            train_losses.append(sum_loss)
            loss_history.append(sum_loss)
            optimizer.step()

            if is_in_plateau(loss_history, threshold=500):
                # Reduce learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    curr_lr = param_group['lr']
                loss_history = []
                print(f"{version} - Learning rate reduced to {curr_lr}")

            if use_wandb:
                loss_deque.append(sum_loss)
                log['iter_time'] = time.time()-begin
                log['epoch'] = epoch+i/len(train_dataloader)
                log['learning_rate'] = curr_lr
                wandb.log(log)
                if len(loss_deque) == LOSS_LEN:
                    wandb.log({'train_loss': np.mean(sum_loss)})

        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data in val_dataloader:
                images, gt_bboxes, gt_labels, gt_kps, _ = data
                batch_size = images.size(0)
                
                gt_instances_sn = []
                for j in range(batch_size):
                    gt_instances = {
                        'bboxes': gt_bboxes[j].to(device),
                        'labels': gt_labels[j].to(device),
                        'kps': gt_kps[j].to(device)
                    }
                    gt_instances_sn.append(SimpleNamespace(**gt_instances))

                img_metas = [{
                    'pad_shape': (224, 224, 3),
                    'img_shape': (224, 224, 3),
                }]*batch_size

                cls_quality_score, bbox_pred, kps_pred = model(images.to(device))
                
                loss = model.head.loss_by_feat(
                    cls_quality_score,
                    bbox_pred,
                    kps_pred,
                    gt_instances_sn,
                    img_metas)
                
                sum_loss = 0
                for loss_name, loss_val in loss.items():
                    for loss_idx, lv in enumerate(loss_val):
                        mult = 1 if 'kps' not in loss_name else 1/100
                        sum_loss += float(lv.detach())*mult
                
                val_losses.append(sum_loss)

        avg_val_loss = np.mean(val_losses)
        if use_wandb:
            wandb.log({
                'val_loss': avg_val_loss,
                'epoch': epoch
            })
        
        # Save checkpoint
        if temp_model_path:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'learning_rate': curr_lr,
                'loss_history': loss_history
            }
            torch.save(checkpoint, temp_model_path)
        
        print(f'Epoch {epoch}: Train Loss = {np.mean(train_losses):.4f}, Val Loss = {avg_val_loss:.4f}')
        
        if curr_lr < min_lr:
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
    parser.add_argument('--dataset_path', required=True, help='Dataset path. Ex: /path/ccpd2019/ccpd_base/')
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
    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.start_lr, weight_decay=0.01)
    batch_size = args.batch_size if not args.debug else 4

    model_data_config = resolve_model_data_config(model.backbone)
    data_transform = timm.data.create_transform(
        input_size=model_data_config['input_size'],
        mean=model_data_config['mean'],
        std=model_data_config['std'],
        interpolation=model_data_config['interpolation'],
        crop_mode='squash',
        crop_pct=1,
    )

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
        project='alpr_detection_v4'
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