from types import SimpleNamespace
from collections import deque
import os
import argparse

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
                'kps': gt_kps.to(DEVICE),
                'ocrs': torch.LongTensor([1,3,2,3,2]).to(DEVICE)
            }
            gt_instances_sn.append(SimpleNamespace(**gt_instances))
        yield images, [img_meta]*batch_size, gt_instances_sn


def get_args():
    parser = argparse.ArgumentParser(description='Trains the network')
    parser.add_argument('--load', required=False, help='Path to checkpoint model')
    parser.add_argument('--dataset_path', required=True, help='Dataset path. Ex: /path/ccpd2019/ccpd_base/')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
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

    model = Detector('mobilenetv3_large_100', 1, test_cfg, use_kps=True, use_ocr=True)
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = False
    for param in model.fpn.parameters():
        param.requires_grad = False

    if args.load:
        state_dict = torch.load(args.load, map_location=torch.device(DEVICE))
        # state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    summary(model, input_size=(2, 3, 224, 224))
    
    model.to(DEVICE)
    if args.compile:
        model = torch.compile(model)

    # Build param_group where each group consists of a single parameter.
    # `param_group_names` is created so we can keep track of which param_group
    # corresponds to which parameter.
    param_groups = []
    param_group_names = []
    param_name_idx = 1 if args.compile else 0
    for name, parameter in model.named_parameters():
        param_group_names.append(name)
        name = name.split('.')[param_name_idx]
        param_groups.append({'params': [parameter], 'lr': learning_rates[name]})
    
    optimizer = torch.optim.AdamW(param_groups, lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=400, verbose=True)
    batch_size = args.batch_size if not args.debug else 4

    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            project="detector-kps-ocr-v3",
            config={
                'learning_rate': 0.001,
                'architecture': 'mobilenetV3-large',
                'batch_size': args.batch_size,
                'compile': args.compile,
                'num_workers': args.num_workers,
            }
        )

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
        dataset = ALPRDataset(
            os.path.join(args.dataset_path, 'alpr_annotation.csv'),
            args.dataset_path,
            data_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=True)
    else:
        dataloader = dummy_dataloader

    os.makedirs('model_bin', exist_ok=True)
    torch.save(model.state_dict(), f'model_bin/alpr_v17_{0}.pth')

    LOSS_LEN = 400
    with torch.autograd.set_detect_anomaly(False):
        for epoch in range(args.start_epoch,args.end_epoch+1,1):
            loss_deque = deque(maxlen=LOSS_LEN)
            for i, data in enumerate(dataloader):
                sum_loss = 0
                begin = time.time()
                log = {}
                images, gt_bboxes, gt_labels, gt_kps, gt_ocrs = data
                gt_instances_sn = []
                for j in range(batch_size):
                    gt_instances = {
                        'bboxes': gt_bboxes[j].to(DEVICE),
                        'labels': gt_labels[j].to(DEVICE),
                        'kps': gt_kps[j].to(DEVICE),
                        'ocrs': gt_ocrs[j].to(DEVICE),
                    }
                    gt_instances_sn.append(SimpleNamespace(**gt_instances))

                img_metas = [{
                    'pad_shape': model_data_config['input_size'][::-1],
                    'img_shape': model_data_config['input_size'][::-1],
                }]*batch_size

                optimizer.zero_grad()

                cls_quality_score, bbox_pred, kps_pred = model(images.to(DEVICE), test_cfg=test_cfg)
                
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

                loss_deque.append(sum_loss)
                # Extract learning rates and add to log
                for name, param_group in zip(param_group_names, optimizer.param_groups):
                    group_name = name.split('.')[param_name_idx]  # Adjust the splitting logic based on your param_group_names
                    log[f'lr_{group_name}'] = param_group['lr']

                log['iter_time'] = time.time()-begin
                log['epoch'] = epoch+i/len(dataloader)
                if args.wandb: wandb.log(log)

                optimizer.step()

                if len(loss_deque) == LOSS_LEN:
                    scheduler.step(np.mean(sum_loss))
                    wandb.log({'loss': np.mean(sum_loss)})

            torch.save(model.state_dict(), f'model_bin/alpr_v17_{epoch+1}.pth')
            