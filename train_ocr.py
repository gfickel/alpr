from types import SimpleNamespace
from collections import deque
import os
import argparse

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
from detector import Detector, CRNN
from dataloader import ALPRDataset

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data_transform(input_size, mean, std):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.float() / 255.0),  # Convert to float and scale to [0, 1]
        transforms.Resize(input_size),
        transforms.Normalize(mean=mean, std=std)
    ])

def set_requires_grad(modules, requires_grad):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = requires_grad

def update_learning(epoch, learning_rates, param_group_names, optimizer, model, first_update):
    # Define which parts of the model to train at each epoch range
    training_schedule = [
        {'train_backbone': False, 'train_head': True, 'train_fpn': True, 'train_ocr': True},  # Epoch 0-2
        {'train_backbone': True, 'train_head': True, 'train_fpn': True, 'train_ocr': True},  # Epoch 3-7
        {'train_backbone': False, 'train_head': False, 'train_fpn': False, 'train_ocr': True},  # Epoch >= 10
    ]

    # Select the current stage based on the epoch
    if epoch <= 2:
        stage = 0
    elif epoch <= 7:
        stage = 1
    else:
        stage = 2

    # Set requires_grad for model parts based on the selected stage
    set_requires_grad([model.backbone], training_schedule[stage]['train_backbone'])
    set_requires_grad([model.head], training_schedule[stage]['train_head'])
    set_requires_grad([model.fpn], training_schedule[stage]['train_fpn'])
    set_requires_grad([model.ocr], training_schedule[stage]['train_ocr'])

    # Set learning rates for each parameter group
    if first_update:
        param_name_idx = 1 if args.compile else 0
        for name, param_group in zip(param_group_names, optimizer.param_groups):
            group_name = name.split('.')[param_name_idx]  # Adjust the splitting logic based on your param_group_names
            param_group['lr'] = learning_rates[group_name][stage]

    # Determine what to train
    train_bbox = training_schedule[stage]['train_backbone'] or training_schedule[stage]['train_head'] or training_schedule[stage]['train_fpn']
    train_ocr = training_schedule[stage]['train_ocr']

    return train_bbox, train_ocr


def get_args():
    parser = argparse.ArgumentParser(description='Trains the network')
    parser.add_argument('--load', required=False, help='Path to checkpoint model')
    parser.add_argument('--gpu_augmentations', action='store_true', help='Run augmentatino on GPU')
    parser.add_argument('--crnn', action='store_true', help='Vanilla CRNN')
    parser.add_argument('--backbone', type=str, help='Timm model')
    parser.add_argument('--train_version', type=str, required=True, help='Model version')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--num_bilstm', default=2, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='Dataloader workers')
    parser.add_argument('--start_epoch', default=0, type=int, help='Epoch to start training')
    parser.add_argument('--end_epoch', default=50, type=int, help='Epoch to end training')
    parser.add_argument('--crnn_embedding', default=64, type=int, help='CRNN LSTM embedding')
    parser.add_argument('--load_ignoring_ocr', action='store_true', help='Ignore OCR weights when loading model')
    parser.add_argument('--max_pool_width', action='store_true', help='Apply max pool width on backbone features before CRNN')
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

    if args.crnn:
        model = CRNN(3, len(FULL_ALPHABET)+1, args.crnn_embedding)
    else:
        model = Detector(args.backbone, 1, test_cfg, use_kps=True, use_ocr=True, alphabet_size=len(FULL_ALPHABET)+1,
                        crnn_embedding=args.crnn_embedding, max_pool_width=args.max_pool_width, num_bilstm=args.num_bilstm)
    if args.load:
        state_dict = torch.load(args.load, map_location=torch.device(DEVICE))
        try:
            if args.load_ignoring_ocr:
                state_dict = {k:v for k,v in state_dict.items() if 'ocr' not in k}
            model.load_state_dict(state_dict)#, strict=False)
        except:
            state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}
            if args.load_ignoring_ocr:
                state_dict = {k:v for k,v in state_dict.items() if 'ocr' not in k}
            model.load_state_dict(state_dict)#, strict=False)
    

    if args.crnn:
        model_data_config = {
            'input_size': (3, 32, 128),
            'mean': (0.5, 0.5, 0.5),
            'std': (1, 1, 1),
            'interpolation': 'bilinear',
        }
    else:
        model_data_config = resolve_model_data_config(model.backbone)

    summary(model, input_size=(2, *model_data_config['input_size']))
    model.to(DEVICE)
    if args.compile:
        model = torch.compile(model)

    learning_rates = {
        'backbone': [1e-4,1e-4,1e-4,1e-4],
        'head': [1e-3,1e-3,1e-4,1e-4],
        'fpn': [1e-3,1e-3,1e-4,1e-4],
        'ocr': [1e-3,1e-3,1e-4,1e-4],
        'cnn': [1e-3,1e-3,1e-3,1e-3],
        'rnn': [1e-3,1e-3,1e-3,1e-3],
    }

    # Build param_group where each group consists of a single parameter.
    # `param_group_names` is created so we can keep track of which param_group
    # corresponds to which parameter.
    param_groups = []
    param_group_names = []
    param_name_idx = 1 if args.compile else 0
    # for name, parameter in model.named_parameters():
    #     print(name)
    #     param_group_names.append(name)
    #     name = name.split('.')[param_name_idx]
    #     param_groups.append({'params': [parameter], 'lr': learning_rates[name]})
    
    # optimizer = torch.optim.AdamW(param_groups, lr=0.001, weight_decay=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=400, verbose=True)
    batch_size = args.batch_size if not args.debug else 4

    # data_transform = get_data_transform(
    #     input_size=model_data_config['input_size'][1:],
    #     mean=model_data_config['mean'],
    #     std=model_data_config['std'])

    data_transform = timm.data.create_transform(
        input_size=model_data_config['input_size'],
        mean=model_data_config['mean'],
        std=model_data_config['std'],
        interpolation=model_data_config['interpolation'],
        crop_mode='squash',
        crop_pct=1,
    )

    dataset = ALPRDataset(
        '../output_images_plates_gt/alpr_annotation.csv',
        '../output_images_plates_gt/',
        None if args.gpu_augmentations else data_transform,
        resize=model_data_config['input_size'][1:])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=True)

    os.makedirs('model_bin', exist_ok=True)
    torch.save(model.state_dict(), f'model_bin/ocr_alpr_{args.train_version}_{0}.pth')

    running_loss = 0.
    ctc_loss = torch.nn.CTCLoss()
    T = 28
    input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long)
    target_lengths = torch.full(size=(batch_size,), fill_value=7, dtype=torch.long)

    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            project="ocr-v4",
            config={
                'architecture': args.backbone,
                'crnn_embedding': args.crnn_embedding,
                'batch_size': args.batch_size,
                'compile': args.compile,
                'num_workers': args.num_workers,
                'num_bilstm': args.num_bilstm,
            }
        )

    first_update = True
    LOSS_LEN = 200
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.start_epoch,args.end_epoch+1,1):
            if not args.crnn:
                train_bbox, train_ocr = update_learning(epoch, learning_rates, param_group_names, optimizer, model, first_update)
            first_update = False
            loss_deque = deque(maxlen=LOSS_LEN)
            for i, data in enumerate(dataloader):#enumerate(dummy_dataloader(5000, batch_size)):
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

                if args.gpu_augmentations:
                    images = data_transform(images.to(DEVICE))
                if args.crnn:
                    ocrs = model.forward(images.to(DEVICE))
                else:
                    ocrs = model.forward_ocr(images.to(DEVICE), test_cfg=test_cfg)
                
                target_ocrs_list = [gt_instances_sn[x].ocrs for x in range(len(gt_instances_sn))]
                target_ocrs = torch.stack(target_ocrs_list, dim=0)
                ocr_loss = None
                log_probs = torch.nn.functional.log_softmax(ocrs, dim=2)
                ocr_loss = ctc_loss(log_probs, target_ocrs, input_lengths, target_lengths)
                ocr_loss.backward(retain_graph=True)
                optimizer.step()
                log['ocr_ctc'] = float(ocr_loss.detach())
                sum_loss += log['ocr_ctc']

                decoded_preds = decode_ocr(ocrs)
                target_ocrs_decoded = [target.tolist() for target in target_ocrs]
                cer, wer = calculate_metrics(decoded_preds, target_ocrs_decoded)

                log['ocr_cer'] = cer
                log['ocr_wer'] = wer
        
                loss_deque.append(sum_loss)
                # Extract learning rates and add to log
                for name, param_group in zip(param_group_names, optimizer.param_groups):
                    group_name = name.split('.')[param_name_idx]  # Adjust the splitting logic based on your param_group_names
                    log[f'lr_{group_name}'] = param_group['lr']

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # or some other value
                
                if len(loss_deque) == LOSS_LEN:
                    scheduler.step(np.mean(loss_deque))
                    if args.wandb: wandb.log({'loss': np.mean(loss_deque)})

                log['iter_time'] = time.time()-begin
                log['epoch'] = epoch+i/len(dataloader)
                if args.wandb: wandb.log(log)

            torch.save(model.state_dict(), f'model_bin/alpr_ocr_{args.train_version}_{epoch+1}.pth')
            