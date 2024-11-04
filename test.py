from torchvision import transforms
import argparse
import time
import os
import json
import shutil
from types import SimpleNamespace   

import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import *
from maskocr import MaskOCR
from dataloader import ALPRDataset



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_args():
    parser = argparse.ArgumentParser(description='Converts ALPR datasets to test/train')
    parser.add_argument('--model_path', help='Model binary path')
    parser.add_argument('--model_config', help='Model config path')
    parser.add_argument('--debug_image_interval', default=0, type=int, help='One every N images to save on disk for debug')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # cfg = importlib.import_module(args.model_config)
    vocab = FULL_ALPHABET
    vocab_size = len(vocab)+1
    with open(args.model_config, 'r') as fid:
        model_cfg = json.load(fid)
        model_cfg['overlap'] = model_cfg.get('overlap', 0)
        cfg = SimpleNamespace(**{**model_cfg, **{'vocab_size': vocab_size}})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MaskOCR(cfg.img_height, cfg.img_width, cfg.patch_size, cfg.embed_dim, cfg.num_heads, cfg.num_encoder_layers,
                    cfg.num_decoder_layers, cfg.vocab_size, cfg.max_sequence_length, dropout=cfg.dropout, emb_dropout=cfg.emb_dropout,
                    overlap=cfg.overlap)

    # train_dataset = SyntheticOCRDataset(vocab, max_sequence_length, 50000)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    # val_dataset = SyntheticOCRDataset(vocab, max_sequence_length, batch_size*4)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    data_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandAugment(num_ops=3, magnitude=7),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img.float() / (255.0 if cfg.norm_image else 1.0)),
    ])
    
    state_dict = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()


    if cfg.img_height == 32:
        test_ds = 'plates_ccpd_challenge'
    elif cfg.img_height == 48:
        test_ds = 'plates_ccpd_challenge_48'

    dataset = ALPRDataset(
        f'../alpr_datasets/{test_ds}/alpr_annotation.csv',
        f'../alpr_datasets/{test_ds}/', #None)
        # '../alpr_datasets/CCPD2019/alpr_annotation.csv',
        # '../alpr_datasets/CCPD2019/ccpd_base/', #None)
        data_transform,
        return_image_path=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6, drop_last=True)

    os.makedirs('out_images', exist_ok=True)
    ocr_missing, ocr_errors, ocr_correct = 0, 0, 0
    runtimes = []
    for i, data in enumerate(dataloader):#enumerate(dummy_dataloader(5000, batch_size)):
        images, images_path, gt_ocr = data
        begin = time.time()
        ocrs_logits = model(images.to(DEVICE))
        runtimes.append(time.time()-begin)
        log_ocrs = torch.nn.functional.log_softmax(ocrs_logits.to('cpu'), dim=2)
        ocrs_pred = torch.argmax(log_ocrs, dim=-1)
        
        for im_idx in range(gt_ocr.shape[0]):
            if len(ocrs_pred[im_idx]) > 0:
                if torch.equal(ocrs_pred[im_idx][1:], gt_ocr[im_idx][1:]):
                    ocr_correct += 1
                    # print(f'Good: {ocrs_pred[im_idx]} | {gt_ocr[im_idx]}')
                else:
                    ocr_errors += 1
                    ocr_text = indices_to_text(ocrs_pred[0].detach().cpu(), vocab)
                    shutil.copy(images_path[im_idx], f'out_images/{ocr_text}.png')
                    # print(f'Error: {ocrs_pred[im_idx]} | {gt_ocr[im_idx]}')
            else:
                # print('Missing')
                ocr_missing += 1

        print(f'Acc: {ocr_correct/(ocr_correct+ocr_missing+ocr_errors)*100:.1f}% - {ocr_correct}|{ocr_correct+ocr_missing+ocr_errors}\t\t', end='\r')
        if i == 500:
            break

    print(f'Acc: {ocr_correct/(ocr_correct+ocr_missing+ocr_errors)*100:.1f}% - {ocr_correct}|{ocr_correct+ocr_missing+ocr_errors}\t\t')
    print(f'Runtime mean: {np.mean(runtimes)*1000:.2f}ms, std: {np.std(runtimes)*1000:.2}ms')
    print('\n\n')
