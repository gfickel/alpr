from types import SimpleNamespace
import argparse
import os

import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import timm
from timm.data.config import resolve_model_data_config

from utils import *
from detector import Detector
from dataloader import ALPRDataset



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_args():
    parser = argparse.ArgumentParser(description='Converts ALPR datasets to test/train')
    parser.add_argument('--model_path', help='Model binary path')
    parser.add_argument('--debug_image_interval', default=0, type=int, help='One every N images to save on disk for debug')
    parser.add_argument('--dataset_name', choices=['ccpd2019'], default='ccpd2019')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    print(f'Using device {DEVICE}')

    test_cfg = {
            'score_thr': 0.5,
            'nms': {
                'type': 'nms',
                'iou_threshold': 0.5
            },
    }
    model = Detector('mobilenetv3_large_100', 1, use_kps=True, use_ocr=True, test_cfg=test_cfg, alphabet_size=len(FULL_ALPHABET)+1, crnn_embedding=256, max_pool_width=False)
    state_dict = torch.load(args.model_path, map_location=torch.device(DEVICE))
    # state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}

    # model = torch.compile(model)
    model.to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()



    model_data_config = resolve_model_data_config(model.backbone)
    data_transform = timm.data.create_transform(
        input_size=model_data_config['input_size'],
        mean=model_data_config['mean'],
        std=model_data_config['std'],
        interpolation=model_data_config['interpolation'],
        crop_mode='squash',
    )

    dataset = ALPRDataset(
        # '../ccpd_base/alpr_annotation_test.csv',
        # '../ccpd_base/', #None)
        '../alpr_datasets/CCPD2019/alpr_annotation.csv',
        '../alpr_datasets/CCPD2019/ccpd_base/', #None)
        data_transform,
        return_image_path=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2, drop_last=True)

    os.makedirs('out_images', exist_ok=True)
    ocr_missing, ocr_errors, ocr_correct = 0, 0, 0
    for i, data in enumerate(dataloader):#enumerate(dummy_dataloader(5000, batch_size)):
        images, images_path, gt_bboxes, gt_labels, gt_kps, gt_ocr = data
        img_metas = [{
            'pad_shape': model_data_config['input_size'][::-1],
            'img_shape': model_data_config['input_size'][::-1],
        }]

        
        cls_quality_score, bbox_pred, kps_pred, ocrs_logits, have_bbox = model(images.to(DEVICE))
        res = model.head.predict_by_feat(
            cls_quality_score, bbox_pred, kps_pred,
            batch_img_metas=None,
            cfg=test_cfg, rescale=False)
        
        ocrs_pred = decode_ocr(ocrs_logits)
        for im_idx in range(gt_ocr.shape[0]):
            if len(ocrs_pred[im_idx]) > 0:
                if ocrs_pred[im_idx] == gt_ocr[im_idx]:
                    ocr_correct += 1
                    # print(f'Good: {ocrs_pred[im_idx]} | {gt_ocr[im_idx]}')
                else:
                    ocr_errors += 1
                    print(f'Error: {ocrs_pred[im_idx]} | {gt_ocr[im_idx]}')
            else:
                # print('Missing')
                ocr_missing += 1

        if args.debug_image_interval > 0 and i%args.debug_image_interval == 0:
            image = Image.open(images_path[0])
            W, H = image.size
            w_scale = W/model_data_config['input_size'][1]
            h_scale = H/model_data_config['input_size'][1]
            draw = ImageDraw.Draw(image)
            
            for bbox_idx in range(res[0]['bboxes'].shape[0]):
                bbox = res[0]['bboxes'][bbox_idx].detach().tolist()
                bbox = [bbox[0]*w_scale, bbox[1]*h_scale, bbox[2]*w_scale, bbox[3]*h_scale]
                bbox = list(map(int, bbox))

                draw.rectangle(((bbox[0],bbox[1]), (bbox[2], bbox[3])), outline='red', width=4)

            print('\n', gt_bboxes.shape)
            bbox = [gt_bboxes[0,0,0]*w_scale, gt_bboxes[0,0,1]*h_scale, gt_bboxes[0,0,2]*w_scale, gt_bboxes[0,0,3]*h_scale]
            bbox = list(map(int, bbox))

            draw.rectangle(((bbox[0],bbox[1]), (bbox[2], bbox[3])), outline='blue', width=4)

            if res[0]['kps'].numel() > 0:
                for pt_idx in range(0, res[0]['kps'][0].shape[0], 2):
                    x, y = res[0]['kps'][0][pt_idx].detach(), res[0]['kps'][0][pt_idx+1].detach()
                    draw.ellipse((int(x*w_scale)-5,int(y*h_scale)-5,int(x*w_scale)+5,int(y*h_scale)+5), fill='green', outline='blue')

            image.save(f'out_images/{os.path.basename(images_path[0])}')

        print(f'Acc: {ocr_correct/(ocr_correct+ocr_missing+ocr_errors)*100:.1f}% - {ocr_correct}|{ocr_correct+ocr_missing+ocr_errors}\t\t', end='\r')
    print('\n\n')