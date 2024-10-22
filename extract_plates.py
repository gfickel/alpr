import os
from PIL import Image
from tqdm import tqdm
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
    parser = argparse.ArgumentParser(description='Crops the licence plates')
    parser.add_argument('--model_path', help='Model binary path')
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
    model = Detector('mobilenetv3_large_100', 1, use_kps=True, use_ocr=False, test_cfg=test_cfg, alphabet_size=len(FULL_ALPHABET)+1, crnn_embedding=256, max_pool_width=False)
    state_dict = torch.load(args.model_path, map_location=torch.device(DEVICE))
    state_dict = {k:v for k,v in state_dict.items() if 'ocr' not in k}
    # state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}

    # model = torch.compile(model)
    model.to(DEVICE)
    try:
        model.load_state_dict(state_dict)
    except:
        state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}
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
        # '../ccpd_base/alpr_annotation.csv',
        # '../ccpd_base/', #None)
        '../alpr_datasets/CCPD2019/alpr_annotation.csv',
        '../alpr_datasets/CCPD2019/ccpd_base/', #None)
        data_transform,
        return_image_path=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=2, drop_last=True)


    os.makedirs('out_images_plates_gt', exist_ok=True)

    for data in tqdm(dataloader, desc="Processing Images", unit="image"):
        images, images_path, gt_bboxes, gt_labels, gt_kps, gt_ocr = data
        img_metas = [{
            'pad_shape': model_data_config['input_size'][::-1],
            'img_shape': model_data_config['input_size'][::-1],
        }]

        # cls_quality_score, bbox_pred, kps_pred, ocrs_logits, have_bbox = model(images.to(DEVICE))
        # res = model.head.predict_by_feat(
        #     cls_quality_score, bbox_pred, kps_pred,
        #     batch_img_metas=None,
        #     cfg=test_cfg, rescale=False)
        
        for im_idx in range(len(images_path)):
            image = Image.open(images_path[im_idx])
            W, H = image.size
            w_scale = W / model_data_config['input_size'][1]
            h_scale = H / model_data_config['input_size'][1]

            # for bbox_idx in range(res[im_idx]['bboxes'].shape[0]):
            #     bbox = res[im_idx]['bboxes'][bbox_idx].detach().tolist()
            #     bbox = [bbox[0]*w_scale, bbox[1]*h_scale, bbox[2]*w_scale, bbox[3]*h_scale]
            #     bbox = list(map(int, bbox))

            #     # Crop the image using the bounding box
            #     plate_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            #     # Save the cropped image in the out_images folder
            #     plate_image.save(f'out_images_plates/{os.path.basename(images_path[im_idx])}')

            for bbox_idx in range(gt_bboxes.shape[0]):
                bbox = gt_bboxes[bbox_idx].detach().tolist()[0]
                bbox = [bbox[0]*w_scale, bbox[1]*h_scale, bbox[2]*w_scale, bbox[3]*h_scale]
                bbox = list(map(int, bbox))

                # Crop the image using the bounding box
                plate_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                # Save the cropped image in the out_images folder
                plate_image.save(f'out_images_plates_gt/{os.path.basename(images_path[im_idx])}')

    print('Car plates have been extracted and saved.')
