import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
import timm
from timm.data.config import resolve_model_data_config
from PIL import Image


from utils import *
from maskocr import MaskOCR
from detector import Detector

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Image.MAX_IMAGE_PIXELS = None

class ALPR(nn.Module):
    """
    ALPR model that combines detection and OCR.
    """
    def __init__(self, detector_model_path: str, ocr_model_path: str, ocr_config: str):
        super().__init__()
        
        # Load detection model
        test_cfg = {
            'score_thr': 0.25,
            'nms': {
                'type': 'nms',
                'iou_threshold': 0.5
            },
        }

        self.detection_model = Detector('mobilenetv4_hybrid_medium.e500_r224_in1k', 1, test_cfg, use_kps=True)
        state_dict = torch.load(detector_model_path, map_location=DEVICE)
        self.detection_model.load_state_dict(state_dict['model_state_dict'])
        self.detection_model.to(DEVICE)
        self.detection_model.eval()

        # Load OCR model
        self.vocab = FULL_ALPHABET
        vocab_size = len(self.vocab) + 1
        with open(ocr_config, 'r') as fid:
            model_cfg = json.load(fid)

        self.ocr_model = MaskOCR(
            model_cfg['img_height'], model_cfg['img_width'], 
            model_cfg['patch_size'], model_cfg['embed_dim'],
            model_cfg['num_heads'], model_cfg['num_encoder_layers'], 
            model_cfg['num_decoder_layers'], vocab_size, 
            model_cfg['max_sequence_length'], 
            dropout=model_cfg['dropout'],
            emb_dropout=model_cfg['emb_dropout'], 
            overlap=model_cfg['overlap']
        )
        
        state_dict_ocr = torch.load(ocr_model_path, map_location=DEVICE)
        self.ocr_model.load_state_dict(state_dict_ocr, strict=False)
        self.ocr_model.to(DEVICE)
        self.ocr_model.eval()

        # Setup transforms
        self.ocr_data_transform = transforms.Compose([
            transforms.Resize((model_cfg['img_height'], model_cfg['img_width'])),
            transforms.ToTensor(),
        ])

        model_data_config = resolve_model_data_config(self.detection_model.backbone)
        self.detection_data_transform = timm.data.create_transform(
            input_size=model_data_config['input_size'],
            mean=model_data_config['mean'],
            std=model_data_config['std'],
            interpolation=model_data_config['interpolation'],
            crop_mode='squash',
            crop_pct=1,
        )

    def forward(self, image: Image):
        """Forward pass of the model.
        
        Returns:
            bbox_res List[Tuple[int,int,int,int]]: list of boxes, each one represented as left,top,right,bottom
            ocr_res List[str]: OCRs list
            kps_res List[Tuple[KPS]]: List of keypoints, each one as ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
        """
        with torch.no_grad():
            # Run detection
            detection_image = self.detection_data_transform(image).unsqueeze(0).to(DEVICE)
            
            scores, bbox_preds, kps_preds = self.detection_model(detection_image, image.size[::-1])

            # Process each detection
            ocr_res, bbox_res, kps_res = [], [], []
            for i in range(bbox_preds.shape[0]):
                bbox_np = bbox_preds[i].cpu().numpy()
                l, t, r, b = bbox_np[0], bbox_np[1], bbox_np[2], bbox_np[3]

                evens = kps_preds[i][::2].tolist()  # [x1, x2, x3, x4]
                odds = kps_preds[i][1::2].tolist()   # [y1, y2, y3, y4]
                kps_res.append(list(zip(evens, odds)) ) # ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
                                
                # Extract region and run OCR
                region = image.crop((l, t, r, b))
                ocr_image = self.ocr_data_transform(region).unsqueeze(0).to(DEVICE)
                ocrs_logits = self.ocr_model(ocr_image)
                log_ocrs = torch.nn.functional.log_softmax(ocrs_logits, dim=2)
                ocrs_pred = torch.argmax(log_ocrs, dim=-1)
                ocr_text = indices_to_text(ocrs_pred[0].cpu(), self.vocab)

                ocr_res.append(ocr_text)
                bbox_res.append((int(l), int(t), int(r), int(b)))

            return bbox_res, ocr_res, kps_res

    def run_im_path(self, img_path: str):
        """Run inference on an image file.
        Returns:
            bbox_res List[Tuple[int,int,int,int]]: list of boxes, each one represented as left,top,right,bottom
            ocr_res List[str]: OCRs list
            kps_res List[Tuple[KPS]]: List of keypoints, each one as ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
        """
        image = Image.open(img_path).convert('RGB')
        return self.forward(image)


