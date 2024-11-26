from typing import Tuple

import torch
from torch import nn
import timm
from timm.data.config import resolve_model_data_config

from head import GFLHead
from bifpn import BiFpn
from model_config import get_efficientdet_config
from utils import distance2kps, distance2bbox


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Detector(nn.Module):

    def __init__(self, backbone: str, num_classes: int, test_cfg: dict, use_kps: bool=False, zero_weights: bool=False):
        super(Detector, self).__init__()
        self.test_cfg = test_cfg
        
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3)).to(DEVICE)

        model_data_config = resolve_model_data_config(self.backbone)
        print(model_data_config)
        self.model_data_config = model_data_config
        self.input_size = model_data_config['input_size']

        fpn_config = get_efficientdet_config()
        fpn_config.image_size = model_data_config['input_size'][1:]
        feat_info = self.backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        self.fpn = BiFpn(fpn_config, feat_info).to(DEVICE)
        self.head = GFLHead(num_classes, fpn_config.fpn_channels, use_kps=use_kps, zero_weights=zero_weights).to(DEVICE)
        
    def forward(self, x: torch.Tensor, input_shape: Tuple[int, int]=None):
        feats_bb = self.backbone(x)
        feats_neck = self.fpn(feats_bb)

        cls_quality_score, bbox_pred, kps_pred = self.head.forward(feats_neck)

        if input_shape is None:
            return cls_quality_score, bbox_pred, kps_pred

        res = self.head.predict_by_feat(
                cls_quality_score, bbox_pred, kps_pred, cfg=self.test_cfg, rescale=True)
        

        scores = res[0]['scores']
        bboxes = res[0]['bboxes']
        kps = res[0]['kps']

        bboxes_scaled, kps_scaled = self.rescale_detections(
            bboxes, kps, input_shape, x.shape[-2:])
        
        return scores, bboxes_scaled, kps_scaled
    
    def rescale_detections(self, bboxes, keypoints, original_shape, model_shape):
        scale_h = original_shape[0] / model_shape[0]
        scale_w = original_shape[1] / model_shape[1]
        
        # Rescale bboxes [x1,y1,x2,y2]
        bboxes[:, [0,2]] *= scale_w  # rescale x coordinates
        bboxes[:, [1,3]] *= scale_h  # rescale y coordinates
        
        # Rescale keypoints [x1,y1,x2,y2,x3,y3,x4,y4]
        keypoints[:, [0,2,4,6]] *= scale_w  # rescale x coordinates
        keypoints[:, [1,3,5,7]] *= scale_h  # rescale y coordinates
        
        return bboxes, keypoints