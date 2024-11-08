import torch
from torch import nn
import timm
from timm.data.config import resolve_model_data_config

from head import GFLHead
from bifpn import BiFpn
from model_config import get_efficientdet_config


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Detector(nn.Module):

    def __init__(self, backbone: str, num_classes: int, test_cfg: dict, use_kps: bool=False):
        super(Detector, self).__init__()
        self.test_cfg = test_cfg
        
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3)).to(DEVICE)

        model_data_config = resolve_model_data_config(self.backbone)
        self.model_data_config = model_data_config
        self.input_size = model_data_config['input_size']

        fpn_config = get_efficientdet_config()
        fpn_config.image_size = model_data_config['input_size'][1:]
        feat_info = self.backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        self.fpn = BiFpn(fpn_config, feat_info).to(DEVICE)
        self.head = GFLHead(num_classes, fpn_config.fpn_channels, use_kps=use_kps).to(DEVICE)
        
    def forward(self, x: torch.Tensor, test_cfg: dict=None):
        feats_bb = self.backbone(x)
        feats_neck = self.fpn(feats_bb)

        cls_quality_score, bbox_pred, kps_pred = self.head.forward(feats_neck)

        return cls_quality_score, bbox_pred, kps_pred
