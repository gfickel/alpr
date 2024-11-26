# Code from OpenMMLab project, mostly from https://github.com/open-mmlab/mmdetection

from types import SimpleNamespace 
from typing import Union, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import *
from losses import GIoULoss, DistributionFocalLoss, quality_focal_loss, SmoothL1Loss
from anchor_generator import AnchorGenerator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
            torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x



class GFLHead(nn.Module):
    """Anchor Head with Generalized Focal Loss

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=64,
                 stacked_convs=4,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_max=16,
                 use_kps=False,
                 zero_weights=False):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        self.feat_channels = feat_channels
        self.cls_out_channels = num_classes
        self.use_kps = use_kps

        self.prior_generator = AnchorGenerator(
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]
        )

        self.NK = 4
        self.integral = Integral(self.reg_max)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.giou_loss = GIoULoss()
        self.loss_cls = quality_focal_loss
        self.loss_kps = SmoothL1Loss(beta=0.1111111111111111, loss_weight=0.1)
        self._init_layers(zero_weights)

    @property
    def num_anchors(self) -> int:
        return self.prior_generator.num_base_priors[0]

    def _init_layers(self, zero_weights):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                self._conv_module(
                    chn, self.feat_channels, self.norm_cfg)
                )
            self.reg_convs.append(
                self._conv_module(
                    chn, self.feat_channels, self.norm_cfg)
                )
            if self.use_kps:
                self.stride_kps = nn.Conv2d(
                    self.feat_channels, self.NK*2*self.num_anchors, 3, padding=1)


        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])
        
        if zero_weights:
            # Initialize all conv layers weights to zero
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


    def _conv_module(self, in_channels: int,
                     out_channels: int,
                     norm_cfg: dict=None):
        if norm_cfg is None:
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
                nn.ReLU()
            )
        elif norm_cfg['type'] == 'BN':
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(),
                nn.ReLU()
            )
        else:  # norm_cfg['type'] == 'GN':
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.GroupNorm(norm_cfg['num_groups'], out_channels),
                nn.ReLU()
            )

        return conv


    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    batch_img_metas: List[dict],
                    device: Union[torch.device, str] = 'cuda') \
            -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors.
                Defaults to cuda.

        Returns:
            tuple:

                - anchor_list (list[list[Tensor]]): Anchors of each image.
                - valid_flag_list (list[list[Tensor]]): Valid flags of each
                  image.
        """
        num_imgs = len(batch_img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        cls_scores, bbox_preds, kps_preds = [], [], []
        for idx, (f, s) in enumerate(zip(feats, self.scales)):
            score, bbox, kps = self.forward_single(f, s)
            cls_scores.append(score)
            bbox_preds.append(bbox)
            kps_preds.append(kps)

        return cls_scores, bbox_preds, kps_preds

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.gfl_cls(cls_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()

        if self.use_kps:
            kps_pred = self.stride_kps(reg_feat)
        else:
            kps_pred = bbox_pred.new_zeros( (bbox_pred.shape[0], self.NK*2, bbox_pred.shape[2], bbox_pred.shape[3]), requires_grad=True )
        if torch.isnan(bbox_pred).any():
            print('bbox nan')
        return cls_score, bbox_pred, kps_pred


    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)


    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, kps_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            kps_targets: Tensor, kps_weights: Tensor,
                            stride: Tuple[int], avg_factor: int) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            stride (Tuple[int]): Stride in this scale level.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        kps_pred = kps_pred.permute(0, 2, 3,
                                        1).reshape(-1, self.NK*2)
        kps_targets = kps_targets.reshape( (-1, self.NK*2) )
        kps_weights = kps_weights.reshape( (-1, self.NK*2) )

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(pos_anchor_centers,
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)


            pos_kps_targets = kps_targets[pos_inds]
            pos_kps_pred = kps_pred[pos_inds]
            pos_kps_weights = kps_weights.max(dim=1)[0][pos_inds] * weight_targets
            pos_kps_weights = pos_kps_weights.reshape( (-1, 1) )
            
            pos_decode_kps_targets = kps2distance(pos_anchor_centers, pos_kps_targets / stride[0])
            pos_decode_kps_pred = pos_kps_pred

            # regression loss
            loss_bbox = self.giou_loss(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)
            
            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)


            loss_kps = self.loss_kps(
                pos_decode_kps_pred,
                pos_decode_kps_targets,
                weight=200,#pos_kps_weights,
                avg_factor=1.0)
                
            kps_tmp = distance2kps(pos_anchor_centers, pos_decode_kps_pred) * stride[0]
            kps_gt_tmp = distance2kps(pos_anchor_centers, pos_decode_kps_targets) * stride[0]

            if torch.isnan(loss_dfl) or torch.isnan(loss_bbox):
                print('nan')
        else:
            loss_bbox = torch.zeros(1, requires_grad=True).to(DEVICE)
            loss_dfl = torch.zeros(1, requires_grad=True).to(DEVICE)
            loss_kps = torch.zeros(1, requires_grad=True).to(DEVICE)
            weight_targets = bbox_pred.new_tensor(1)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            # weight=label_weights,
            avg_factor=avg_factor)

        return loss_cls, loss_bbox, loss_dfl, loss_kps, weight_targets.sum()

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            kps_preds: List[Tensor],
            batch_gt_instances: List[SimpleNamespace],
            batch_img_metas: List[dict],
            batch_gt_instances_ignore = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, keypoints_targets_list, keypoints_weights_list, avg_factor) = cls_reg_targets

        # avg_factor = reduce_mean(
        #     torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, losses_dfl, losses_kps, \
            avg_factor = multi_apply(
                self.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                kps_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                keypoints_targets_list,
                keypoints_weights_list,
                self.prior_generator.strides,
                avg_factor=avg_factor)

        avg_factor = sum(avg_factor)/len(avg_factor)
        # avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        losses_kps = list(map(lambda x: x / avg_factor, losses_kps))
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl, loss_kps=losses_kps)

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples,
                rescale: bool = False) -> List[Tensor]:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        kps_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[dict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> List[Tensor]:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(cls_scores[0].shape[0]):
            img_meta = batch_img_metas[img_id] if batch_img_metas is not None else None
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            kps_pred_list = select_single_mlvl(
                kps_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                kps_pred_list=kps_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                kps_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: dict,
                                rescale: bool = False,
                                with_nms: bool = True):
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. GFL head does not need this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (:obj: `ConfigDict`): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
            is False and mlvl_score_factor is None, return mlvl_bboxes and
            mlvl_scores, else return mlvl_bboxes, mlvl_scores and
            mlvl_score_factor. Usually with_nms is False is used for aug
            test. If with_nms is True, then return the following format

            - det_bboxes (Tensor): Predicted bboxes with shape
              [num_bboxes, 5], where the first 4 columns are bounding
              box positions (tl_x, tl_y, br_x, br_y) and the 5-th
              column are scores between 0 and 1.
            - det_labels (Tensor): Predicted labels of the corresponding
              box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = 20

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_kps = []
        for level_idx, (cls_score, bbox_pred, kps_pred, stride, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list, kps_pred_list,
                    self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            
            kps_pred = kps_pred.permute(1, 2, 0).reshape(
                -1, self.NK*2) * stride[0]
            
            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = self.filter_scores_and_topk(
                scores, cfg['score_thr'], nms_pre,
                dict(bbox_pred=bbox_pred, kps_pred=kps_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            kps = filtered_results['kps_pred']

            # bboxes = self.bbox_coder.decode(
            #     self.anchor_center(priors), bbox_pred, max_shape=img_shape)

            bboxes = distance2bbox(self.anchor_center(priors), bbox_pred)

            kps = distance2kps(self.anchor_center(priors), kps)
            # kps = kps.reshape((kps.shape[0], -1, 2))

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_kps.append(kps)

        results = {
            'bboxes': torch.cat(mlvl_bboxes),
            'scores': torch.cat(mlvl_scores),
            'labels': torch.cat(mlvl_labels),
            'kps': torch.cat(mlvl_kps),
        }

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
    

    def _bbox_post_process(self,
                           results,
                           cfg,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # if rescale:
        #     assert img_meta.get('scale_factor') is not None
        #     scale_factor = [1 / s for s in img_meta['scale_factor']]
        #     results['bboxes'] = scale_boxes(results['bboxes'], scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results['bboxes'])
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results['bboxes'].numel() > 0:
            bboxes = results['bboxes']
            det_bboxes, keep_idxs = batched_nms(bboxes, results['scores'],
                                                results['labels'], cfg['nms'])
        
            for key in results:
                results[key] = results[key][keep_idxs]
            # some nms would reweight the score, such as softnms
            results['scores'] = det_bboxes[:, -1]
            # results = results[:1]

        return results
    
    def filter_scores_and_topk(self, scores, score_thr, topk, results=None):
        """Filter results using score threshold and topk candidates.

        Args:
            scores (Tensor): The scores, shape (num_bboxes, K).
            score_thr (float): The score filter threshold.
            topk (int): The number of topk candidates.
            results (dict or list or Tensor, Optional): The results to
            which the filtering rule is to be applied. The shape
            of each item is (num_bboxes, N).

        Returns:
            tuple: Filtered results

                - scores (Tensor): The scores after being filtered, \
                    shape (num_bboxes_filtered, ).
                - labels (Tensor): The class labels, shape \
                    (num_bboxes_filtered, ).
                - anchor_idxs (Tensor): The anchor indexes, shape \
                    (num_bboxes_filtered, ).
                - filtered_results (dict or list or Tensor, Optional): \
                    The filtered results. The shape of each item is \
                    (num_bboxes_filtered, N).
        """
        valid_mask = scores > score_thr
        scores = scores[valid_mask]
        valid_idxs = torch.nonzero(valid_mask)

        num_topk = min(topk, valid_idxs.size(0))
        # torch.sort is actually faster than .topk (at least on GPUs)
        scores, idxs = scores.sort(descending=True)
        scores = scores[:num_topk]
        topk_idxs = valid_idxs[idxs[:num_topk]]
        keep_idxs, labels = topk_idxs.unbind(dim=1)

        filtered_results = None
        if results is not None:
            if isinstance(results, dict):
                filtered_results = {k: v[keep_idxs] for k, v in results.items()}
            elif isinstance(results, list):
                filtered_results = [result[keep_idxs] for result in results]
            elif isinstance(results, torch.Tensor):
                filtered_results = results[keep_idxs]
            else:
                raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                        f'but get {type(results)}.')
        return scores, labels, keep_idxs, filtered_results


    def get_targets(self,
                    anchor_list: List[Tensor],
                    valid_flag_list: List[Tensor],
                    batch_gt_instances,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore = None,
                    unmap_outputs=True) -> tuple:
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_keypoints_targets, all_keypoints_weights, 
         pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply(
             self._get_targets_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs)
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results['pos_inds'].shape[0] for results in sampling_results_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        
        keypoints_targets_list = images_to_levels(all_keypoints_targets,
                                             num_level_anchors)
        keypoints_weights_list = images_to_levels(all_keypoints_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list,
                keypoints_targets_list, keypoints_weights_list, avg_factor)

    def _get_targets_single(self,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            num_level_anchors: List[int],
                            gt_instances,
                            img_meta: dict,
                            gt_instances_ignore = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors (list[int]): Number of anchors of each scale
                level.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: N is the number of total anchors in the image.

            - anchors (Tensor): All anchors in the image with shape (N, 4).
            - labels (Tensor): Labels of all anchors in the image with
              shape (N,).
            - label_weights (Tensor): Label weights of all anchor in the
              image with shape (N,).
            - bbox_targets (Tensor): BBox targets of all anchors in the
              image with shape (N, 4).
            - bbox_weights (Tensor): BBox weights of all anchors in the
              image with shape (N, 4).
            - pos_inds (Tensor): Indices of positive anchor with shape
              (num_pos,).
            - neg_inds (Tensor): Indices of negative anchor with shape
              (num_neg,).
            - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           -1)
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        pred_instances = dict(priors=anchors)
        assign_result = assign(
            anchors,
            num_level_anchors_inside,
            gt_instances.bboxes,
            gt_labels=gt_instances.labels,
            gt_bboxes_ignore=gt_instances_ignore)

        sampling_result = self._sample(
            assign_result=assign_result,
            pred_instances=anchors,
            gt_instances=gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        kps_targets = anchors.new_zeros(size=(anchors.shape[0], self.NK*2))
        kps_weights = anchors.new_zeros(size=(anchors.shape[0], self.NK*2))

        pos_inds = sampling_result['pos_inds']
        neg_inds = sampling_result['neg_inds']
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result['pos_gt_bboxes']
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result['pos_gt_labels']
            label_weights[pos_inds] = 1.0

            if self.use_kps:
                pos_assigned_gt_inds = sampling_result['pos_assigned_gt_inds'][0] # BUG: added [0]
                # kps_targets[pos_inds, :] = gt_instances.kps[pos_assigned_gt_inds,:,:2].reshape( (-1, self.NK*2) )
                kps_targets[pos_inds, :] = gt_instances.kps[pos_assigned_gt_inds].reshape( (-1, self.NK*2) )
                gt_kps_mean = gt_instances.kps[pos_assigned_gt_inds]#,:,2]
                kps_weights[pos_inds, :] = torch.ones(self.NK*2).to(DEVICE)#torch.mean(gt_kps_mean, dim=1, keepdims=True)
            
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

            if self.use_kps:
                kps_targets = unmap(kps_targets, num_total_anchors, inside_flags)
                kps_weights = unmap(kps_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                kps_targets, kps_weights,
                pos_inds, neg_inds, sampling_result)



    def _sample(self, assign_result, pred_instances, gt_instances):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors, points, or bboxes predicted by the model,
                shape(n, 4).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            :dict:: sampler results
        """
        gt_bboxes = gt_instances.bboxes
        # priors = pred_instances.priors
        priors = pred_instances

        pos_inds = torch.nonzero(
            assign_result['gt_inds'] > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result['gt_inds'] == 0, as_tuple=False).squeeze(-1).unique()

        gt_flags = priors.new_zeros(priors.shape[0], dtype=torch.uint8)

        if gt_bboxes.numel() == 0:
            # hack for index error case
            pos_gt_bboxes = gt_bboxes.view(-1, 4)
            pos_assigned_gt_inds = None
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_assigned_gt_inds = assign_result['gt_inds'][pos_inds] - 1
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long()]

 
        pos_gt_labels = assign_result['labels'][pos_inds]
        return dict(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            pos_gt_bboxes=pos_gt_bboxes,
            pos_gt_labels=pos_gt_labels,
            pos_assigned_gt_inds=pos_assigned_gt_inds,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside



if __name__ == '__main__':
    head = GFLHead(7, 9)
    feats = [torch.rand(1, 9, s, s) for s in [4, 8, 16, 32, 64]]
    print(len(feats), feats[0].shape)
    cls_quality_score, bbox_pred = head.forward(feats)
    print(len(cls_quality_score), len(head.scales))
    assert len(cls_quality_score) == len(head.scales)
    print(len(cls_quality_score), cls_quality_score[0].shape)
    print(len(bbox_pred), bbox_pred[0].shape, bbox_pred[1].shape, bbox_pred[2].shape)

    img_meta = {
        'pad_shape': (640,640,3),
        'img_shape': (640,640,3),
    }

    gt_bboxes = torch.Tensor(
        [[23.6667, 23.8757, 40.6326, 40.8874]])
        #  [[120.6667, 121.8757, 190.6326, 250.8874]]])
    gt_labels = torch.LongTensor([2])

    gt_instances = {
        'bboxes': gt_bboxes,
        'labels': gt_labels
    }
    gt_instances_sn = [SimpleNamespace(**gt_instances)]
