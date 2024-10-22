""" Original code from: https://github.com/rwightman/efficientdet-pytorch/

PyTorch EfficientDet model

Based on official Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
Paper: https://arxiv.org/abs/1911.09070

Hacked together by Ross Wightman
"""
import logging
import itertools
from functools import partial
from typing import List, Callable, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

try:
    from timm.layers import create_conv2d, create_pool2d, get_act_layer
except ImportError:
    from timm.models.layers import create_conv2d, create_pool2d, get_act_layer

_USE_SCALE = False
_ACT_LAYER = get_act_layer('silu')


def get_feat_sizes(image_size: Tuple[int, int], max_level: int):
    """Get feat widths and heights for all levels.
    Args:
      image_size: a tuple (H, W)
      max_level: maximum feature level.
    Returns:
      feat_sizes: a list of tuples (height, width) for each level.
    """
    feat_size = image_size
    feat_sizes = [feat_size]
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append(feat_size)
    return feat_sizes


def panfpn_config(min_level, max_level, weight_method=None):
    """PAN FPN config.

    This defines FPN layout from Path Aggregation Networks as an alternate to
    BiFPN, it does not implement the full PAN spec.

    Paper: https://arxiv.org/abs/1803.01534
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level, min_level - 1, -1):
        # top-down path.
        offsets = [level_last_id(i), level_last_id(i + 1)] if i != max_level else [level_last_id(i)]
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': offsets,
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level, max_level + 1):
        # bottom-up path.
        offsets = [level_last_id(i), level_last_id(i - 1)] if i != min_level else [level_last_id(i)]
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': offsets,
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    return p


def qufpn_config(min_level, max_level, weight_method=None):
    """A dynamic quad fpn config that can adapt to different min/max levels.

    It extends the idea of BiFPN, and has four paths:
        (up_down -> bottom_up) + (bottom_up -> up_down).

    Paper: https://ieeexplore.ieee.org/document/9225379
    Ref code: From contribution to TF EfficientDet
    https://github.com/google/automl/blob/eb74c6739382e9444817d2ad97c4582dbe9a9020/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'
    quad_method = 'fastattn'
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    level_first_id = lambda level: node_ids[level][0]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path 1.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    for i in range(min_level + 1, max_level):
        # bottom-up path 2.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))

    i = max_level
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [level_first_id(i)] + [level_last_id(i - 1)],
        'weight_method': weight_method
    })
    node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(min_level + 1, max_level + 1, 1):
        # bottom-up path 3.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [
                level_first_id(i), level_last_id(i - 1) if i != min_level + 1 else level_first_id(i - 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(max_level - 1, min_level, -1):
        # top-down path 4.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [node_ids[i][0]] + [node_ids[i][-1]] + [level_last_id(i + 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    i = min_level
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [node_ids[i][0]] + [level_last_id(i + 1)],
        'weight_method': weight_method
    })
    node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    # NOTE: the order of the quad path is reversed from the original, my code expects the output of
    # each FPN repeat to be same as input from backbone, in order of increasing reductions
    for i in range(min_level, max_level + 1):
        # quad-add path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [node_ids[i][2], node_ids[i][4]],
            'weight_method': quad_method
        })
        node_ids[i].append(next(id_cnt))

    return p

def bifpn_config(min_level, max_level, weight_method=None):
    """BiFPN config.
    Adapted from https://github.com/google/automl/blob/56815c9986ffd4b508fe1d68508e268d129715c1/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)],
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)],
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))
    return p

def get_fpn_config(fpn_name, min_level=3, max_level=7):
    if not fpn_name:
        fpn_name = 'bifpn_fa'
    name_to_config = {
        'bifpn_sum': bifpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
        'bifpn_attn': bifpn_config(min_level=min_level, max_level=max_level, weight_method='attn'),
        'bifpn_fa': bifpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
        'pan_sum': panfpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
        'pan_fa': panfpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
        'qufpn_sum': qufpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
        'qufpn_fa': qufpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
    }
    return name_to_config[fpn_name]



class SequentialList(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""
    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for ix in x:
            if torch.isnan(ix).any():
                print('x SequentialList nan')
        for module in self:
            x = module(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding='',
            bias=False,
            channel_multiplier=1.0,
            pw_kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=_ACT_LAYER,
    ):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(
            in_channels,
            int(in_channels * channel_multiplier),
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            depthwise=True,
        )
        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier),
            out_channels,
            pw_kernel_size,
            padding=padding,
            bias=bias,
        )
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBnAct2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            padding='',
            bias=False,
            norm_layer=nn.BatchNorm2d,
            act_layer=_ACT_LAYER,
    ):
        super(ConvBnAct2d, self).__init__()
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )

        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride)

        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, input):
        if torch.isnan(input).any():
            print('ConvBnAct2d init nan')
        x = self.conv(input)
        if torch.isnan(x).any():
            print('ConvBnAct2d conv nan')
        if self.bn is not None:
            x = self.bn(x)
            if torch.isnan(x).any():
                print('ConvBnAct2d bn nan')
        if self.act is not None:
            x = self.act(x)
            if torch.isnan(x).any():
                print('ConvBnAct2d act nan')
        if torch.isnan(x).any():
            print('ConvBnAct2d nan')
        return x
    

class BiFpnLayer(nn.Module):
    def __init__(
            self,
            feature_info,
            feat_sizes,
            fpn_config,
            fpn_channels,
            num_levels=5,
            pad_type='',
            downsample=None,
            upsample=None,
            norm_layer=nn.BatchNorm2d,
            act_layer=_ACT_LAYER,
            apply_resample_bn=False,
            pre_act=True,
            separable_conv=True,
            redundant_bias=False,
    ):
        super(BiFpnLayer, self).__init__()
        self.num_levels = num_levels
        # fill feature info for all FPN nodes (chs and feat size) before creating FPN nodes
        fpn_feature_info = feature_info + [
            dict(num_chs=fpn_channels, size=feat_sizes[fc['feat_level']]) for fc in fpn_config.nodes]

        self.fnode = nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            combine = FpnCombine(
                fpn_feature_info,
                fpn_channels,
                tuple(fnode_cfg['inputs_offsets']),
                output_size=feat_sizes[fnode_cfg['feat_level']],
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_resample_bn=apply_resample_bn,
                redundant_bias=redundant_bias,
                weight_method=fnode_cfg['weight_method'],
            )

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels,
                out_channels=fpn_channels,
                kernel_size=3,
                padding=pad_type,
                bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            if pre_act:
                conv_kwargs['bias'] = redundant_bias
                conv_kwargs['act_layer'] = None
                after_combine.add_module('act', act_layer(inplace=True))
            after_combine.add_module(
                'conv', SeparableConv2d(**conv_kwargs) if separable_conv else ConvBnAct2d(**conv_kwargs))

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))

        self.feature_info = fpn_feature_info[-num_levels::]

    def forward(self, x: List[torch.Tensor]):
        for ix in x:
            if torch.isnan(ix).any():
                print('x BiFpnLayer nan')

        for fn in self.fnode:
            x.append(fn(x))
            if torch.isnan(x[-1]).any():
                print('bifpnlayer nan')
        return x[-self.num_levels::]


class Interpolate2d(nn.Module):
    r"""Resamples a 2d Image

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(
            self,
            size: Optional[Union[int, Tuple[int, int]]] = None,
            scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
            mode: str = 'nearest',
            align_corners: bool = False,
    ) -> None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        res = F.interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=False,
        )
        if torch.isnan(res).any():
            print('Interpolate nan')
        return res


class ResampleFeatureMap(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            input_size,
            output_size,
            pad_type='',
            downsample=None,
            upsample=None,
            norm_layer=nn.BatchNorm2d,
            apply_bn=False,
            redundant_bias=False,
    ):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size

        if in_channels != out_channels:
            self.add_module(
                'conv',
                ConvBnAct2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=pad_type,
                    norm_layer=norm_layer if apply_bn else None,
                    bias=not apply_bn or redundant_bias,
                    act_layer=None,
                )
            )

        if input_size[0] > output_size[0] and input_size[1] > output_size[1]:
            if downsample in ('max', 'avg'):
                stride_size_h = int((input_size[0] - 1) // output_size[0] + 1)
                stride_size_w = int((input_size[1] - 1) // output_size[1] + 1)
                if stride_size_h == stride_size_w:
                    kernel_size = stride_size_h + 1
                    stride = stride_size_h
                else:
                    # FIXME need to support tuple kernel / stride input to padding fns
                    kernel_size = (stride_size_h + 1, stride_size_w + 1)
                    stride = (stride_size_h, stride_size_w)
                down_inst = create_pool2d(downsample, kernel_size=kernel_size, stride=stride, padding=pad_type)
            else:
                if _USE_SCALE:  # FIXME not sure if scale vs size is better, leaving both in to test for now
                    scale = (output_size[0] / input_size[0], output_size[1] / input_size[1])
                    down_inst = Interpolate2d(scale_factor=scale, mode=downsample)
                else:
                    down_inst = Interpolate2d(size=output_size, mode=downsample)
            self.add_module('downsample', down_inst)
        else:
            if input_size[0] < output_size[0] or input_size[1] < output_size[1]:
                if _USE_SCALE:
                    scale = (output_size[0] / input_size[0], output_size[1] / input_size[1])
                    self.add_module('upsample', Interpolate2d(scale_factor=scale, mode=upsample))
                else:
                    self.add_module('upsample', Interpolate2d(size=output_size, mode=upsample))


    def forward(self, x):
        if torch.isnan(x).any():
            print('ResampleFeatureMap forward nan')
        res = super(ResampleFeatureMap, self).forward(x)
        if torch.isnan(res).any():
            print('ResampleFeatureMap forward nan')
        return res
            


class FpnCombine(nn.Module):
    def __init__(
            self,
            feature_info,
            fpn_channels,
            inputs_offsets,
            output_size,
            pad_type='',
            downsample=None,
            upsample=None,
            norm_layer=nn.BatchNorm2d,
            apply_resample_bn=False,
            redundant_bias=False,
            weight_method='attn',
    ):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            self.resample[str(offset)] = ResampleFeatureMap(
                feature_info[offset]['num_chs'],
                fpn_channels,
                input_size=feature_info[offset]['size'],
                output_size=output_size,
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_bn=apply_resample_bn,
                redundant_bias=redundant_bias,
            )

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        for ix in x:
            if torch.isnan(ix).any():
                print('x FpnCombine nan')
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        out = torch.sum(out, dim=-1)
        if torch.isnan(out).any():
            print('FpnCombine nan')
        return out


class Fnode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        for ix in x:
            if torch.isnan(ix).any():
                print('x Fnode nan')
        return self.after_combine(self.combine(x))



class BiFpn(nn.Module):

    def __init__(self, config, feature_info):
        super(BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(
            config.fpn_name, min_level=config.min_level, max_level=config.max_level)

        feat_sizes = get_feat_sizes(config.image_size, max_level=config.max_level)
        prev_feat_size = feat_sizes[config.min_level]
        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            feat_size = feat_sizes[level + config.min_level]
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                feature_info[level]['size'] = feat_size
            else:
                # Adds a coarser level by downsampling the last feature map
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    input_size=prev_feat_size,
                    output_size=feat_size,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    apply_bn=config.apply_resample_bn,
                    redundant_bias=config.redundant_bias,
                )
                in_chs = config.fpn_channels
                feature_info.append(dict(num_chs=in_chs, size=feat_size))
            prev_feat_size = feat_size

        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                feat_sizes=feat_sizes,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                pre_act=not config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        for ix in x:
            if torch.isnan(ix).any():
                print('x BiFpn nan')
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x
