"""EfficientDet Configurations

Adapted from official impl at https://github.com/google/automl/tree/master/efficientdet

TODO use a different config system (OmegaConfig -> Hydra?), separate model from train specific hparams
"""

from omegaconf import OmegaConf
from copy import deepcopy


def default_detection_model_configs():
    """Returns a default detection configs."""
    h = OmegaConf.create()

    h.image_size = (512, 512)

    # FPN and head config
    h.pad_type = 'same'  # original TF models require an equivalent of Tensorflow 'SAME' padding
    h.act_type = 'hard_swish'
    h.norm_layer = None  # defaults to batch norm when None
    h.norm_kwargs = dict(eps=.001, momentum=.01)
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_channels = 64
    h.separable_conv = True
    h.apply_resample_bn = True
    h.conv_bn_relu_pattern = False
    h.downsample_type = 'max'
    h.upsample_type = 'nearest'
    h.redundant_bias = True  # original TF models have back to back bias + BN layers, not necessary!
    h.head_bn_level_first = False  # change order of BN in head repeat list of lists, True for torchscript compat
    h.head_act_type = None  # activation for heads, same as act_type if None

    h.fpn_config = None
    h.fpn_drop_path_rate = 0.  # No stochastic depth in default. NOTE not currently used, unstable training

    h.min_level = 2
    h.max_level = 6
    h.num_levels = h.max_level - h.min_level + 1
    h.num_scales = 3

    h.fpn_name = 'bifpn_sum',
    h.mean=(0.5, 0.5, 0.5),
    h.std=(0.5, 0.5, 0.5),
    
    return h


tf_efficientdet_lite_common = dict(
    fpn_name='bifpn_sum',
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    act_type='relu6',
)




def get_efficientdet_config(model_name='tf_efficientdet_d1'):
    """Get the default config for EfficientDet based on model name."""
    h = default_detection_model_configs()
    h.num_levels = h.max_level - h.min_level + 1
    h = deepcopy(h)  # may be unnecessary, ensure no references to param dict values
    # OmegaConf.set_struct(h, True)  # FIXME good idea?
    h.fpn_name = h.fpn_name[0]
    return h