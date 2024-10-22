import torch
from torch import nn
from torchvision.ops import roi_align
import timm
from timm.data.config import resolve_model_data_config

from head import GFLHead
from bifpn import BiFpn
from model_config import get_efficientdet_config


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_channels, embedding_size, output_channels):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_channels, embedding_size, bidirectional=True)
        self.embedding = nn.Linear(embedding_size * 2, output_channels)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):

    def __init__(self, nc, alphabet_size, embedding_size, leakyRelu=False):
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, embedding_size, embedding_size),
            BidirectionalLSTM(embedding_size, embedding_size, alphabet_size))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        return output


class CRNNUsingBackbone(nn.Module):

    def __init__(self, input_channels, alphabet_size, embedding_size, leakyRelu=False, max_pool_width=True, num_bilstm=2):
        super(CRNNUsingBackbone, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        # nm = [32, 64, 128, 128, 256, 256, 256]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = input_channels if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        if max_pool_width:
            cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((4, 2)))  # input_channelsx14x28
        else:
            cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((4, 1)))  # input_channelsx14x56
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((4, 1)))  # 128x3x28/56
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((3, 1)))  # 256x1x28/56
        
        convRelu(4, True)
        convRelu(5)
    
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(nm[-1], embedding_size, embedding_size),
            BidirectionalLSTM(embedding_size, embedding_size, alphabet_size))
        # layers = []

        # for i in range(num_bilstm):
        #     in_size = nm[-1] if i == 0 else embedding_size
        #     out_size = alphabet_size if i == num_bilstm - 1 else embedding_size
        #     layers.append(BidirectionalLSTM(in_size, embedding_size, out_size))

        # self.rnn = nn.Sequential(*layers)


    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        return output


class Detector(nn.Module):

    def __init__(self, backbone: str, num_classes: int, test_cfg: dict, use_kps: bool=False, use_ocr: bool=False,
                 alphabet_size: int=0, crnn_embedding: int=64, max_pool_width: bool=True, num_bilstm=2):
        super(Detector, self).__init__()
        self.use_ocr = use_ocr
        self.test_cfg = test_cfg
        self.alphabet_size = alphabet_size

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
        if self.use_ocr:
            input_ch = fpn_config.fpn_channels*fpn_config.num_levels
            self.ocr = CRNNUsingBackbone(input_ch, alphabet_size, crnn_embedding, max_pool_width=max_pool_width, num_bilstm=num_bilstm)

            feats_bb = self.backbone(torch.randn(1, *model_data_config['input_size']).to(DEVICE))
            feats_neck = self.fpn(feats_bb)
            sizes = [x.shape for x in feats_neck]
            self.upsamplers = []
            for size in sizes[1:]:
                self.upsamplers.append(nn.Upsample(sizes[0][-2:], mode='bilinear'))

    def forward(self, x: torch.Tensor, test_cfg: dict=None):
        feats_bb = self.backbone(x)
        feats_neck = self.fpn(feats_bb)

        cls_quality_score, bbox_pred, kps_pred = self.head.forward(feats_neck)

        test_cfg = self.test_cfg if test_cfg is None else test_cfg
        batch_size = x.shape[0]
        ocrs_logits, have_bbox = None, None
        if self.use_ocr:
            res = self.head.predict_by_feat(
                cls_quality_score, bbox_pred, kps_pred,
                batch_img_metas=None,
                cfg=test_cfg, rescale=False)
            
            # Capping at 1 bbox per image
            have_bbox = [False]*len(res)
            rois = []
            for idx in range(len(res)):
                if res[idx]['kps'].numel() > 0:
                    have_bbox[idx] = True
                    rois.append(res[idx]['bboxes'][0].reshape(1,4))


            if sum(have_bbox) > 0:
                # Crop the original image given the bounding boxes
                cropped_boxes = roi_align(x[have_bbox], rois, output_size=self.input_size[1:])
                
                # (im-mean)/std -> x
                # mean = self.model_data_config['mean']
                # std = self.model_data_config['std']
                # im = x*std+mean
                # from torchvision.utils import save_image
                # def save_im_util(im_tensor, im_path):
                #     print('save_im', im_tensor.shape)
                #     test_im = im_tensor.detach().to('cpu').reshape(3,224,224)
                #     test_im = test_im.permute(1,2,0) # CHW -> HWC
                #     test_im = test_im * torch.Tensor(list(std))
                #     test_im = torch.add(test_im, torch.Tensor(list(mean)))
                #     test_im = test_im.permute(2,0,1) # HWC -> CHW
                #     save_image(test_im, im_path)

                    

                # save_im_util(cropped_boxes[0], 'plate.png')
                # save_im_util(x[0], 'full_im.png')
                # save_im_util(x[0][rois[0]], 'plate_roi.png')
                # print('\nRois BEFORE\n', rois[0])
                # exit(1)

                # Normal flow, calling the backbone and neck
                feats_bb_ocr = self.backbone(cropped_boxes)
                feats_neck_ocr = self.fpn(feats_bb_ocr)

                # We are not dealing with a multi-scale OCR, so upsample the smaller features and
                # get a single, high dimensional, feature map
                feats_neck_ocr_upsample = [ups(x) for ups,x in zip(self.upsamplers, feats_neck_ocr[1:])]

                final_feats_neck_ocr = torch.cat([feats_neck_ocr[0]]+feats_neck_ocr_upsample, dim=1)

                # We have a single, kinda of high dimensional and resolution featuremap that we
                # can pass to our OCR network
                ocrs_logits = self.ocr(final_feats_neck_ocr)
            else:
                ocrs_logits = torch.zeros((28, batch_size, self.alphabet_size), requires_grad=True)
                # ocrs[:,have_bbox,:] = ocr

            # for im_idx in range(len(res)):
            #     rois = res[im_idx]['bboxes']
            #     if rois.shape[0] == 0:
            #         ocr = ''
            #     else:
            #         # ocr_features = [self.upsamplers[idx](x) for idx,x in enumerate(feats_neck[1:])]
            #         # ocr_features = torch.cat([feats_neck[0]]+ocr_features, dim=1)
            #         # ocr = self.ocr(ocr_features)

            #     ocrs.append(ocr)
        

        return cls_quality_score, bbox_pred, kps_pred, ocrs_logits, have_bbox


    def forward_ocr(self, x: torch.Tensor, test_cfg: dict=None):
        test_cfg = self.test_cfg if test_cfg is None else test_cfg
        feats_bb_ocr = self.backbone(x)
        feats_neck_ocr = self.fpn(feats_bb_ocr)

        # We are not dealing with a multi-scale OCR, so upsample the smaller features and
        # get a single, high dimensional, feature map
        feats_neck_ocr_upsample = [ups(x) for ups,x in zip(self.upsamplers, feats_neck_ocr[1:])]

        final_feats_neck_ocr = torch.cat([feats_neck_ocr[0]]+feats_neck_ocr_upsample, dim=1)

        # We have a single, kinda of high dimensional and resolution featuremap that we
        # can pass to our OCR network
        ocrs_logits = self.ocr(final_feats_neck_ocr)
        
        return ocrs_logits
