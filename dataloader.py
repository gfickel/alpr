import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import torch
from PIL import Image

from utils import *


class ALPRDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform, return_image_path=False, resize=None, grayscale=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.return_image_path = return_image_path
        self.to_tensor = transforms.ToTensor()
        self.resize = resize
        self.grayscale = grayscale

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = None
        while image is None:
            img_path = os.path.join(self.img_dir, os.path.basename(self.img_labels.iloc[idx, 0]))
            try:
                image = Image.open(img_path)
                if self.grayscale:
                    image = image.convert('L')
                # image = read_image(img_path)
            except:
                image = None
            if image != None:
                break
            idx = (idx+1) % len(self.img_labels)
                    
        W, H = image.size
        # H, W = image.shape[1:]
        w_scale = 224/W
        h_scale = 224/H
        l,t,r,b = self.img_labels.iloc[idx, 1:5].tolist()
        kps = self.img_labels.iloc[idx, 5:5+8].tolist()
        
        ocrs = self.img_labels.iloc[idx, 5+8:].tolist()
        ocrs = [int(x) for x in ocrs]
        ocrs[0] = PROVINCES_IDX[ocrs[0]-1]
        try:
            ocrs[1:] = [ADS_IDX[x-1] for x in ocrs[1:]]
        except:
            print(ocrs, '\n', len(ADS_IDX))
            exit(1)

        gt_ocrs = torch.LongTensor(ocrs)

        gt_bboxes = torch.Tensor(
            [[l*w_scale, t*h_scale, r*w_scale, b*h_scale]])
        gt_kps = torch.Tensor(
            [[pt*w_scale if idx%2==0 else pt*h_scale for idx,pt in enumerate(kps)]]
        )
        gt_labels = torch.LongTensor([0])

        if self.transform:
            image = self.transform(image)
        else:
            if self.resize:
                image = image.resize(self.resize)
                
            # image = self.to_tensor(image)
        
        if self.return_image_path:
            # return image, img_path, gt_bboxes, gt_labels, gt_kps, gt_ocrs
            return image, img_path, gt_ocrs
        else:
            # return image, gt_bboxes, gt_labels, gt_kps, gt_ocrs
            return image, gt_ocrs