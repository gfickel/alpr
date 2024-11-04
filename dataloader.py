import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageEnhance, ImageFilter
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


class OCRSafeAugment:
    def __init__(self, strength=0.5):
        """
        Initialize OCR-safe augmentation with controllable strength.
        
        Args:
            strength (float): Global strength of augmentations, from 0.0 to 1.0
        """
        self.strength = strength
        
    def apply_perspective(self, img):
        """Safe perspective transform that preserves text readability using torchvision"""
        width, height = img.size
        
        # Calculate safe perspective points that won't cut off text
        margin_w = int(width * 0.1 * self.strength)
        margin_h = int(height * 0.1 * self.strength)
        
        # Define start points (original image corners)
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        
        # Define end points (randomly perturbed corners within safe margins)
        endpoints = [
            [random.randint(0, margin_w), random.randint(0, margin_h)],  # top-left
            [width - 1 - random.randint(0, margin_w), random.randint(0, margin_h)],  # top-right
            [width - 1 - random.randint(0, margin_w), height - 1 - random.randint(0, margin_h)],  # bottom-right
            [random.randint(0, margin_w), height - 1 - random.randint(0, margin_h)]  # bottom-left
        ]
        
        # Convert image to tensor if it's not already
        if not isinstance(img, torch.Tensor):
            img_tensor = transforms.ToTensor()(img)
        else:
            img_tensor = img
            
        # Apply perspective transform
        transformed_img = TF.perspective(
            img_tensor,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=InterpolationMode.BILINEAR,
            fill=[0, 0, 0]  # black fill for areas outside the transform
        )
        
        # Convert back to PIL if input was PIL
        if isinstance(img, Image.Image):
            transformed_img = transforms.ToPILImage()(transformed_img)
            
        return transformed_img
    
    def apply_elastic_transform(self, img):
        """Elastic deformation that maintains character integrity"""
        img_tensor = transforms.ToTensor()(img)
        _, h, w = img_tensor.shape
        
        # Generate displacement fields
        grid_scale = 4  # Larger value = more subtle distortion
        dx = torch.rand(h // grid_scale, w // grid_scale) * 2 - 1
        dy = torch.rand(h // grid_scale, w // grid_scale) * 2 - 1
        
        # Upscale displacement fields and apply smoothing
        dx = F.interpolate(dx.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bicubic')[0, 0]
        dy = F.interpolate(dy.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bicubic')[0, 0]
        
        # Scale displacement based on strength
        displacement_scale = 0.01 * self.strength
        dx *= displacement_scale * w
        dy *= displacement_scale * h
        
        # Create sampling grid
        x_grid = torch.arange(w).float().repeat(h, 1)
        y_grid = torch.arange(h).float().repeat(w, 1).t()
        
        x_grid = x_grid + dx
        y_grid = y_grid + dy
        
        # Normalize coordinates to [-1, 1]
        x_grid = 2 * (x_grid / (w - 1)) - 1
        y_grid = 2 * (y_grid / (h - 1)) - 1
        
        # Stack and reshape
        grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)
        
        # Apply sampling grid
        img_tensor = F.grid_sample(img_tensor.unsqueeze(0), grid, align_corners=True)[0]
        
        return transforms.ToPILImage()(img_tensor)
    
    def apply_color_jitter(self, img):
        """Apply color jittering with controlled intensity"""
        factors = {
            'brightness': random.uniform(1 - 0.4 * self.strength, 1 + 0.4 * self.strength),
            'contrast': random.uniform(1 - 0.4 * self.strength, 1 + 0.4 * self.strength),
            'saturation': random.uniform(1 - 0.4 * self.strength, 1 + 0.4 * self.strength),
            'hue': random.uniform(-0.2 * self.strength, 0.2 * self.strength)
        }
        
        for factor, value in factors.items():
            if factor == 'brightness':
                img = ImageEnhance.Brightness(img).enhance(value)
            elif factor == 'contrast':
                img = ImageEnhance.Contrast(img).enhance(value)
            elif factor == 'saturation':
                img = ImageEnhance.Color(img).enhance(value)
            elif factor == 'hue':
                img = transforms.functional.adjust_hue(img, value)
        return img
    
    def apply_blur(self, img):
        """Apply slight blur with controlled intensity"""
        radius = self.strength * 0.5  # Max blur radius of 0.5 pixels
        return img.filter(ImageFilter.GaussianBlur(radius))
    
    def __call__(self, img):
        """Apply all augmentations with random chance"""
        augmentations = [
            (self.apply_perspective, 0.5),
            (self.apply_elastic_transform, 0.5),
            (self.apply_color_jitter, 0.7),
            (self.apply_blur, 0.3)
        ]
        
        for aug_func, prob in augmentations:
            if random.random() < prob:
                img = aug_func(img)
        
        return img

# Create the complete transformation pipeline
def create_ocr_transform(augment_strength=1.0):
    return transforms.Compose([
        OCRSafeAugment(strength=augment_strength),
        transforms.ToTensor(),
        # Add any additional transforms here
    ])