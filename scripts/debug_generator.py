import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.join(os.path.abspath(__file__))), '..' ))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from dataloader import ALPRDataset, create_ocr_transform

# Assuming SyntheticOCRDataset is in a file named synthetic_ocr_dataset.py
# from maskocr_pretrain_simple_v2 import SyntheticOCRDataset

def save_synthetic_ocr_images(output_dir, num_images=100):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    data_transform = create_ocr_transform(augment_strength=2.25)
    local_path = 'alpr_datasets'
    train_ds = 'plates_ccpd_base_48'
    
    dataset = ALPRDataset(
        f'../{local_path}/{train_ds}/alpr_annotation.csv',
        f'../{local_path}/{train_ds}/', #None)
        data_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1, drop_last=True)


    # Generate and save images
    for i, (image, text_indices) in enumerate(dataloader):
        text = 'dummy'

        # Save the image
        image_path = os.path.join(output_dir, f'image_{i:04d}.png')
        save_image(image, image_path)

        print(f"Saved image and text for sample {i}: {text}", end='\r')
        if i > num_images:
            break

if __name__ == '__main__':
    output_directory = 'synthetic_ocr_debug_images'
    num_images_to_generate = 100

    save_synthetic_ocr_images(output_directory, num_images_to_generate)
    print(f"Generated {num_images_to_generate} images in '{output_directory}'")