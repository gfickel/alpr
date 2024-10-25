import os
import argparse
import json
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import wandb

from utils import *
from dataloader import ALPRDataset
from maskocr import *

# Set up interactive mode
plt.ion()



class SyntheticOCRDataset(Dataset):
    def __init__(self, vocab, seq_length=10, num_samples=1000, img_height=32, img_width=128):
        self.vocab = vocab
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.data = self._generate_data()
        
        # Define augmentations
        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def _generate_data(self):
        data = []
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found: {font_path}. Please install it or provide a different font path.")
        
        for _ in range(self.num_samples):
            text = ''.join(random.choices(self.vocab[1:], k=self.seq_length))
            
            # Create RGB image with random background color
            background_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            image = Image.new('RGB', (self.img_width, self.img_height), color=background_color)
            draw = ImageDraw.Draw(image)
            
            # Randomly adjust font size
            font_size = random.randint(int(self.img_height * 0.65), int(self.img_height * 0.85))
            font = ImageFont.truetype(font_path, font_size)
            
            # Calculate text size using font.getbbox instead of draw.textsize
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Randomly position text while ensuring it fits within the image
            max_x = max(0, self.img_width - text_width)
            max_y = max(0, self.img_height - text_height)
            position = (random.randint(0, int(max_x*.5)), random.randint(0, int(max_y*.5)))
            
            # Choose a random dark color for text
            text_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            
            # Rotate text
            rotation = random.uniform(-5, 5)
            draw.text(position, text, font=font, fill=text_color)
            image = image.rotate(rotation, expand=True, fillcolor=background_color)
            draw = ImageDraw.Draw(image)
            
            # Crop image back to original size
            left = (image.width - self.img_width) / 2
            top = (image.height - self.img_height) / 2
            right = (image.width + self.img_width) / 2
            bottom = (image.height + self.img_height) / 2
            image = image.crop((left, top, right, bottom))
            
            image_np = np.array(image).astype(np.float32) / 255.0
            data.append((image_np, text))

        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, text = self.data[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = self.transform(image)
        text_indices = [self.vocab.index(char) for char in text]
        return image, torch.tensor(text_indices)


def train_model(model, train_function, train_dataloader, val_dataloader, device, vocab, 
                model_name=None, num_epochs=40, use_wandb=False, config=None, **train_kwargs):
    """
    Manages the model: loads if exists, trains and saves at each epoch.
    Returns the loaded or trained model
    """
    if model_name is None:
        model_name = train_function.__name__

    final_model_path = os.path.join('model_bin', f"{model_name}_final.pth")
    temp_model_path = os.path.join('model_bin', f"{model_name}_temp.pth")
    
    start_epoch = 0
    if os.path.exists(temp_model_path):
        print(f"Loading existing temporary model from {temp_model_path}")
        checkpoint = torch.load(temp_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    elif os.path.exists(final_model_path):
        print(f"Loading existing final model from {final_model_path}")
        checkpoint = torch.load(final_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return
    else:
        print(f"No existing model found. Training new model using {model_name} for {num_epochs} epochs.")

    # Initialize wandb
    if use_wandb:
        wandb.init(project="alpr_ocr", config=config, group=train_kwargs['version'], name=model_name)
        wandb.watch(model)

    # Train the model
    train_function(model, train_dataloader, val_dataloader, device, vocab, 
                    num_epochs=num_epochs, start_epoch=start_epoch, 
                    temp_model_path=temp_model_path, use_wandb=use_wandb, **train_kwargs)

    # Save the final model
    print(f"Saving final trained model to {final_model_path}")
    torch.save({'model_state_dict': model.state_dict()}, final_model_path)

    # Remove temporary checkpoint
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

    if use_wandb:
        wandb.finish()

    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments for image processing and model configuration.")
    
    parser.add_argument('--img_height', type=int, default=32, help='Height of the input image')
    parser.add_argument('--img_width', type=int, default=128, help='Width of the input image')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[32, 8], help='Patch size for image processing (height, width)')
    parser.add_argument('--batch_size', type=int, default=512*7, help='Batch size for training')
    parser.add_argument('--embed_dim', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=12, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--max_sequence_length', type=int, default=7, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--emb_dropout', type=float, default=0.1, help='Embedding dropout rate')
    parser.add_argument('--version', type=str, required=True, help='Training Version')
    parser.add_argument('--norm_image', type=int, default=0, help='Normalize the input image')
    parser.add_argument('--overlap', type=int, default=0, help='Patch Overlap')
    parser.add_argument('--device', type=int, default=0, help='Normalize the input image')
    parser.add_argument('--start_lr', type=float, default=1e-4, help='Starting learning rate')
    parser.add_argument('--plateau_thr', type=int, default=-1, help='Number of batches to use on dlib plateau detection')
    parser.add_argument('--wandb', action='store_true', help='Whether to log with wandb or not')
    
    args = parser.parse_args()
    return args

def save_arguments_to_json(args, filename):
    # Convert args namespace to dictionary
    args_dict = vars(args)
    
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    print(f"Arguments saved to {filename}")

def main():
    cfg = parse_arguments()
    save_arguments_to_json(cfg, os.path.join('configs', f'{cfg.version}.json'))
    current_device = cfg.device
    device = torch.device(f'cuda:{current_device}' if torch.cuda.is_available() else 'cpu')
    
    vocab = FULL_ALPHABET
    vocab_size = len(vocab)+1

    model = MaskOCR(cfg.img_height, cfg.img_width, cfg.patch_size, cfg.embed_dim, cfg.num_heads, cfg.num_encoder_layers,
                    cfg.num_decoder_layers, vocab_size, cfg.max_sequence_length, dropout=cfg.dropout, emb_dropout=cfg.emb_dropout,
                    overlap=cfg.overlap)

    data_transform = transforms.Compose([
        transforms.RandAugment(num_ops=3, magnitude=7),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img.float() / (255.0 if cfg.norm_image else 1.0)),
    ])

    local_path = '.' if torch.cuda.is_available() else 'alpr_datasets'
    if cfg.img_height == 32:
        train_ds = 'output_images_plates_gt'
        val_ds = 'plates_ccpd_weather'
    elif cfg.img_height == 48:
        train_ds = 'plates_ccpd_base_48'
        val_ds = 'plates_ccpd_weather_48'

    train_dataset = ALPRDataset(
        f'../{local_path}/{train_ds}/alpr_annotation.csv',
        f'../{local_path}/{train_ds}/', #None)
        data_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)

    val_dataset = ALPRDataset(
        f'../{local_path}/{val_ds}/alpr_annotation.csv',
        f'../{local_path}/{val_ds}/', #None)
        data_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=False)

    # summary(model, input_size=(2, 3, cfg.img_height, cfg.img_width), depth=5)

    train_model(model, train_visual_pretraining, train_dataloader, val_dataloader, device, vocab,
                model_name=f'train_visual_pretraining_{cfg.version}', num_epochs=6, version=cfg.version,
                start_lr=cfg.start_lr, plateau_threshold=cfg.plateau_thr, use_wandb=cfg.wandb, config=cfg)
    
    # Then, train for text recognition
    train_model(model, train_text_recognition, train_dataloader, val_dataloader, device, vocab,
                model_name=f'train_text_recognition_{cfg.version}', num_epochs=20, freeze_encoder=True,
                version=cfg.version, start_lr=cfg.start_lr, plateau_threshold=cfg.plateau_thr, use_wandb=cfg.wandb, config=cfg)
    train_model(model, train_text_recognition, train_dataloader, val_dataloader, device, vocab,
                model_name=f'train_text_recognition_full_{cfg.version}', num_epochs=120, freeze_encoder=False,
                version=cfg.version, start_lr=cfg.start_lr, plateau_threshold=cfg.plateau_thr, use_wandb=cfg.wandb, config=cfg)
    torch.save(model.state_dict(), f'model_bin/my_model_{cfg.version}.pth')

if __name__ == "__main__":
    main()