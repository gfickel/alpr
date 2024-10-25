import math
import random
import time

import dlib
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import *
from vit_mae import ViT, MAE
# from dlib_loss_plateu import is_in_plateau


def is_in_plateau(vec, threshold):
    dlib_simple = dlib.count_steps_without_decrease(vec)
    dlib_robust = dlib.count_steps_without_decrease_robust(vec)
    return dlib_simple > threshold and dlib_robust > threshold

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(0)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return x + pe.to(x.device)

class LatentContextualRegressor(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_len):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, mask_queries, visible_patch_representations):
        # mask_queries shape: [num_masked_patches, batch_size, embed_dim]
        # Add positional encoding to mask_queries
        mask_queries = self.pos_encoder(mask_queries)
        
        # visible_patch_representations shape: [batch_size, num_patches, embed_dim]
        return self.transformer_decoder(mask_queries, visible_patch_representations)


class MaskOCR_Encoder(nn.Module):
    def __init__(self, img_height, img_width, patch_size, embed_dim, num_heads, num_layers, dropout, emb_dropout, overlap):
        super().__init__()
        self.vit = ViT(
            image_size=(img_height, img_width),
            patch_size=patch_size,
            num_classes=0,  # We don't need classification, so we use embed_dim as num_classes
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=embed_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            overlap=overlap,
        )
        self.num_patches = (img_height // patch_size[0]) * (img_width // patch_size[1])
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
    def forward(self, x, mask=None):
        x = self.vit(x)
        
        if mask is not None:
            x[mask] = 0
        
        return x

class MaskOCR_Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_sequence_length, dropout, emb_dropout):
        super().__init__()
        self.character_queries = nn.Parameter(torch.randn(max_sequence_length, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(embed_dim, max_sequence_length)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, memory, tgt_mask=None):
        # memory shape: [batch_size, num_patches, embed_dim]
        batch_size = memory.size(0)
        
        # Expand character queries to match batch size
        tgt = self.character_queries.expand(-1, batch_size, -1)
        # tgt shape: [max_sequence_length, batch_size, embed_dim]
        
        tgt = self.pos_encoder(tgt)
        # tgt shape remains: [max_sequence_length, batch_size, embed_dim]
        
        # Transpose memory to match expected input of transformer decoder
        memory = memory.transpose(0, 1)
        # memory shape: [num_patches, batch_size, embed_dim]
        
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        # output shape: [max_sequence_length, batch_size, embed_dim]
        
        return output
    
class MaskOCR(nn.Module):
    def __init__(self, img_height, img_width, patch_size, embed_dim, num_heads, num_encoder_layers, num_decoder_layers,
                 vocab_size, max_sequence_length, dropout, emb_dropout, overlap):
        super().__init__()
        self.encoder = MaskOCR_Encoder(img_height, img_width, patch_size, embed_dim, num_heads, num_encoder_layers, dropout, emb_dropout, overlap)
        self.decoder = MaskOCR_Decoder(embed_dim, num_heads, num_decoder_layers, max_sequence_length, dropout, emb_dropout)
        
        self.classifier = nn.Linear(embed_dim, vocab_size)
        
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        
        # For visual pre-training
        self.mae = MAE(
            encoder=self.encoder.vit,
            decoder_dim=embed_dim,
            masking_ratio=0.7,
            decoder_depth=4,
            decoder_heads=num_heads,
            decoder_dim_head=64
        )
        
        self.regressor = LatentContextualRegressor(embed_dim, num_heads, 4, max_sequence_length)
    
    def forward(self, images, mask=None):
        encoder_output = self.encoder(images, mask)
        decoder_output = self.decoder(encoder_output)
        decoder_output = decoder_output.transpose(0, 1)
        logits = self.classifier(decoder_output)
        logits = logits.view(-1, self.max_sequence_length, self.vocab_size)
        return logits
    
    def visual_pretraining_forward(self, images):
        return self.mae(images)
    
    def language_pretraining_forward(self, images, char_mask, patch_mask):
        # images shape: [batch_size, 3, img_height, img_width]
        # char_mask shape: [batch_size, max_sequence_length]
        # patch_mask shape: [batch_size, num_patches]
        
        with torch.no_grad():
            # encoder_output shape: [batch_size, num_patches, embed_dim]
            encoder_output = self.encoder(images, patch_mask)
        
        # masked_encoder_output shape: [batch_size, num_patches, embed_dim]
        masked_encoder_output = encoder_output.clone()
        masked_encoder_output[patch_mask] = 0
        
        # decoder_output shape: [max_sequence_length, batch_size, embed_dim]
        decoder_output = self.decoder(masked_encoder_output)
        
        # classifier output shape: [max_sequence_length, batch_size, vocab_size]
        return self.classifier(decoder_output), char_mask

def text_recognition_loss(predictions, targets, padding_idx):
    # predictions shape: [batch_size, max_sequence_length, vocab_size]
    # targets shape: [batch_size, max_sequence_length]
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx)
    
    # Reshape predictions and targets for loss calculation
    predictions = predictions.view(-1, predictions.size(-1))
    targets = targets.view(-1)
    
    return loss_fn(predictions, targets)


def train_visual_pretraining(model, dataloader, val_dataloader, device, vocab, num_epochs=10, start_epoch=0, temp_model_path=None,
                             version='', start_lr=1e-4, plateau_threshold=-1, use_wandb=False):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=start_lr, weight_decay=0.05, betas=(.9, .95))
    if plateau_threshold < 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    fig, axes, fig2, axes2 = None, None, None, None
    curr_lr = start_lr

    # Load optimizer and scheduler states if resuming
    if start_epoch > 0 and temp_model_path:
        checkpoint = torch.load(temp_model_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if plateau_threshold < 0:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    loss_history = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            
            optimizer.zero_grad()
            
            _, loss, _, _ = model.visual_pretraining_forward(images)
            
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
        
        avg_loss = np.mean(loss_history)


        # Calculate validation loss
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_images, _ in val_dataloader:
                val_images = val_images.to(device)
                _, val_loss, _, _ = model.visual_pretraining_forward(val_images)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)

        if plateau_threshold < 0:
            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0]
        else:
            if is_in_plateau(loss_history, threshold=plateau_threshold):
                # Reduce learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                    curr_lr = param_group['lr']
                loss_history = []
                print(f"{version} - Learning rate reduced to {curr_lr}")

        print(f"{version} - Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"lr: {curr_lr:.6f}, "
            f"Time: {time.time() - epoch_start_time:.2f} seconds")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "learning_rate": curr_lr,
                "epoch_time": time.time() - epoch_start_time,
            })

        # Plot examples and reconstructed images
        # fig2, axes2 = plot_reconstructed_images(fig2, axes2, model, val_dataloader, device)

        # Save checkpoint at the end of each epoch
        
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if plateau_threshold < 0:
            state_dict['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(state_dict, temp_model_path)

def visual_pretraining_loss(decoded_patches, true_patch_values, predicted_masked_representations, true_masked_representations, lambda_value=0.05):
    # decoded_patches shape: [num_masked_patches, batch_size, patch_dim]
    # true_patch_values shape: [num_masked_patches, batch_size, patch_dim]
    # predicted_masked_representations shape: [num_masked_patches, batch_size, embed_dim]
    # true_masked_representations shape: [num_masked_patches, batch_size, embed_dim]
    
    prediction_loss = nn.MSELoss()(decoded_patches, true_patch_values)
    alignment_loss = nn.MSELoss()(predicted_masked_representations, true_masked_representations)
    return prediction_loss + lambda_value * alignment_loss


def create_vertical_strip_mask(batch_size, num_patches, mask_ratio):
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    num_masked_patches = int(num_patches * mask_ratio)
    for i in range(batch_size):
        masked_indices = torch.randperm(num_patches)[:num_masked_patches]
        mask[i, masked_indices] = True
    return mask

def train_text_recognition(model, dataloader, val_dataloader, device, vocab, freeze_encoder, num_epochs=60, start_epoch=0,
                           temp_model_path=None, version='', start_lr=1e-4, plateau_threshold=-1, use_wandb=False):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=start_lr, weight_decay=0.05, betas=(.9, .95))
    if plateau_threshold < 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    fig, axes = None, None
    curr_lr = start_lr

    # Load optimizer and scheduler states if resuming
    if start_epoch > 0 and temp_model_path:
        checkpoint = torch.load(temp_model_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if plateau_threshold < 0:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Freeze Encoder?
    for param in model.encoder.parameters():
        param.requires_grad = not freeze_encoder
    
    loss_history = []
    for epoch in range(start_epoch, num_epochs, 1):
        epoch_start_time = time.time()
        model.train()
        for images, text_indices in dataloader:
            images = images.to(device)
            text_indices = text_indices.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = text_recognition_loss(outputs, text_indices, padding_idx=-1)
            
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
        
        avg_loss = np.mean(loss_history)

        # Calculate validation loss
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_images, val_text_indices in val_dataloader:
                val_images = val_images.to(device)
                val_text_indices = val_text_indices.to(device)
                
                val_outputs = model(val_images)
                val_loss = text_recognition_loss(val_outputs, val_text_indices, padding_idx=-1)
                
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)


        if plateau_threshold < 0:
            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0]
        else:
            if is_in_plateau(loss_history, threshold=plateau_threshold):
                # Reduce learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    curr_lr = param_group['lr']
                loss_history = []

                print(f"{version} - Learning rate reduced to {curr_lr}")
                

        print(f"{version} - Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"lr: {curr_lr:.6f}, "
            f"Time: {time.time() - epoch_start_time:.2f} seconds")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "learning_rate": curr_lr,
                "epoch_time": time.time() - epoch_start_time,
            })

        # Plot examples
        # fig, axes = plot_examples(fig, axes, model, val_dataloader, device, vocab)

        # Save checkpoint at the end of each epoch
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if plateau_threshold < 0:
            state_dict['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(state_dict, temp_model_path)

def create_char_and_patch_masks(batch_size, num_patches, num_chars, mask_ratio=0.15):
    char_mask = torch.rand(batch_size, num_chars) < mask_ratio
    patch_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    
    for i in range(batch_size):
        masked_chars = char_mask[i].nonzero().squeeze(1)  # Change: squeeze(1) instead of squeeze()
        
        # If no characters are masked, randomly mask one character
        if masked_chars.numel() == 0:
            masked_chars = torch.tensor([random.randint(0, num_chars - 1)])
            char_mask[i, masked_chars] = True
        
        for char in masked_chars:
            start_patch = (char * num_patches) // num_chars
            end_patch = ((char + 1) * num_patches) // num_chars
            patch_mask[i, start_patch:end_patch] = True
    
    return char_mask, patch_mask

def language_pretraining_loss(predictions, targets, char_mask):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    losses = loss_fn(predictions.view(-1, predictions.size(-1)), targets.view(-1))
    masked_losses = losses.view_as(targets)[char_mask]
    return masked_losses.mean()

def train_language_pretraining(model, dataloader, device, vocab, num_epochs=50):
    model.to(device)
    
    # Freeze the encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        for images, text_indices in dataloader:
            images = images.to(device)
            text_indices = text_indices.to(device)
            
            char_mask, patch_mask = create_char_and_patch_masks(images.size(0), model.encoder.num_patches, text_indices.size(1))
            char_mask, patch_mask = char_mask.to(device), patch_mask.to(device)
            
            optimizer.zero_grad()
            
            predictions, char_mask = model.language_pretraining_forward(images, char_mask, patch_mask)
            loss = language_pretraining_loss(predictions, text_indices, char_mask)
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")