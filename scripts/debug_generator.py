import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from dataloader import ALPRDataset

# Assuming SyntheticOCRDataset is in a file named synthetic_ocr_dataset.py
# from maskocr_pretrain_simple_v2 import SyntheticOCRDataset

def save_synthetic_ocr_images(output_dir, num_images=100):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define vocabulary (adjust as needed)
    vocab = list('0123456789')

    # Create the dataset
    # dataset = SyntheticOCRDataset(vocab=vocab, num_samples=num_images, seq_length=6)

    # # Create a DataLoader (not strictly necessary, but useful if you want to process in batches)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandAugment(num_ops=2, magnitude=6),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img.float() / 1.0),  # Convert to float and scale to [0, 1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])
    ])
    dataset = ALPRDataset(
        '../output_images_plates_gt/alpr_annotation.csv',
        '../output_images_plates_gt/', #None)
        data_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1, drop_last=True)


    # Generate and save images
    for i, (image, text_indices) in enumerate(dataloader):
        # Convert text indices back to string
        # text = ''.join([vocab[idx] for idx in text_indices[0] if vocab[idx] != '<pad>'])
        text = 'dummy'

        # Denormalize the image
        # image = image * 0.5# + 0.5

        # Save the image
        image_path = os.path.join(output_dir, f'image_{i:04d}.png')
        save_image(image, image_path)

        # Save the text in a separate file
        text_path = os.path.join(output_dir, f'image_{i:04d}.txt')
        with open(text_path, 'w') as f:
            f.write(text)

        print(f"Saved image and text for sample {i}: {text}")

if __name__ == '__main__':
    output_directory = 'synthetic_ocr_debug_images'
    num_images_to_generate = 100

    save_synthetic_ocr_images(output_directory, num_images_to_generate)
    print(f"Generated {num_images_to_generate} images in '{output_directory}'")