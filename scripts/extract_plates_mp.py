import os
import shutil
import pandas as pd
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm
import argparse

def crop_and_save_image(row, input_dir, output_dir=''):
    """
    Function to crop an image based on the bounding box coordinates
    and save the cropped image to the specified output directory.
    """
    image_path = row['image_path']
    left = row['left']
    top = row['top']
    right = row['right']
    bottom = row['bottom']

    # Open the image
    try:
        curr_im_path = os.path.join(input_dir, os.path.basename(image_path))
        image = Image.open(curr_im_path)
    except Exception as e:
        print(f"Error opening image {curr_im_path}: {e}")
        return
    
    # Define the crop box (left, top, right, bottom)
    crop_box = (left, top, right, bottom)
    # Crop the image
    cropped_image = image.crop(crop_box)
    # cropped_image = cropped_image.resize((128, 32))
    cropped_image = cropped_image.resize((192, 48))

    # Define the output path
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    # Save the cropped image
    try:
        cropped_image.save(output_path)
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")

def process_images(csv_file, input_dir, output_dir, num_workers):
    """
    Function to process images from a CSV file using multiprocessing.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(csv_file, output_dir)

    # Create a pool of workers
    pool = mp.Pool(num_workers)

    # Apply the cropping function in parallel
    args = [(row, input_dir, output_dir) for _, row in df.iterrows()]
    for _ in tqdm(pool.starmap(crop_and_save_image, args), total=len(df)):
        pass

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process CCPD2019 dataset images')
    parser.add_argument('--ccpd_dir', type=str, required=True,
                        help='Path to CCPD2019 directory')
    parser.add_argument('--categories', type=str, nargs='+', 
                        default=['ccpd_base'],
                        help='Categories to process (default: ccpd_base)')
    parser.add_argument('--num_workers', type=int, 
                        default=mp.cpu_count(),
                        help='Number of worker processes (default: number of CPU cores)')
    args = parser.parse_args()

    for category in args.categories:
        # Define the path to the CSV file
        csv_file = os.path.join(args.ccpd_dir, category, 'alpr_annotation.csv')
        # Define the output directory for cropped images
        output_dir = f'../alpr_datasets/plates_{category}_48'
        input_dir = os.path.join(args.ccpd_dir, category)
        
        # Process the images
        process_images(csv_file, input_dir, output_dir, args.num_workers)