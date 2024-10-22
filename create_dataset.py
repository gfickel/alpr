import itertools
import argparse
import os
import csv

from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(description='Converts ALPR datasets to test/train')
    parser.add_argument('--dataset_path', help='Dataset path')
    parser.add_argument('--dataset_name', choices=['ccpd2019'])
    return parser.parse_args()


def convert_ccpd2019(args):
    for category in ['ccpd_base', 'ccpd_weather', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_np', 'ccpd_tilt', 'ccpd_weather']:
        path = os.path.join(args.dataset_path, category)
        max_letter = 0

        with open(os.path.join(path, 'alpr_annotation.csv'), 'w', newline='') as fid:
            writer = csv.writer(fid, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['image_path', 'left', 'top', 'right', 'bottom',
                             'kp1_x', 'kp1_y', 'kp2_x', 'kp2_y', 'kp3_x', 'kp3_y', 'kp4_x', 'kp4_y',
                             'ocr_1', 'ocr_2', 'ocr_3', 'ocr_4', 'ocr_5', 'ocr_6', 'ocr_7'])
            num_imgs = len(os.listdir(path))
            print('num_images', num_imgs, path)
            for idx,im in enumerate(os.listdir(path)):
                im_path = os.path.join(path, im)
                if os.path.isdir(im_path) or '.jpg' not in im:
                    continue
                try:
                    _, _, bbox_coord, kps, plate, _, _ = im.split('-')
                except BaseException as e:
                    print(e)
                    print(im)
                    continue
                tl, br = bbox_coord.split('_')
                l, t = map(int, tl.split('&'))
                r, b = map(int, br.split('&'))

                kp_list = kps.split('_')
                kp_list = [[int(y) for y in x.split('&')] for x in kp_list]
                kp_list = list(itertools.chain(*kp_list))

                plate = plate.split('_')
                plate = [int(x)+1 for x in plate]
                max_letter = max(max_letter, max(plate))

                writer.writerow([os.path.abspath(im_path), l, t, r, b, *kp_list, *plate])
                print(f'{idx}/{num_imgs}', end='\r')
            
        print('\nMaximum letter: ', max_letter)

if __name__ == '__main__':
    args = get_args()
    if args.dataset_name == 'ccpd2019':
        convert_ccpd2019(args)
