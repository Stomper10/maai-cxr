import os
import glob 
import pandas as pd
import numpy as np
import cv2
# import matplotlib.pyplot as plt
from tqdm.auto import tqdm as tq
from joblib import Parallel, delayed

def make_dir(directory:str):
    if not os.path.exists(directory):
        os.makedirs(directory) 
        
def center_crop(img, target_size=384):
    h, w, c = img.shape
    if (h<target_size) | (w<target_size):
        set_size = max(h, w)
    elif h>w:
        set_size = w
    else:
        set_size = h
    crop_width = set_size
    crop_height = set_size
    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2
    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
    return crop_img

def resize(img, target_size=384):
    return cv2.resize(img, (target_size,target_size), interpolation=cv2.INTER_LANCZOS4)


if __name__ == '__main__':
    
    output_folder='/home/n1/gyuseonglee/workspace/datasets/chexpert-processed/train'
    make_dir(output_folder)
    
    def process(img_list, img_idx, target_size=384, output_folder=output_folder, extension="jpg"):
        img_path = img_list[img_idx]
        img = cv2.imread(img_path)
        img = resize(center_crop(img, target_size=target_size), target_size=target_size)
        filename = f"{output_folder}/" + '-'.join(img_path.replace('.jpg', '').split('/')[-3:]) + f'.{extension}'
        cv2.imwrite(filename, img.astype(np.uint8))

    imgs = pd.read_csv('/home/n1/gyuseonglee/workspace/datasets/chexpert/CheXpert-v1.0/train.csv')
    imgs['Path'] = imgs['Path'].str.replace('CheXpert-v1.0', '/home/n1/gyuseonglee/workspace/datasets/chexpert/CheXpert-v1.0', regex=False)
    imgs = imgs.Path.values


    Parallel(n_jobs=16)(
        delayed(process)(imgs, idx) for idx in tq(range(len(imgs)))
    )