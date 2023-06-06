import os
import glob
import argparse
import numpy as np
import cv2
from tqdm.auto import tqdm as tq
from joblib import Parallel, delayed



def make_dir(directory:str):
    if not os.path.exists(directory):
        os.makedirs(directory) 


def center_crop(img, target_size=320):
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

def resize(img, target_size=320):
    return cv2.resize(img, (target_size,target_size), interpolation=cv2.INTER_LANCZOS4)

def process(pid, copy_from_list, copy_to_list, target_size=320):
    img = cv2.imread(copy_from_list[pid])
    img = resize(center_crop(img, target_size=target_size), target_size=target_size)
    target_dir = copy_to_list[pid]
    cv2.imwrite(target_dir, img.astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', action='store', default='valid')
    parser.add_argument('-s', '--size', action='store', default=320)
    args = parser.parse_args()
    mode = args.mode
    size = int(args.size)

    # mode = 'train'
    # size = 320
    image_folder = "/data/s1/gyuseong/CheXpert-v1.0"
    target_folder = f"/data/s1/gyuseong/chexpert-resized/{mode}_{size}"

    make_dir(target_folder)
    
    start_patient = 1 if mode == 'train' else 64541
    end_patient = 65240+1 if mode == 'train' else 64740+1
    # end_patient = 10

    print("-- load image directories...")
    d = sum([glob.glob(f"{image_folder}/{mode}/patient{str(idx).zfill(5)}/study*/*.jpg") for idx in tq(range(start_patient, end_patient))], [])

    d = [d[idx] for idx in range(len(d)) if 'lateral' not in d[idx]]

    patients = [d[idx].split(f"v1.0/{mode}/")[1].split("/study")[0] for idx in range(len(d))]
    studies  = [d[idx].split(f"{patients[idx]}")[1].split("/view")[0].replace("/", "") for idx in range(len(d))]
    filename = [d[idx].split(f"{studies[idx]}")[1].replace("/", "") for idx in range(len(d))]
    
    copy_from_list = d[:]
    copy_to_list   = [f"{target_folder}/{patients[pid]}_{studies[pid]}_{filename[pid]}" for pid in range(len(d))]

    print("-- process images...")
    Parallel(n_jobs=-1)(delayed(process)(pid, copy_from_list, copy_to_list, size)
    for pid in tq(range(len(copy_from_list))))    
