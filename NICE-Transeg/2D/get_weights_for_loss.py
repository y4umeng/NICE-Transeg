import os
import glob
import sys
import random
import time
import torch
import numpy as np
import scipy.ndimage
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.nn as nn

# project imports
from datagenerators_2D import NICE_Transeg_Dataset, NICE_Transeg_Dataset_Infer, print_gpu_usage
import networks_2D
import losses_2D

# git pull && python -u NICE-Transeg/2D/get_weights_for_loss.py --valid_dir ./data/OASIS2D/Train --device gpu0

# nohup python -u NICE-Transeg/2D/train_seg_2D.py --train_dir ./data/OASIS2D/Train/ --valid_dir ./data/OASIS2D/Val --atlas_dir ./data/OASIS2D/Atlas/ --load_model ./checkpoints/transeg2D_55_epoch_0.7599_dsc.pt --device gpu1 --model_dir ./transeg2D_2 --batch_size 2 > ./logs/transeg2D_oasis.txt &

# 1256188
    
def train(train_dir, 
          valid_dir, 
          atlas_dir,
          device,
          classes
          ):


    # device handling
    if 'gpu' in device:
        num_devices = int(device[-1]) + 1
        if num_devices == 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(num_devices)])
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        num_devices = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    train_dl = DataLoader(NICE_Transeg_Dataset_Infer(train_dir, device), batch_size=1, shuffle=False, drop_last=False)
    valid_dl = DataLoader(NICE_Transeg_Dataset_Infer(valid_dir, device), batch_size=1, shuffle=False, drop_last=False)
    atlas_dl = DataLoader(NICE_Transeg_Dataset_Infer(atlas_dir, device), batch_size=1, shuffle=False, drop_last=False) 
    counter = [0.0] * classes
    total = 0.0
    for _, valid_labels in train_dl:
        for label in torch.flatten(valid_labels):
            counter[int(label.item())]+=1
            total += 1
    for _, valid_labels in valid_dl:
        for label in torch.flatten(valid_labels):
            counter[int(label.item())]+=1
            total += 1
    for _, valid_labels in atlas_dl:
        for label in torch.flatten(valid_labels):
            counter[int(label.item())]+=1
            total += 1
    # counter = [c/total for c in counter]
    print(f"Num expected classes: {classes}")
    print(f"LABEL WEIGHTS:")
    print(counter)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_dir", type=str,
                        dest="train_dir", default='./',
                        help="folder with training data")
    parser.add_argument("--valid_dir", type=str,
                        dest="valid_dir", default='./',
                        help="folder with validation data")
    parser.add_argument("--atlas_dir", type=str,
                        dest="atlas_dir", default='./',
                        help="folder with atlas data")
    parser.add_argument("--device", type=str, default='cuda',
                        dest="device", help="cpu or cuda")
    parser.add_argument("--classes", type=int,
                        dest="classes", default=36,
                        help="number of classes for segmentation")
    args = parser.parse_args()
    train(**vars(args))