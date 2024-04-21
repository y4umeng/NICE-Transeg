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
from datagenerators_2D import NICE_Transeg_Dataset_Infer, NICE_Transeg_Dataset, print_gpu_usage
import networks_2D
import losses_2D

# git pull && python -u NICE-Transeg/2D/get_weights_for_loss.py --train_dir ./data/OASIS2D/Train/ --valid_dir ./data/OASIS2D/Val --atlas_dir ./data/OASIS2D/Atlas/ --device gpu0
# git pull && python -u NICE-Transeg/2D/get_weights_for_loss.py --train_dir ./data/IXI2D/Train/ --valid_dir ./data/IXI2D/Val --atlas_dir ./data/IXI2D/Atlas/ --device gpu0
# nohup python -u NICE-Transeg/2D/get_weights_for_loss.py --train_dir ./data/BraTS2D/Train/ --valid_dir ./data/BraTS2D/Val --atlas_dir ./data/BraTS2D/Atlas/ --device gpu0 > ./brats_weights.txt &
# 4081777
def train(train_dir, 
          valid_dir, 
          atlas_dir,
          device,
          classes
          ):

    # OASIS: [6245764.0, 817139.0, 707018.0, 55237.0, 12096.0, 101756.0, 8750.0, 37100.0, 14386.0, 17683.0, 72653.0, 70820.0, 82188.0, 3561.0, 836838.0, 704297.0, 53040.0, 12900.0, 103123.0, 8264.0, 31512.0, 10669.0, 73802.0, 82589.0, 5135.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
    counter = {} 
    total = 0.0
    with torch.no_grad():
        for _, valid_labels in atlas_dl:
            print(valid_labels.shape)
            print(torch.max(valid_labels))
            for label in torch.flatten(valid_labels):
                
                if label.item() in counter:
                    counter[label.item()]+=1
                else: counter[label.item()] = 1
                total += 1
        # for _, valid_labels in valid_dl:
        #     for label in torch.flatten(valid_labels):
        #         counter[int(label.item())]+=1
        #         total += 1
        # for _, valid_labels in atlas_dl:
        #     for label in torch.flatten(valid_labels):
        #         counter[int(label.item())]+=1
        #         total += 1
        # while counter[-1] == 0:
        #     counter = counter[:-1]
        print(f"Number of labels: {len(counter)}")
        print("WEIGHTS:")
        # counter = [1.0/o for o in counter]
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