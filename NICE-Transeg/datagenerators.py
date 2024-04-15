import os, sys
import numpy as np
import scipy.ndimage
import torch
# from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
from os import path
import random

class NICE_Transeg_Dataset(Dataset):
    def __init__(self, data_path, device, atlas_path, transform=torch.from_numpy, file_type='*.pkl'):
        self.transform = transform
        self.device = device
        self.images = []
        self.labels = []
        self.atlas = []
        self.atlas_labels = []

        if(file_type=='*.pkl'):
            files = glob(path.join(data_path, file_type))
            self.files = files
            print(f"{data_path.split('/')[-1]} file num: {len(files)}")

            atlas_files = glob(path.join(atlas_path, file_type)) 
            print(f"{atlas_path.split('/')[-1]} file num: {len(atlas_files)}") 
            for atlas in atlas_files:
                image, label = np.load(atlas, allow_pickle=True)
                self.atlas.append(self.transform(image).unsqueeze(0).to(self.device))
                self.atlas_labels.append(self.transform(label).float().unsqueeze(0).to(self.device))
        
        else:
            # Paths for data and labels
            data_files = sorted(glob(path.join(data_path, "data", file_type)))
            label_files = sorted(glob(path.join(data_path, "label", file_type)))
            
            # Ensure that data and label files match
            if len(data_files) != len(label_files):
                raise ValueError("The number of data files and label files do not match.")

            self.files = list(zip(data_files, label_files))
            print(f"Data file num: {len(data_files)}")

            # Load atlas files
            atlas_data_files = sorted(glob(path.join(atlas_path, "data", file_type)))
            atlas_label_files = sorted(glob(path.join(atlas_path, "label", file_type)))

            atlas_files = list(zip(atlas_data_files, atlas_label_files))
            print(f"Atlas file num: {len(atlas_files)}")
            for atlas_data, atlas_label in atlas_files:
                image = np.load(atlas_data)
                label = np.load(atlas_label)
                self.atlas.append(self.transform(image).unsqueeze(0).to(self.device))
                self.atlas_labels.append(self.transform(label).float().unsqueeze(0).to(self.device))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, label = np.load(self.files[idx], allow_pickle=True)
        atlas_idx = random.randint(0, len(self.atlas)-1)
        return self.transform(image).unsqueeze(0).to(self.device), self.transform(label).float().unsqueeze(0).to(self.device), self.atlas[atlas_idx], self.atlas_labels[atlas_idx]


class NICE_Transeg_Dataset_Infer(Dataset):
    def __init__(self, data_path, device, transform=torch.from_numpy, file_type='*.pkl'):
        self.transform = transform
        self.device = device
        self.images = []
        self.labels = []
        files = glob(path.join(data_path, file_type))
        self.files = files
        print(f"{data_path.split('/')[-1]} file num: {len(files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, label = np.load(self.files[idx], allow_pickle=True)
        return self.transform(image).unsqueeze(0).to(self.device), self.transform(label).float().unsqueeze(0).to(self.device)
        # return torch.reshape(self.transform(image)[:,:,:144], (144, 192, 160)).unsqueeze(0).to(self.device), self.transform(label).unsqueeze(0).to(self.device)
    
def print_gpu_usage(note=""):
    print(f"{note}: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024), flush=True)

def process_label():
    #process labeling information for FreeSurfer
    import re
    seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255]


    file1 = open('/Users/yau/Documents/sem6/medical image anal/NICE-Transeg/NICE-Transeg/ixi_labels.txt', 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict