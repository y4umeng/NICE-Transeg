import os, sys
import numpy as np
import scipy.ndimage
import torch
# from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
from os import path

class NICE_Transeg_Dataset(Dataset):
    def __init__(self, data_path, device, transform=torch.from_numpy):
        self.transform = transform
        self.device = device
        self.images = []
        self.labels = []
        files = glob(path.join(data_path, "*.pkl"))
        self.files = files
        print(f"{data_path.split('/')[-1]} file num: {len(files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, label = np.load(self.files[idx], allow_pickle=True)
        return self.transform(image).unsqueeze(0).to(self.device), self.transform(label).float().unsqueeze(0).to(self.device)
        # return torch.reshape(self.transform(image)[:,:,:144], (144, 192, 160)).unsqueeze(0).to(self.device), self.transform(label).unsqueeze(0).to(self.device)
    
def print_gpu_usage(note=""):
    print(f"{note}: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

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

print(process_label())