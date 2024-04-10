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
        return self.transform(image).unsqueeze(0).to(self.device), self.transform(label).unsqueeze(0).to(self.device)
        # return torch.reshape(self.transform(image)[:,:,:144], (144, 192, 160)).unsqueeze(0).to(self.device), self.transform(label).unsqueeze(0).to(self.device)
        # return self.images[idx].unsqueeze(0), self.labels[idx].unsqueeze(0)
    
def print_gpu_usage(note=""):
    print(f"{note}: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))