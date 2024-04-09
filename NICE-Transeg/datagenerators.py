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
        self.images = []
        self.labels = []
        files = glob(path.join(data_path, "*.pkl"))
        self.files = files
        print(f"file num: {len(files)}")
        # for f in files:
        #     image, label = np.load(f, allow_pickle=True)
        #     self.images.append(torch.reshape(transform(image)[:,:,:144], (144, 192, 160)))
        #     self.labels.append(transform(label))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, label = np.load(self.files[idx], allow_pickle=True)
        return torch.reshape(self.transform(image)[:,:,:144], (144, 192, 160)), self.transform(label)
        # return self.images[idx].unsqueeze(0), self.labels[idx].unsqueeze(0)
    
def print_gpu_usage(note=""):
    print(note)
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))