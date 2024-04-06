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
        self.images = []
        self.labels = []
        files = glob(path.join(data_path, "*.pkl"))
        print(f"file num: {len(files)}")
        for f in files:
            image, label = np.load(f, allow_pickle=True)
            self.images.append(transform(image).to(device))
            self.labels.append(transform(label).to(device))
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]