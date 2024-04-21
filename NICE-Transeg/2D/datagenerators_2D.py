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
    def __init__(self, data_path, device, atlas_path, file_type='*.npy', transform=torch.from_numpy):
        self.transform = transform
        self.device = device
        self.atlas = []
        self.atlas_labels = []

        # Paths for data
        self.files = glob(path.join(data_path, "data", file_type))
        print(f"Data file num: {len(self.files)}")

        # Load atlas files
        atlas_data_files = sorted(glob(path.join(atlas_path, "data", file_type)))
        atlas_label_files = sorted(glob(path.join(atlas_path, "label", file_type)))

        atlas_files = list(zip(atlas_data_files, atlas_label_files))
        print(f"Atlas file num: {len(atlas_files)}")
        for atlas_data, atlas_label in atlas_files:
            image = np.load(atlas_data)
            label = np.load(atlas_label)
            image = np.squeeze(image);
            label = np.squeeze(label);
            self.atlas.append(self.transform(image).float().unsqueeze(0).to(self.device))
            self.atlas_labels.append(self.transform(label).float().unsqueeze(0).to(self.device))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = np.load(self.files[idx], allow_pickle=False)
        atlas_idx = random.randint(0, len(self.atlas)-1)
        image = np.squeeze(image);
        return self.transform(image).float().unsqueeze(0).to(self.device), self.atlas[atlas_idx], self.atlas_labels[atlas_idx]

class NICE_Transeg_Dataset_Infer(Dataset):
    def __init__(self, data_path, device, file_type='*.npy', transform=torch.from_numpy):
        self.transform = transform
        self.device = device
        self.data = sorted(glob(path.join(data_path, "data", file_type)))
        self.labels = sorted(glob(path.join(data_path, "label", file_type)))
        if len(self.data) != len(self.labels): raise ValueError("The number of validation or testing images and labels do not match.")
        print(f"Validation/Test file num: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.load(self.data[idx], allow_pickle=False)
        label = np.load(self.labels[idx], allow_pickle=False)
        image = np.squeeze(image);
        label = np.squeeze(label);
        return self.transform(image).float().unsqueeze(0).to(self.device), self.transform(label).float().unsqueeze(0).to(self.device)

def print_gpu_usage(note=""):
    print(f"{note}: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024), flush=True)