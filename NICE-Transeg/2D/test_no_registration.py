# py imports
import os
import sys
import glob
import time
import numpy as np
import torch
import scipy.ndimage
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.nn as nn

# project imports
from datagenerators_2D import NICE_Transeg_Dataset_Infer
import networks_2D
import losses_2D as losses

# git pull && python -u NICE-Transeg/2D/test_no_registration.py --test_dir ./data/OASIS2D/Test/ --device gpu0
# git pull && python -u NICE-Transeg/2D/test_2D.py --test_dir ./data/IXI2D/Test/ --device gpu0 --load_model ./checkpoints/

def Dice(vol1, vol2, labels=None, nargout=1):
    
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return np.mean(dicem)
    else:
        return (dicem, labels)

def test(test_dir,
         device, 
         load_model):
    
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
    
    

    # prepare model
    model = networks_2D.NICE_Trans()
    print('loading', load_model)
    # state_dict = torch.load(load_model, map_location=device)
    # # load the state dictionary that was saved with 'module.' prefix
    # new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    # model.load_state_dict(new_state_dict)
    if num_devices > 0:
        model = nn.DataParallel(model)
    #load data
    test_pairs = DataLoader(NICE_Transeg_Dataset_Infer(test_dir, device), batch_size=2*num_devices, shuffle=False, drop_last=True)
    model.to(device)
    model.eval()

    # transfer model
    SpatialTransformer = networks_2D.SpatialTransformer_block(mode='nearest')
    SpatialTransformer.to(device)
    SpatialTransformer.eval()
    
    AffineTransformer = networks_2D.AffineTransformer_block(mode='nearest')
    AffineTransformer.to(device)
    AffineTransformer.eval()

    NJD = losses.NJD(device)
    
    # testing loop
    Dice_result = [] 
    NJD_result = []
    Affine_result = []
    Runtime_result = []
    for test_images, test_labels in test_pairs:

        fixed_vol = test_images[0][None,...].float()
        fixed_seg = test_labels[0][None,...].float()

        moving_vol = test_images[1][None,...].float()
        moving_seg = test_labels[1][None,...].float()

        t = time.time()
        # with torch.no_grad():
        #     pred = model(fixed_vol, moving_vol)
        Runtime_val = time.time() - t
        
        warped_seg = moving_seg
        affine_seg = moving_seg
        
        fixed_seg = fixed_seg.detach().cpu().numpy().squeeze()
        warped_seg = warped_seg.detach().cpu().numpy().squeeze()
        affine_seg = affine_seg.detach().cpu().numpy().squeeze()
        
        Dice_val = Dice(warped_seg, fixed_seg)
        Dice_result.append(Dice_val)
        
        Affine_val = Dice(affine_seg, fixed_seg)
        Affine_result.append(Affine_val)
        
        # NJD_val = NJD.loss(pred[1])
        # NJD_result.append(NJD_val.cpu().item())
        
        
        Runtime_result.append(Runtime_val)
        
        # print('Final Dice: {:.3f} ({:.3f})'.format(np.mean(Dice_val), np.std(Dice_val)))
        # print('Affine Dice: {:.3f} ({:.3f})'.format(np.mean(Affine_val), np.std(Affine_val)))
        # print('NJD: {:.3f}'.format(NJD_val))
        # print('Runtime: {:.3f}'.format(Runtime_val))

    Dice_result = np.array(Dice_result)
    print('Average Final Dice: {:.3f} ({:.3f})'.format(np.mean(Dice_result), np.std(Dice_result)))
    Affine_result = np.array(Affine_result)
    print('Average Affine Dice: {:.3f} ({:.3f})'.format(np.mean(Affine_result), np.std(Affine_result)))
    NJD_result = np.array(NJD_result)
    print('Average NJD: {:.3f} ({:.3f})'.format(np.mean(NJD_result), np.std(NJD_result)))
    Runtime_result = np.array(Runtime_result)
    print('Average Runtime mean: {:.7f} ({:.7f})'.format(np.mean(Runtime_result[1:]), np.std(Runtime_result[1:])))

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--test_dir", type=str,
                        dest="test_dir", default='./',
                        help="folder with testing data")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="cpu or gpuN")
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='./',
                        help="load model file to initialize with")

    args = parser.parse_args()
    test(**vars(args))