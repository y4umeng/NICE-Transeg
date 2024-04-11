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
from datagenerators import NICE_Transeg_Dataset
import networks
import losses


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
    
    
def NJD(displacement):

    D_y = (displacement[1:,:-1,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_x = (displacement[:-1,1:,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_z = (displacement[:-1,:-1,1:,:] - displacement[:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    Ja_value = D1-D2+D3
    
    return np.sum(Ja_value<0)

# train on 3 cuda gpu
# python NICE-Transeg/train.py --train_dir ./data/IXI/Train/ --valid_dir ./data/IXI/Val --atlas_dir ./data/IXI/Atlas/ --device gpu0 

def train(train_dir,
          valid_dir, 
          atlas_dir,
          model_dir,
          load_model,
          device,
          initial_epoch,
          epochs,
          steps_per_epoch,
          batch_size):

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # prepare model
    model = networks.NICE_Trans(use_checkpoint=True)

    # device handling
    if 'gpu' in device:
        gpus = int(device[-1])
        print(f'{gpus + 1} GPUs requested')
        print(f'{torch.cuda.device_count()} GPUs found')

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(gpus+1)])
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
        if device[-1] != '0': 
            model = nn.DataParallel(model)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'
        
    model.to(device)

    if load_model != './':
        print('loading', load_model)
        state_dict = torch.load(load_model, map_location=device)
        model.load_state_dict(state_dict)
    
    # transfer model
    SpatialTransformer = networks.SpatialTransformer_block(mode='nearest')
    SpatialTransformer.to(device)
    SpatialTransformer.eval()
    
    AffineTransformer = networks.AffineTransformer_block(mode='nearest')
    AffineTransformer.to(device)
    AffineTransformer.eval()
    
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # prepare losses
    Losses = [losses.NCC(win=9).loss, losses.Regu_loss, losses.NCC(win=9).loss]
    Weights = [1.0, 1.0, 1.0]

    train_dl = DataLoader(NICE_Transeg_Dataset(train_dir, device), batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(NICE_Transeg_Dataset(valid_dir, device), batch_size=2, shuffle=True)
    atlas_dl = DataLoader(NICE_Transeg_Dataset(atlas_dir, device), batch_size=1, shuffle=True)
    
    # training/validate loops
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()

        # training
        model.train()
        train_losses = []
        train_total_loss = []
        for images, _ in train_dl:
            for atlas, _ in atlas_dl:
                pred = model(images, atlas)

                loss = 0
                loss_list = []
                for i, Loss in enumerate(Losses):
                    curr_loss = Loss(atlas, pred[i]) * Weights[i]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss

                train_losses.append(loss_list)
                train_total_loss.append(loss.item())

                # backpropagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # validation
        print("Validation begins.")
        model.eval()
        valid_Dice = []
        valid_Affine = []
        valid_NJD = []
        for valid_images, valid_labels in valid_dl:

            fixed_vol = valid_images[0][None,...].float()
            fixed_seg = valid_labels[0][None,...].float()

            moving_vol = valid_images[1][None,...].float()
            moving_seg = valid_labels[1][None,...].float()

            # run inputs through the model to produce a warped image and flow field
            with torch.no_grad():
                pred = model(fixed_vol, moving_vol)
                warped_seg = SpatialTransformer(moving_seg, pred[1])
                affine_seg = AffineTransformer(moving_seg, pred[3])
                
            fixed_seg = fixed_seg.detach().cpu().numpy().squeeze()
            warped_seg = warped_seg.detach().cpu().numpy().squeeze()
            Dice_val = Dice(warped_seg, fixed_seg)
            valid_Dice.append(Dice_val)
            
            affine_seg = affine_seg.detach().cpu().numpy().squeeze()
            Affine_val = Dice(affine_seg, fixed_seg)
            valid_Affine.append(Affine_val)
            
            flow = pred[1].detach().cpu().permute(0, 2, 3, 4, 1).numpy().squeeze()
            NJD_val = NJD(flow)
            valid_NJD.append(NJD_val)
        
        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
        time_info = 'Total %.2f sec' % (time.time() - start_time)
        train_losses = ', '.join(['%.4f' % f for f in np.mean(train_losses, axis=0)])
        train_loss_info = 'Train loss: %.4f  (%s)' % (np.mean(train_total_loss), train_losses)
        valid_Dice_info = 'Valid final DSC: %.4f' % (np.mean(valid_Dice))
        valid_Affine_info = 'Valid affine DSC: %.4f' % (np.mean(valid_Affine))
        valid_NJD_info = 'Valid NJD: %.2f' % (np.mean(valid_NJD))
        print(' - '.join((epoch_info, time_info, train_loss_info, valid_Dice_info, valid_Affine_info, valid_NJD_info)), flush=True)
    
        # save model checkpoint
        torch.save(model.state_dict(), os.path.join(model_dir, '%02d.pt' % (epoch+1)))
    

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
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='./models/',
                        help="models folder")
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='./',
                        help="load model file to initialize with")
    parser.add_argument("--device", type=str, default='cuda',
                        dest="device", help="cpu or cuda")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="initial epoch")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=100,
                        help="number of epoch")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=1,
                        help="iterations of each epoch")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch size")

    args = parser.parse_args()
    train(**vars(args))