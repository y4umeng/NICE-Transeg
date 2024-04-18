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
from datagenerators import NICE_Transeg_Dataset, NICE_Transeg_Dataset_Infer, print_gpu_usage
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
    
def train(train_dir,
          valid_dir, 
          atlas_dir,
          model_dir,
          load_model,
          device,
          initial_epoch,
          epochs,
          batch_size,
          verbose
          ):

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # device handling
    if 'gpu' in device:
        num_devices = int(device[-1]) + 1
        assert(batch_size == num_devices)
        if num_devices == 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(num_devices)])
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        num_devices = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    # prepare model
    print("Initializing MINI NICE-Transeg")
    model = networks.NICE_Transeg(use_checkpoint=True)

    if num_devices > 0:
        model = nn.DataParallel(model)

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
    Losses = [losses.NCC(win=9).loss, losses.Regu_loss(device).loss, losses.NCC(win=9).loss]
    Weights = [1.0, 1.0, 1.0]

    train_dl = DataLoader(NICE_Transeg_Dataset(train_dir, device, atlas_dir), batch_size=batch_size, shuffle=True, drop_last=False)
    valid_dl = DataLoader(NICE_Transeg_Dataset_Infer(valid_dir, device), batch_size=2, shuffle=True, drop_last=True)

    # training/validate loops
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()

        # training
        model.train()
        train_losses = []
        train_total_loss = []
        for image, atlas, atlas_seg in train_dl:
            assert(atlas.shape[0] == image.shape[0])
            print(f'segmentation: {atlas_seg.shape}')
            batch_start_time = time.time()

            # forward pass
            if verbose: print_gpu_usage("before forward pass")
            pred = model(image, atlas)
            if verbose: print_gpu_usage("after forward pass")

            # registration loss calculation
            loss = 0
            loss_list = []
            reg_labels = [image, np.zeros((1)), image]
            for i, Loss in enumerate(Losses):
                curr_loss = Loss(reg_labels[i], pred[i]) * Weights[i]
                loss_list.append(curr_loss.detach().item())
                loss += curr_loss

            # segmentation loss calculation
            with torch.no_grad():
                warped_atlas_seg = SpatialTransformer(atlas_seg, pred[1])
                print(f'warped atlas seg: {warped_atlas_seg.shape}') 
                # cross = nn.DataParallel(nn.CrossEntropyLoss())(pred[3].detach().long(), warped_atlas_seg.squeeze().detach().long())
                seg_fix = pred[3].half()
                print(f'seg_fix: {seg_fix.shape}')
                softmaxed = nn.DataParallel(nn.LogSoftmax(dim=1))(seg_fix) 
                print(f"SOFTMAXED: {softmaxed.shape}")
                cross = nn.NLLLoss()(softmaxed, warped_atlas_seg.squeeze().detach().long()) 

            train_losses.append(loss_list)
            train_total_loss.append(loss.detach().item())
            if verbose: 
                print_gpu_usage("after loss calc")
                print(f"loss: {loss}")

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose: 
                print_gpu_usage("after backwards pass")
                print('Total %.2f sec' % (time.time() - batch_start_time))
        
        # validation
        if verbose: print("Validation begins.")
        model.eval()
        valid_Dice = []
        valid_Affine = []
        valid_NJD = []
        for valid_images, valid_labels in valid_dl:
            assert(valid_images.shape[0] == 2)
            batch_start_time = time.time()

            fixed_vol = valid_images[0][None,...].float()
            fixed_seg = valid_labels[0][None,...].float()

            moving_vol = valid_images[1][None,...].float()
            moving_seg = valid_labels[1][None,...].float()

            # run inputs through the model to produce a warped image and flow field
            with torch.no_grad():
                if verbose: print_gpu_usage("before validation forward pass")
                pred = model(fixed_vol, moving_vol)
                if verbose: print_gpu_usage("after validation forward pass")
                warped_seg = SpatialTransformer(moving_seg, pred[1])
                affine_seg = AffineTransformer(moving_seg, pred[-1])
                
                fixed_seg = fixed_seg.detach().cpu().numpy().squeeze()
                warped_seg = warped_seg.detach().cpu().numpy().squeeze()
                Dice_val = Dice(warped_seg, fixed_seg)
                valid_Dice.append(Dice_val)

                if verbose: print_gpu_usage("after valid dice")
                
                affine_seg = affine_seg.detach().cpu().numpy().squeeze()
                Affine_val = Dice(affine_seg, fixed_seg)
                valid_Affine.append(Affine_val)

                if verbose: print_gpu_usage("after affine dice")

                flow = pred[1].detach().cpu().permute(0, 2, 3, 4, 1).numpy().squeeze()
                NJD_val = losses.NJD(flow)
                valid_NJD.append(NJD_val)

                if verbose: 
                    print_gpu_usage("after njd")
                    print('Total Validation %.2f sec' % (time.time() - batch_start_time)) 
        
        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
        time_info = 'Total %.2f sec' % (time.time() - start_time)
        train_losses = ', '.join(['%.4f' % f for f in np.mean(train_losses, axis=0)])
        train_loss_info = 'Train loss: %.4f  (%s)' % (np.mean(train_total_loss), train_losses)
        valid_Dice_info = 'Valid final DSC: %.4f' % (np.mean(valid_Dice))
        valid_Affine_info = 'Valid affine DSC: %.4f' % (np.mean(valid_Affine))
        valid_NJD_info = 'Valid NJD: %.5f' % (np.mean(valid_NJD))
        print(' - '.join((epoch_info, time_info, train_loss_info, valid_Dice_info, valid_Affine_info, valid_NJD_info)), flush=True)
        # save model checkpoint
        torch.save(model.state_dict(), os.path.join(model_dir, '%02d_epoch_%.4f_dsc.pt' % (epoch+1, np.mean(valid_Dice))))
    

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
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch size")
    parser.add_argument("-verbose", "-v", action='store_true')
    args = parser.parse_args()
    train(**vars(args))