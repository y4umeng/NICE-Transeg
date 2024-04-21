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
from torchmetrics.classification import MulticlassAccuracy

# project imports
from datagenerators_2D import NICE_Transeg_Dataset, NICE_Transeg_Dataset_Infer, print_gpu_usage
import networks_2D
import losses_2D

# git pull && python -u NICE-Transeg/2D/train_seg_2D.py --train_dir ./data/OASIS2D/Train/ --valid_dir ./data/OASIS2D/Val --atlas_dir ./data/OASIS2D/Atlas/ --device gpu1 --batch_size 2 --classes 25 -v
# git pull && python -u NICE-Transeg/2D/train_seg_2D.py --train_dir ./data/IXI2D/Train/ --valid_dir ./data/IXI2D/Val --atlas_dir ./data/IXI2D/Atlas/ --device gpu1 --batch_size 2 --classes 256 -v

# nohup python -u NICE-Transeg/2D/train_seg_2D.py --train_dir ./data/OASIS2D/Train/ --valid_dir ./data/OASIS2D/Val --atlas_dir ./data/OASIS2D/Atlas/ --load_model ./checkpoints/transeg2D_55_epoch_0.7599_dsc.pt --device gpu1 --model_dir ./transeg2D_2 --batch_size 2 > ./logs/transeg2D_oasis.txt &
# nohup python -u NICE-Transeg/2D/train_seg_2D.py --train_dir ./data/IXI2D/Train/ --valid_dir ./data/IXI2D/Val --atlas_dir ./data/IXI2D/Atlas/ --device gpu1 --batch_size 2 --classes 256 --model_dir transeg_IXI > ./logs/transeg2D_IXI.txt & 

# 1629025
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
          verbose,
          classes
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
    else:
        num_devices = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    # prepare model
    print("Initializing NICE-Transeg")
    model = networks_2D.NICE_Transeg(num_classes=classes, use_checkpoint=False, verbose=verbose) 

    if num_devices > 0:
        model = nn.DataParallel(model)

    model.to(device)

    if load_model != './':
        print('loading', load_model)
        state_dict = torch.load(load_model, map_location=device)
        model.load_state_dict(state_dict)
    
    # transfer model
    SpatialTransformer = networks_2D.SpatialTransformer_block(mode='nearest')
    SpatialTransformer.to(device)
    SpatialTransformer.eval()
    
    AffineTransformer = networks_2D.AffineTransformer_block(mode='nearest')
    AffineTransformer.to(device)
    AffineTransformer.eval()
    
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # prepare losses
    RegistrationLosses = [losses_2D.NCC(win=9).loss, losses_2D.Regu_loss().loss, losses_2D.NCC(win=9).loss]
    RegistrationWeights = [1.0, 1.0, 1.0]
    print(f'Registration Loss Weights: {RegistrationWeights}')
    label_weights = [1.0] * classes
    if classes==25: 
        label_weights = [1.601085151472262e-07, 1.2237820003695823e-06, 1.4143911470429324e-06, 1.8103807230660607e-05, 8.267195767195767e-05, 9.827430323519007e-06, 0.00011428571428571428, 2.6954177897574123e-05, 6.951202558042541e-05, 5.6551490131764975e-05, 1.3764056542744278e-05, 1.412030499858797e-05, 1.2167226359079185e-05, 0.0002808199943836001, 1.194974415597762e-06, 1.419855543896964e-06, 1.885369532428356e-05, 7.751937984496124e-05, 9.697157763059648e-06, 0.00012100677637947725, 3.173394262503174e-05, 9.372949667260286e-05, 1.3549768298962087e-05, 1.2108149995762147e-05, 0.00019474196689386563]
    elif classes==256:
        label_weights = [1.098744135453177e-05, 0, 5.261496369567505e-05, 5.746465923457074e-05, 0.0013003901170351106, 0, 0, 0, 0.009523809523809525, 0, 0.000357653791130186, 0.0014814814814814814, 0.0007616146230007616, 0.001141552511415525, 0.002257336343115124, 0, 0.02857142857142857, 0.0030581039755351682, 0, 0, 0, 0, 0, 0, 0.003745318352059925, 0, 0.05263157894736842, 0, 0.1, 0, 0, 0.019230769230769232, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.361642807356174e-05, 5.577555914998048e-05, 0.0015873015873015873, 0, 0, 0, 0.021739130434782608, 0, 0.00038226299694189603, 0.001314060446780552, 0.0006738544474393531, 0.0009784735812133072, 0.0024875621890547263, 0, 0, 0, 0, 0.034482758620689655, 0, 0, 0, 0, 0.014925373134328358, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01694915254237288, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.16666666666666666, 0.011627906976744186, 0.003289473684210526]
    assert(len(label_weights) == classes)
    label_weights = torch.tensor(label_weights).to(device)
    SegmentationLosses = [nn.CrossEntropyLoss(weight=label_weights), losses_2D.MulticlassDiceLoss(num_classes=classes)] 
    SegmentationWeights = [1.0, 1.0]
    print(f'Segmentation Loss Weights: {SegmentationWeights}')

    JointLosses = [losses_2D.MulticlassDiceLoss(num_classes=classes, logit_targets=True)]
    JointWeights = [1.0]
    print(f'Joint Loss Weights: {JointWeights}')

    NJD = losses_2D.NJD(device) 

    train_dl = DataLoader(NICE_Transeg_Dataset(train_dir, device, atlas_dir), batch_size=batch_size, shuffle=True, drop_last=False)
    valid_dl = DataLoader(NICE_Transeg_Dataset_Infer(valid_dir, device), batch_size=2, shuffle=False, drop_last=True)

    # training/validate loops
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()

        # training
        model.train()
        train_losses = []
        train_total_loss = []
        for image, atlas, atlas_seg in train_dl:
            assert(atlas.shape[0] == image.shape[0])
            batch_start_time = time.time()

            # forward pass
            if verbose: print_gpu_usage("before forward pass")
            pred = model(image, atlas)
            if verbose: print_gpu_usage("after forward pass")

            # registration loss calculation
            loss = 0
            loss_list = []
            registration_labels = [image, np.zeros((1)), image]

            for i, Loss in enumerate(RegistrationLosses):
                curr_loss = Loss(registration_labels[i], pred[i]) * RegistrationWeights[i]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            seg_fix = pred[3]
            seg_moving = pred[4]

            # segmentation cross entropy
            curr_loss = SegmentationLosses[0](seg_moving, atlas_seg.squeeze().long()) * SegmentationWeights[0]
            loss_list.append(curr_loss.item())
            loss += curr_loss

            # segmentation dice
            curr_loss = SegmentationLosses[1](seg_moving, atlas_seg.squeeze().long()) * SegmentationWeights[1]
            loss_list.append(curr_loss.item())
            loss += curr_loss 

            # joint dice
            warped_moving_seg = SpatialTransformer(seg_moving, pred[1]).squeeze()
            curr_loss = JointLosses[0](warped_moving_seg, seg_fix) * JointWeights[0]
            loss_list.append(curr_loss.item())
            loss += curr_loss 

            train_losses.append(loss_list)
            train_total_loss.append(loss.item())
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
        valid_seg_accuracy = []
        acc = MulticlassAccuracy(num_classes=classes).to(device)
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

                valid_seg_accuracy.append(acc(pred[-3], fixed_seg.squeeze(dim=0)).cpu().item())
                valid_seg_accuracy.append(acc(pred[-2], moving_seg.squeeze(dim=0)).cpu().item())

                fixed_seg = fixed_seg.detach().cpu().numpy().squeeze()
                warped_seg = warped_seg.detach().cpu().numpy().squeeze()
                Dice_val = Dice(warped_seg, fixed_seg)
                valid_Dice.append(Dice_val)

                if verbose: print_gpu_usage("after valid dice")
                
                affine_seg = affine_seg.detach().cpu().numpy().squeeze()
                Affine_val = Dice(affine_seg, fixed_seg)
                valid_Affine.append(Affine_val)

                if verbose: print_gpu_usage("after affine dice")

                NJD_val = NJD.loss(pred[1])
                valid_NJD.append(NJD_val.cpu().item())

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
        # valid_NJD_info = 'Valid NJD: %.5f' % (np.mean(valid_NJD))
        valid_seg_accuracy_info = 'Valid Seg Accuracy: %.4f' % (np.mean(valid_seg_accuracy))
        print(' - '.join((epoch_info, time_info, train_loss_info, valid_Dice_info, valid_Affine_info, valid_seg_accuracy_info)), flush=True)
        # save model checkpoint
        torch.save(model.state_dict(), os.path.join(model_dir, 'transeg2D_%02d_epoch_%.4f_dsc.pt' % (epoch+1, np.mean(valid_Dice))))
    

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
                        dest="epochs", default=1000,
                        help="number of epoch")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch size")
    parser.add_argument("--classes", type=int,
                        dest="classes", default=25,
                        help="number of classes for segmentation")
    parser.add_argument("-verbose", "-v", action='store_true')
    args = parser.parse_args()
    train(**vars(args))