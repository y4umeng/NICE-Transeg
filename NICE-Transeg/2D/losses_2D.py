import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import scipy


class NCC:
    def __init__(self, win=9):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_pred.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class Grad:
    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        
        # dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        # dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        # dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            # dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) #+ torch.mean(dz)
        grad = d / 2.0 #3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        
        return grad

class NJD:
    def __init__(self, device):
        # Initialize gradient kernels for 2D
        self.gradx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), padding='same', bias=False)
        self.gradx.weight = nn.Parameter(torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).reshape(1, 1, 3, 1).to(device))
        
        self.grady = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3), padding='same', bias=False)
        self.grady.weight = nn.Parameter(torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).reshape(1, 1, 1, 3).to(device))
        
        # Identity matrix for 2D
        self.eye = torch.eye(2, 2).reshape(2, 2, 1, 1).to(device)

    def batched_loss(self, batched_disp):
        N, _, H, W = batched_disp.shape
        loss = 0
        for n in range(N):
            loss += self.loss(batched_disp[n].unsqueeze(0))
        return loss / N

    def loss(self, disp):
        N, _, H, W = disp.shape  # batch_size, 2, H, W
        disp = torch.reshape(disp.permute(0, 2, 3, 1), (N, 2, H, W))

        gradx_disp = torch.stack([self.gradx(disp[:, i, :, :]) for i in range(2)], axis=1)
        grady_disp = torch.stack([self.grady(disp[:, i, :, :]) for i in range(2)], axis=1)

        grad_disp = torch.stack([gradx_disp, grady_disp], axis=0)
        
        jacobian = grad_disp + self.eye
        jacobian = jacobian[:, :, :, 1:-1, 1:-1]  # Adjust slicing to avoid border issues
        
        # Compute determinant of the 2x2 Jacobian
        jacdet = jacobian[0, 0, :, :, :] * jacobian[1, 1, :, :, :] - jacobian[0, 1, :, :, :] * jacobian[1, 0, :, :, :]
        
        return torch.sum(jacdet < 0) / torch.prod(torch.tensor(jacdet.shape))
    
class Regu_loss:
    def __init__(self, device='cuda'):
        self.NJD = NJD(device)
    def loss(self, y_true, y_pred):
        return Grad('l2').loss(y_true, y_pred) + 1e-5 * self.NJD.batched_loss(y_pred)