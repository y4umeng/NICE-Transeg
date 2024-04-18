import torch
import torch.nn.functional as F
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
        
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        
        return grad

# def NJD(disp, device='cuda'):
#     # Negative Jacobian Determinant adapted from TransMorph repo
#     disp = torch.tensor(disp)
#     gradx  = torch.tensor([-0.5, 0, 0.5]).reshape(1, 3, 1, 1).to(device)
#     grady  = torch.tensor([-0.5, 0, 0.5]).reshape(1, 1, 3, 1).to(device)
#     gradz  = torch.tensor([-0.5, 0, 0.5]).reshape(1, 1, 1, 3).to(device)

#     gradx_disp = F.conv3d(input=disp, weight=gradx, padding=(0,1,0), stride=1).to(device)
#     grady_disp = F.conv3d(intput=disp, weight=grady, padding=(0,0,1), stride=1).to(device)
#     gradz_disp = F.conv3d(input=disp, weight=gradz, padding=(1,0,0), stride=1).to(device)

#     jacobian = torch.eye(3, device=device).reshape(1,3,3,1,1,1)+torch.stack([gradx_disp, grady_disp, gradz_disp], dim=2)

#     jacdet = torch.det(jacobian.squeeze(0))

#     return np.sum(jacdet<0) / np.prod(jacdet.shape)  
#     # not sure abt this
#     # negative_dets = (jacdet < 0).float().mean().item()
#     # return negative_dets

def NJD_new(disp):
    # Negative Jacobian Determinant adapted from TransMorph repo
    disp = torch.reshape(disp, (1, 3, 160, 192, 224))
    
    gradx  = torch.tensor([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = torch.tensor([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = torch.tensor([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = torch.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = torch.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = torch.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = torch.concat([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    return np.sum(jacdet<0) / np.prod(jacdet.shape) 

def NJD_old(disp):
    # Negative Jacobian Determinant adapted from TransMorph repo
    disp = np.reshape(disp, (1, 3, 160, 192, 224))
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_torch  = torch.tensor([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    disp_torch = torch.tensor(disp)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    gradx_disp_torch = torch.stack([F.conv3d(disp_torch[:, 0, :, :, :], gradx_torch, padding='same'),
                           F.conv3d(disp_torch[:, 1, :, :, :], gradx_torch, padding='same'),
                           F.conv3d(disp_torch[:, 2, :, :, :], gradx_torch, padding='same')], axis=1)
    
    print(f"NP SUM: {np.sum(gradx_disp)}")
    print(f'TORCH SUM : {torch.sum(gradx_disp_torch)}')
    

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    return np.sum(jacdet<0) / np.prod(jacdet.shape) 

class NJD_trans:
    def __init__(self, Lambda=1e-5):
        self.Lambda = Lambda
        
    def get_Ja(self, displacement):

        D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])
        D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])
        D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])

        D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
        D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
        D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
        
        return D1-D2+D3

    def loss(self, _, y_pred):

        displacement = y_pred.permute(0, 2, 3, 4, 1)
        Ja = self.get_Ja(displacement)
        Neg_Jac = 0.5*(torch.abs(Ja) - Ja)
    
        return self.Lambda*torch.sum(Neg_Jac)

class Regu_loss:
    def __init__(self, device='cuda'):
        self.device = device
    def loss(self, y_true, y_pred):
        return Grad('l2').loss(y_true, y_pred)
        # commented out for now
        # return Grad('l2').loss(y_true, y_pred) + 1e-5 * NJD(y_pred, self.device)