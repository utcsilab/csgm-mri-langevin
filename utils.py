import torch
import torch.fft as torch_fft
import numpy as np
import sigpy as sp
from torch import nn
from torch.nn import functional as F
from PIL import Image
import os
from torchvision import transforms
import shutil
from collections import OrderedDict
import math
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision
from datetime import datetime
import glob
import torch.distributed as dist
import h5py
import functools
import logging
import warnings
import pickle
import re
import copy
from torch.optim import Adam

class SphericalOptimizer():
    def __init__(self, params):
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True)+1e-9).sqrt() for param in params}
    @torch.no_grad()
    def step(self, closure=None):
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9).sqrt())
            param.mul_(self.radii[param])


class MappingProxy(nn.Module):
    def __init__(self, gaussian_ft):
        super(MappingProxy, self).__init__()
        self.mean = torch.nn.Parameter(gaussian_ft["mean"], requires_grad=False)
        self.std = torch.nn.Parameter(gaussian_ft["std"], requires_grad=False)
        self.lrelu = torch.nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.lrelu(self.std * x + self.mean)
        return x


class BicubicDownSample(nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.:
            return (a + 2.) * torch.pow(abs_x, 3.) - (a + 3.) * torch.pow(abs_x, 2.) + 1
        elif 1. < abs_x < 2.:
            return a * torch.pow(abs_x, 3) - 5. * a * torch.pow(abs_x, 2.) + 8. * a * abs_x - 4. * a
        else:
            return 0.0

    def __init__(self, factor=4, device='cuda', padding='reflect'):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor([self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor)
                          for i in range(size)], dtype=torch.float32)
        k = k / torch.sum(k)
        # k = torch.einsum('i,j->ij', (k, k))
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0).float().to(device)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0).float().to(device)
        self.device = device
        self.padding = padding
        #self.padding = 'constant'
        #self.padding = 'replicate'
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor
        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filters1 = self.k1
        filters2 = self.k2
        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        # apply mirror padding
        if nhwc:
            x = torch.transpose(torch.transpose(
                x, 2, 3), 1, 2)   # NHWC to NCHW
        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        x = F.conv2d(input=x.float(), weight=filters1, stride=(stride, 1), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)
        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)
        if nhwc:
            x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        if byte_output:
            return x.type('torch.ByteTensor'.format(self.cuda))
        else:
            return x

# Multicoil forward operator for MRI
class MulticoilForwardMRI(nn.Module):
    def __init__(self, orientation):
        self.orientation = orientation
        super(MulticoilForwardMRI, self).__init__()
        return

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x

    # Centered, orthogonal fft in torch >= 1.7
    def _fft(self, x):
        x = torch_fft.fftshift(x, dim=(-2, -1))
        x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        return x


    '''
    Inputs:
     - image = [B, H, W] torch.complex64/128    in image domain
     - maps  = [B, C, H, W] torch.complex64/128 in image domain
     - mask  = [B, W] torch.complex64/128 w/    binary values
    Outputs:
     - ksp_coils = [B, C, H, W] torch.complex64/128 in kspace domain
    '''
    def forward(self, image, maps, mask):
        # Broadcast pointwise multiply
        coils = image[:, None] * maps

        # Convert to k-space data
        ksp_coils = self._fft(coils)

        if self.orientation == 'vertical':
            # Mask k-space phase encode lines
            ksp_coils = ksp_coils * mask[:, None, None, :]
        elif self.orientation == 'horizontal':
            # Mask k-space frequency encode lines
            ksp_coils = ksp_coils * mask[:, None, :, None]
        else:
            if len(mask.shape) == 3:
                ksp_coils = ksp_coils * mask[:, None, :, :]
            else:
                raise NotImplementedError('mask orientation not supported')


        # Return downsampled k-space
        return ksp_coils


# Generate a mask for MRI downsampling
def get_mask(acs_lines=26, total_lines=384, R=1):
    # Overall sampling budget
    num_sampled_lines = np.floor(total_lines / R)

    # Get locations of ACS lines
    # !!! Assumes k-space is even sized and centered, true for fastMRI
    center_line_idx = np.arange((total_lines - acs_lines) // 2,
                         (total_lines + acs_lines) // 2)

    # Find remaining candidates
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)

    # Sample remaining lines from outside the ACS at random
    random_line_idx = np.random.choice(outer_line_idx,
               size=int(num_sampled_lines - acs_lines), replace=False)

    # Create a mask and place ones at the right locations
    mask = np.zeros((total_lines))
    mask[center_line_idx] = 1.
    mask[random_line_idx] = 1.

    return mask

# Generate measurements directly from raw fastMRI data files
# Includes rescaling to 384 x 384, ACS-based scaling
# and masking
def get_measurements(raw_file, slice_idx, mask):
    # Load file and get slice
    with h5py.File(raw_file, 'r') as data:
        gt_ksp = np.asarray(data['kspace'][slice_idx])

    # Crop lines in k-space to 384
    gt_ksp = sp.resize(gt_ksp, (
        gt_ksp.shape[0], gt_ksp.shape[1], 384))

    # Reduce FoV by half in the readout direction
    gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
    gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], 384,
                                gt_ksp.shape[2]))
    gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space

    # ACS-based scaling
    # !!! Change this to pixel-based if desired
    acs          = sp.resize(gt_ksp, (26, 26))
    scale_factor = np.max(np.abs(acs))

    # Downsample and scale
    measured_ksp = gt_ksp * mask[None, None, :]
    measured_ksp = measured_ksp / scale_factor
    gt_ksp       = gt_ksp / scale_factor

    return measured_ksp, gt_ksp, scale_factor

def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))

def loss_geocross(latent):
    if latent.size() == (1, 512):
        return 0
    else:
        num_latents  = latent.size()[1]
        X = latent.view(-1, 1, num_latents, 512)
        Y = latent.view(-1, num_latents, 1, 512)
        A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
        B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
        D = 2 * torch.atan2(A, B)
        D = ((D.pow(2) * 512).mean((1, 2)) / 8.).mean()
        return D


def get_lr(t, initial_lr, rampdown=0.75, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)


# Source: https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
def pretty(d, indent=0):
    ''' Print dictionary '''
    for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


def get_transformation(image_size):
    return transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor()])


def load_or_learn_mapping(mapping_network=None, gaussian_fit_loc=None,
                          device='cuda', num_samples=100000, z_dim=512, relu_alpha=5):
    try:
        return MappingProxy(torch.load(gaussian_fit_loc, map_location='cpu'))
    except:
        mapping_network.to(device)
        latent = torch.randn((num_samples, z_dim), dtype=torch.float32, device=device)
        out = torch.nn.LeakyReLU(relu_alpha)(mapping_network(latent, None))
        gaussian_fit = {"mean": out.mean((0, 1)), "std": out.std((0, 1))}
        torch.save(gaussian_fit, gaussian_fit_loc)
        return MappingProxy(torch.load(gaussian_fit_loc, map_location='cpu'))


def create_folder(folder):
    if os.path.isdir(folder):
        while 1:
            response = input('Directory exists. Do you want to overwrite? (y/N) ')
            if response == 'y':
                shutil.rmtree(folder)
                break
            elif response == 'N':
                os._exit(0)
    os.makedirs(folder)


def save_images(samples, loc, normalize=False):
    torchvision.utils.save_image(
        samples,
        loc,
        nrow=int(samples.shape[0] ** 0.5),
        normalize=normalize,
        scale_each=True)


def load_dict(model, ckpt, device='cuda'):
    state_dict = torch.load(ckpt, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        print('Loading model failed... Trying to remove the module from the keys...')
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_state_dict[key[len('module.'):]] = value
        model.load_state_dict(new_state_dict)
    return model


def to_rgb(img, old_min=-1, old_max=1):
    return (255 * (img - old_min) / (old_max - old_min + 1e-5)).to(torch.uint8)

def get_time():
    return datetime.now().strftime('%H:%M:%S:%f')


def get_loss_fn(config, latent_z=None, mask=None, start_layer=0):
    device = config['device']
    functions = []
    labels = []
    if 'mse' in config and sum(config['mse']) > 0:
        labels.append('MSE')
        if mask is not None:
            functions.append(lambda x, y: F.mse_loss(x * mask, y))
        else:
            functions.append(lambda x, y: F.mse_loss(x, y))

    if 'geocross' in config and latent_z is not None and config['geocross'] > 0:
        labels.append('Geocross')
        functions.append(lambda x, y: loss_geocross(latent_z[2 * start_layer:]) * config['geocross'])
    return (lambda x, y: [fn(x, y) for fn in functions]), labels



def mp_setup(rank, world_size, port=12345):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def mp_cleanup():
    dist.destroy_process_group()


def update_pbar_desc(pbar, metrics, labels):
    pbar_string = ''
    for metric, label in zip(metrics, labels):
        pbar_string += f'{label}: {metric:.7f}; '
    pbar.set_description(pbar_string)


class MpLogger:
    def __init__(self, logger, rank):
        self.logger = logger
        self.rank = rank

    def info(self, message):
        if self.rank == 0:
            self.logger.info(message)

