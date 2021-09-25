from scipy.ndimage.interpolation import rotate,zoom
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import h5py
import sigpy as sp
from utils import get_mvue
import pickle as pkl
from xml.etree import ElementTree as ET
import sys




class MVU_Estimator_Brain(Dataset):
    def __init__(self, file_list, maps_dir, input_dir,
                 project_dir='./',
                 R=1,
                 image_size=(384,384),
                 acs_size=26,
                  pattern='random',
                 orientation='vertical'):
        # Attributes
        self.project_dir = project_dir
        self.file_list    = file_list
        self.maps_dir     = maps_dir
        self.input_dir      = input_dir
        self.image_size = image_size
        self.R            = R
        self.pattern      = pattern
        self.orientation  = orientation

        # Access meta-data of each scan to get number of slices
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            input_file = os.path.join(self.input_dir, os.path.basename(file))
            with h5py.File(os.path.join(self.project_dir, input_file), 'r') as data:
                self.num_slices[idx] = int(np.array(data['kspace']).shape[0])

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    # Phase encode random mask generator
    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)

        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)

        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
        else:
            raise NotImplementedError('Mask pattern not implemented')

        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        return mask

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1+wout, y1:y1+hout]

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        # Load maps for specific scan and slice
        maps_file = os.path.join(self.maps_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, maps_file), 'r') as data:
            # Get maps
            maps = np.asarray(data['s_maps'][slice_idx])

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.input_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, raw_file), 'r') as data:
            # Get maps
            gt_ksp = np.asarray(data['kspace'][slice_idx])
        # Crop extra lines and reduce FoV in phase-encode
        gt_ksp = sp.resize(gt_ksp, (
            gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0],
                                    gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space

        # Crop extra lines and reduce FoV in phase-encode
        maps = sp.fft(maps, axes=(-2, -1)) # These are now maps in k-space
        maps = sp.resize(maps, (
            maps.shape[0], maps.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0],
                                    maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,)) # Back to k-space
        maps = sp.ifft(maps, axes=(-2, -1)) # Finally convert back to image domain

        # find mvue image
        mvue = get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape))

        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.

        # Scale data
        mvue   = mvue   / scale_factor
        gt_ksp = gt_ksp / scale_factor

        # Compute ACS size based on R factor and sample size
        total_lines = gt_ksp.shape[-1]
        if 1 < self.R <= 6:
            # Keep 8% of center samples
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            acs_lines = np.floor(0.04 * total_lines).astype(int)

        # Get a mask
        mask = self._get_mask(acs_lines, total_lines,
                              self.R, self.pattern)
        # Mask k-space
        if self.orientation == 'vertical':
            gt_ksp *= mask[None, None, :]
        elif self.orientation == 'horizontal':
            gt_ksp *= mask[None, :, None]
        else:
            raise NotImplementedError

        ## name for mvue file
        mvue_file = os.path.join(self.input_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        # Output
        sample = {
                  'mvue': mvue,
                  'maps': maps,
                  'ground_truth': gt_ksp,
                  'mask': mask,
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx,
                  'mvue_file': mvue_file}
        return sample

class MVU_Estimator_Knees(Dataset):
    def __init__(self, file_list, maps_dir, input_dir,
                 project_dir='./',
                 R=1,
                 image_size=(320, 320),
                 acs_size=26,
                 pattern='random',
                 orientation='vertical'):
        # Attributes
        self.project_dir = project_dir
        self.file_list    = file_list
        self.acs_size     = acs_size
        self.maps_dir     = maps_dir
        self.input_dir      = input_dir
        self.R = R
        self.image_size = image_size
        self.pattern      = pattern
        self.orientation  = orientation

        # Access meta-data of each scan to get number of slices
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            raw_file = os.path.join(self.input_dir, os.path.basename(file))
            with h5py.File(os.path.join(self.project_dir, raw_file), 'r') as data:
                value = data['ismrmrd_header'][()]
                value = ET.fromstring(value)
                self.num_slices[idx] = int(value[4][2][3][1].text) + 1

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    # Phase encode random mask generator
    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)

        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)

        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
        else:
            raise NotImplementedError('Mask Pattern not implemented yet...')

        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        return mask

    def _knees_remove_zeros(self, kimage):
        # Compute sum-energy of lines
        # !!! This is because some lines are near-empty
        line_energy = np.sum(np.square(np.abs(kimage)),
                             axis=(0, 1))
        dead_lines  = np.where(line_energy < 1e-12)[0] # Sufficient for FP32
        # Always remove an even number of lines
        dead_lines_front = np.sum(dead_lines < 160)
        dead_lines_back  = np.sum(dead_lines > 160)
        if np.mod(dead_lines_front, 2):
            dead_lines = np.delete(dead_lines, 0)
        if np.mod(dead_lines_back, 2):
            dead_lines = np.delete(dead_lines, -1)
        # Remove dead lines completely
        k_image = np.delete(kimage, dead_lines, axis=-1)
        return k_image

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1+wout, y1:y1+hout]

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        # Load maps for specific scan and slice
        maps_file = os.path.join(self.maps_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, maps_file), 'r') as data:
            # Get maps
            maps = np.asarray(data['s_maps'][slice_idx])

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.input_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, raw_file), 'r') as data:
            # Get maps
            gt_ksp = np.asarray(data['kspace'][slice_idx])
        gt_ksp = self._knees_remove_zeros(gt_ksp)

        # Crop extra lines and reduce FoV by half in readout
        gt_ksp = sp.resize(gt_ksp, (
            gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0],
                                    gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space

        # Crop extra lines and reduce FoV by half in readout
        maps = sp.fft(maps, axes=(-2, -1)) # These are now maps in k-space
        maps = sp.resize(maps, (
            maps.shape[0], maps.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0],
                                    maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,)) # Back to k-space
        maps = sp.ifft(maps, axes=(-2, -1)) # Finally convert back to image domain

        # find mvue image
        mvue = get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape))

        # # Load MVUE slice from specific scan
        mvue_file = os.path.join(self.input_dir,
                                 os.path.basename(self.file_list[scan_idx]))

        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.

        # Scale data
        mvue   = mvue / scale_factor
        gt_ksp = gt_ksp / scale_factor

        # Compute ACS size based on R factor and sample size
        total_lines = gt_ksp.shape[-1]
        if 1 < self.R <= 6:
            # Keep 8% of center samples
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            acs_lines = np.floor(0.04 * total_lines).astype(int)

        # Get a mask
        mask = self._get_mask(acs_lines, total_lines,
                              self.R, self.pattern)

        # Mask k-space
        if self.orientation == 'vertical':
            gt_ksp *= mask[None, None, :]
        elif self.orientation == 'horizontal':
            gt_ksp *= mask[None, :, None]
        else:
            raise NotImplementedError

        # Output
        sample = {
                  'mvue': mvue,
                  'maps': maps,
                  'ground_truth': gt_ksp,
                  'mask': mask,
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx,
                  'mvue_file': mvue_file}
        return sample

class MVU_Estimator_Stanford_Knees(Dataset):
    def __init__(self, file_list, maps_dir, input_dir,
                 project_dir='./',
                 R=1,
                 image_size=(320,320),
                 acs_size=26,
                 pattern='random',
                 orientation='vertical'):
        # Attributes
        self.project_dir = project_dir
        self.acs_size     = acs_size
        self.maps_dir     = maps_dir
        self.input_dir      = input_dir
        self.R = R
        self.image_size = image_size
        self.pattern      = pattern
        self.orientation  = orientation
        self.file_list = sorted(file_list)
        if len(self.file_list) == 0:
            raise IOError('No image files found in the specified path')

        # Access meta-data of each scan to get number of slices
        # self.maps_file = os.path.join(maps_dir, 'Stanford-Knee-Axial-Selected.h5')
        # self.raw_file = os.path.join(input_dir, 'Stanford-Knee-Axial-Selected.h5')
        # with h5py.File(os.path.join(self.project_dir, self.raw_file), 'r') as data:
        #     self.num_slices = np.array(data['kspace']).shape[0]
    @property
    def num_slices(self):
        num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            with h5py.File(os.path.join(self.project_dir, file), 'r') as data:
                num_slices[idx] = np.array(data['kspace']).shape[0]
        return num_slices

    @property
    def slice_mapper(self):
        return np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        # Load specific slice from specific scan
        with h5py.File(os.path.join(self.project_dir, self.file_list[scan_idx]), 'r') as data:
            # Get maps, kspace, masks
            gt_ksp = np.asarray(data['kspace'])[slice_idx]
            maps = np.asarray(data['s_maps'])[slice_idx]
            mask = np.asarray(data['masks'])[slice_idx]


        # find mvue image
        mvue = get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape))

        # # Load MVUE slice from specific scan
        mvue_file = os.path.join(self.input_dir,
                                 os.path.basename(self.file_list[scan_idx]))

        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.

        # Scale data
        mvue   = mvue / scale_factor
        gt_ksp = gt_ksp / scale_factor

        # apply mask
        gt_ksp *= mask[None, :, :]

        # Output
        sample = {
                  'mvue': mvue,
                  'maps': maps,
                  'ground_truth': gt_ksp,
                  'mask': mask,
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx,
                  'mvue_file': mvue_file}
        return sample

# class MVU_Estimator_Stanford_Knees(Dataset):
#     def __init__(self, maps_dir, input_dir,
#                  project_dir='./',
#                  R=1,
#                  image_size=(320,320),
#                  acs_size=26,
#                  pattern='random',
#                  orientation='vertical'):
#         # Attributes
#         self.project_dir = project_dir
#         self.acs_size     = acs_size
#         self.maps_dir     = maps_dir
#         self.input_dir      = input_dir
#         self.R = R
#         self.image_size = image_size
#         self.pattern      = pattern
#         self.orientation  = orientation

#         # Access meta-data of each scan to get number of slices
#         self.num_slices = np.ones(18, dtype=int)
#         self.maps_file = os.path.join(maps_dir, 'Stanford_maps_rotated.h5')
#         self.raw_file = os.path.join(input_dir, 'Stanford_knees.pkl')

#     def __len__(self):
#         return int(np.sum(self.num_slices)) # Total number of slices from all scans

#     # Phase encode random mask generator
#     def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
#         # Overall sampling budget
#         num_sampled_lines = np.floor(total_lines / R)

#         # Get locations of ACS lines
#         # !!! Assumes k-space is even sized and centered, true for fastMRI
#         center_line_idx = np.arange((total_lines - acs_lines) // 2,
#                              (total_lines + acs_lines) // 2)

#         # Find remaining candidates
#         outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
#         if pattern == 'random':
#             # Sample remaining lines from outside the ACS at random
#             random_line_idx = np.random.choice(outer_line_idx,
#                        size=int(num_sampled_lines - acs_lines), replace=False)
#         elif pattern == 'equispaced':
#             # Sample equispaced lines
#             # !!! Only supports integer for now
#             random_line_idx = outer_line_idx[::int(R)]
#         else:
#             raise NotImplementedError('Mask Pattern not implemented yet...')

#         # Create a mask and place ones at the right locations
#         mask = np.zeros((total_lines))
#         mask[center_line_idx] = 1.
#         mask[random_line_idx] = 1.

#         return mask

#     # Cropping utility - works with numpy / tensors
#     def _crop(self, x, wout, hout):
#         w, h = x.shape[-2:]
#         x1 = int(np.ceil((w - wout) / 2.))
#         y1 = int(np.ceil((h - hout) / 2.))

#         return x[..., x1:x1+wout, y1:y1+hout]

#     def _rotatecomplex(self, a,angle,reshape=True):
#         r = rotate(a.real,angle,reshape=reshape,mode='wrap')
#         i = rotate(a.imag,angle,reshape=reshape,mode='wrap')
#         return r+1j*i

#     def __getitem__(self, idx):
#         # Convert to numerical
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         # Load maps for specific scan and slice
#         with h5py.File(os.path.join(self.project_dir, self.maps_file), 'r') as data:
#             # Get maps
#             maps = np.asarray(data[f'ge{idx+1}.h5'])

#         # Load raw data for specific scan and slice
#         with open(os.path.join(self.project_dir, self.raw_file), 'rb') as f:
#             # Get maps
#             data = pkl.load(f)
#             slice_ksp = np.asarray(data[f'ge{idx+1}.h5']['kspace'])

#         # rotate kspace by 90 degrees
#         gt_ksp = slice_ksp.copy()
#         for c,coil in enumerate(slice_ksp):
#             gt_ksp[c,:,:] = self._rotatecomplex(coil,90) # the kspace is rotated, so werotate it back to the original format

#         # find mvue image
#         mvue = get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape))

#         # # Load MVUE slice from specific scan
#         mvue_file = os.path.join(self.input_dir,f'ge{idx+1}.h5')

#         # !!! Removed ACS-based scaling if handled on the outside
#         scale_factor = 1.

#         # Scale data
#         mvue   = mvue / scale_factor
#         gt_ksp = gt_ksp / scale_factor

#         # Compute ACS size based on R factor and sample size
#         total_lines = gt_ksp.shape[-1]
#         if 1 < self.R <= 6:
#             # Keep 8% of center samples
#             acs_lines = np.floor(0.08 * total_lines).astype(int)
#         else:
#             # Keep 4% of center samples
#             acs_lines = np.floor(0.04 * total_lines).astype(int)

#         # Get a mask
#         mask = self._get_mask(acs_lines, total_lines,
#                               self.R, self.pattern)

#         # Mask k-space
#         if self.orientation == 'vertical':
#             gt_ksp *= mask[None, None, :]
#         elif self.orientation == 'horizontal':
#             gt_ksp *= mask[None, :, None]
#         else:
#             raise NotImplementedError

#         # Output
#         sample = {
#                   'mvue': mvue,
#                   'maps': maps,
#                   'ground_truth': gt_ksp,
#                   'mask': mask,
#                   'scale_factor': scale_factor,
#                   # Just for feedback
#                   'scan_idx': 1,
#                   'slice_idx': idx+1,
#                   'mvue_file': mvue_file}
#         return sample

class MVU_Estimator_Abdomen(Dataset):
    def __init__(self, maps_dir, input_dir,
                 project_dir='./',
                 R=1,
                 image_size=(158,320),
                 acs_size=26,
                 pattern='random',
                 rotate=True,
                 orientation='vertical'):
        # Attributes
        self.project_dir = project_dir
        self.acs_size     = acs_size
        self.maps_dir     = maps_dir
        self.input_dir      = input_dir
        self.R = R
        self.image_size = image_size
        self.pattern      = pattern
        self.orientation  = orientation
        self.rotate = rotate

        # Access meta-data of each scan to get number of slices
        self.maps_file = os.path.join(self.project_dir, maps_dir, 'data2.h5')
        self.raw_file = os.path.join(self.project_dir, input_dir, 'data2.h5')
        with h5py.File(self.raw_file, 'r') as f:
            self.num_slices = np.array(f['ksp']).shape[0]

    def __len__(self):
        return self.num_slices # Total number of slices from all scans

    # Phase encode random mask generator
    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)

        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)

        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
        else:
            raise NotImplementedError('Mask Pattern not implemented yet...')

        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        return mask

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1+wout, y1:y1+hout]

    def _rotatecomplex(self, a,angle,reshape=True):
        r = rotate(a.real,angle,reshape=reshape,mode='wrap')
        i = rotate(a.imag,angle,reshape=reshape,mode='wrap')
        return r+1j*i

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load maps for specific scan and slice
        with h5py.File(os.path.join(self.project_dir, self.maps_file), 'r') as data:
            # Get maps
            maps = np.asarray(data['maps'])[idx]

        # Load raw data for specific scan and slice
        with h5py.File(os.path.join(self.project_dir, self.raw_file), 'r') as data:
            # Get maps
            slice_ksp = np.asarray(data['ksp'])[idx]

        # # rotate kspace by 90 degrees
        if self.rotate:
            gt_ksp = slice_ksp.copy()
            for c,coil in enumerate(slice_ksp):
                gt_ksp[c,:,:] = self._rotatecomplex(coil,90) # the kspace is rotated, so werotate it back to the original format
        else:
            gt_ksp = slice_ksp.copy()

        # pad readout in image domain
        x = sp.ifft(gt_ksp, axes=(-1,))
        x = sp.resize(x, (x.shape[0], x.shape[1], self.image_size[1]))

        # pad phase-encode in kspace domain
        gt_ksp = sp.fft(x, axes=(-1,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0], self.image_size[1]))

        # Crop extra lines and reduce FoV by half in readout
        maps = sp.fft(maps, axes=(-1, -2)) # These are now maps in k-space
        maps = sp.ifft(maps, axes=(-1,))
        maps = sp.resize(maps, (
            maps.shape[0], maps.shape[1], self.image_size[1]))
        maps = sp.fft(maps, axes=(-1,))

        # pad phase-encode in kspace domain
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0],
                                    self.image_size[1]))
        maps = sp.ifft(maps, axes=(-1, -2)) # Finally convert back to image domain


        # find mvue image
        mvue = get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape))

        # # Load MVUE slice from specific scan
        mvue_file = os.path.join(self.input_dir,str(idx))

        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.

        # Scale data
        mvue   = mvue / scale_factor
        gt_ksp = gt_ksp / scale_factor

        # Compute ACS size based on R factor and sample size
        if self.orientation == 'horizontal':
            total_lines = gt_ksp.shape[-2]
        elif self.orientation == 'vertical':
            total_lines = gt_ksp.shape[-1]
        else:
            raise NotImplementedError

        if 1 < self.R <= 6:
            # Keep 8% of center samples
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            acs_lines = np.floor(0.04 * total_lines).astype(int)

        # Get a mask
        mask = self._get_mask(acs_lines, total_lines,
                              self.R, self.pattern)

        # Mask k-space
        if self.orientation == 'vertical':
            gt_ksp *= mask[None, None, :]
        elif self.orientation == 'horizontal':
            gt_ksp *= mask[None, :, None]
        else:
            raise NotImplementedError

        # Output
        sample = {
                  'mvue': mvue,
                  'maps': maps,
                  'ground_truth': gt_ksp,
                  'mask': mask,
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': 1,
                  'slice_idx': idx+1,
                  'mvue_file': mvue_file}
        return sample
