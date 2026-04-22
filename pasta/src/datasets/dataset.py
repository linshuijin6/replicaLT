import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import transforms as T, utils
from tqdm import tqdm

import pandas as pd

from src.utils.data_utils import get_neighboring_slices, multislice_data_minimal_process, \
                                                    get_3d_image_transform, crop_minimally_keep_ratio, rescale_intensity, rescale_intensity_3D, \
                                                    resample_and_reshape, process_tabular_data, data_minimal_process, CropOrPad_3D

import nibabel as nib
import numpy as np
import pandas as pd
import os

import logging

LOG = logging.getLogger(__name__)

# constants

DIAGNOSIS_MAP = {"CN": 0, "Dementia": 1, "AD": 1, "MCI": 2}
DIAGNOSIS_MAP_binary = {"CN": 0, "AD": 1, "Dementia": 1}

# dataset

class SlicedScanMRI2PETDataset(Dataset):
    def __init__(
        self,
        resolution = None,
        mri_root_path = '/path/to/mri',
        pet_root_path = '/path/to/pet',
        data_path = None,
        output_dim = 1,
        direction = 'coronal',
        standardized_tabular = True,
        classes=None, # 'binary' or 'multi' (with MCI)
        random_crop=False,
        random_flip=False,
        random_affine=False,
        resample_mri=False,
        dx_labels = ['CN', 'MCI', 'Dementia'],
    ):
        super().__init__()
        self.resolution = resolution
        self.mri_root_path = mri_root_path
        self.pet_root_path = pet_root_path
        self.data_path = data_path
        self.output_dim = output_dim
        self.direction = direction
        self.standardized_tabular = standardized_tabular
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_affine = random_affine
        self.with_label = classes
        self.resample_mri = resample_mri
        self.dx_labels = dx_labels

        self._load()

    def _load(self):
        mri_data = []
        pet_data = []
        diagnosis = []
        tabular_data = []
        mri_uid = []

        PET_shape = (113, 137, 113)
        MRI_shape = (113, 137, 113)

        if self.data_path is not None and 'h5' in self.data_path:       
            print('loaded from h5 file')
            with h5py.File(self.data_path, mode='r') as file:
                for name, group in tqdm(file.items(), total=len(file)):
                    if name == "stats":
                        self.tabular_mean = group["tabular/mean"][:]
                        self.tabular_std = group["tabular/stddev"][:]
                    else:
                        if group.attrs['DX'] not in self.dx_labels:
                            continue
                        if self.resample_mri:
                            _raw_mri_data = group['MRI/T1/data'][:]
                            _resampled_mri_data = resample_and_reshape(_raw_mri_data, (1.5, 1.5, 1.5), PET_shape)
                            input_mri_data = _resampled_mri_data
                            MRI_shape = PET_shape
                            assert input_mri_data.shape == PET_shape
                        else:
                            input_mri_data = group['MRI/T1/data'][:]
                        
                        _pet_data = group['PET/FDG/data'][:]
                        _mri_data = input_mri_data
                        _tabular_data = group['tabular'][:]
                        _diagnosis = group.attrs['DX']

                        _pet_data = np.nan_to_num(_pet_data, copy=False)
                        mri_data.append(_mri_data)
                        pet_data.append(_pet_data)
                        tabular_data.append(_tabular_data)
                        diagnosis.append(_diagnosis)
                        mri_uid.append(name)
        else:
            print('loaded from: ', self.mri_root_path, self.pet_root_path)

            mri_id = os.listdir(self.mri_root_path)
            mri_input = [os.path.join(self.mri_root_path, i, 'mri.nii.gz') for i in mri_id]
            pet_input = [os.path.join(self.pet_root_path, i[:-8], f'pet_fdg.nii.gz') for i in mri_id]
            mri_data = [nib.load(i).get_fdata() for i in mri_input]
            pet_data = [nib.load(i).get_fdata() for i in pet_input]

            csv_info = pd.read_csv('data_info.csv')
            diagnosis = [csv_info.loc[csv_info["IMAGEUID"] == i]['DX'].values[0] for i in mri_id]
            tabular_data = [csv_info.loc[csv_info["IMAGEUID"] == i]['TAB'].values[0] for i in mri_id]
            mri_uid = mri_id



        self.len_data = len(pet_data)
        self._image_data_mri = mri_data
        self._image_data_pet = pet_data
        self._tabular_data = tabular_data
        self._mri_uid = mri_uid
        
      
        LOG.info("DATASET: %s", self.data_path if self.data_path is not None else self.mri_root_path)
        LOG.info("SAMPLES: %d", self.len_data)

        # if self.with_label is not None:
        labels, counts = np.unique(diagnosis, return_counts=True)
        LOG.info("Classes: %s", pd.Series(counts, index=labels))     

        if self.with_label == 'binary':
            self._diagnosis = [DIAGNOSIS_MAP_binary[d] for d in diagnosis]
        elif self.with_label == 'multi':
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]
        else:
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]

    
    def __len__(self):
        return self.len_data


    def __getitem__(self, idx):

        MRI_shape = self.resolution
        PET_shape = self.resolution

        mri_scan = self._image_data_mri[idx]
        pet_scan = self._image_data_pet[idx]
        tabular_data = self._tabular_data[idx]
        mri_uid = self._mri_uid[idx]

        mri_scan = rescale_intensity_3D(mri_scan)
        pet_scan = rescale_intensity_3D(pet_scan)
        mri_scan = CropOrPad_3D(mri_scan, MRI_shape)
        pet_scan = CropOrPad_3D(pet_scan, PET_shape)

        if self.standardized_tabular:
            tabular_data = (tabular_data - self.tabular_mean) / self.tabular_std
            tabular_data = process_tabular_data(tabular_data)

        mri_scan_list = []
        pet_scan_list = []

        data_transform = T.Compose([
                T.ToTensor(),
                # T.CenterCrop((self.resolution[0], self.resolution[1])) if self.resolution is not None else nn.Identity(),
                # T.Resize((self.resolution[0], self.resolution[1]), antialias=True) if self.resolution is not None else nn.Identity(),
                T.RandomVerticalFlip() if self.random_flip else nn.Identity(),
                T.RandomAffine(180, translate=(0.3, 0.3)) if self.random_affine else nn.Identity(),         
            ])

        if self.direction == 'coronal':
            for i in range(MRI_shape[1]):
                if self.output_dim > 1:
                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, mri_scan)
                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, pet_scan)
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = multislice_data_minimal_process(self.output_dim, self.resolution, _mri_data, data_transform)
                    _pet_data = multislice_data_minimal_process(self.output_dim, self.resolution, _pet_data, data_transform)

                else:
                    _pet_data = pet_scan[:, i, :]
                    _mri_data = mri_scan[:, i, :]
                    _pet_data = np.nan_to_num(_pet_data, copy=False)            
                    _mri_data = data_minimal_process(self.resolution, _mri_data, data_transform)
                    _pet_data = data_minimal_process(self.resolution, _pet_data, data_transform)

                mri_scan_list.append(_mri_data)
                pet_scan_list.append(_pet_data)
        
        elif self.direction == 'sagittal':
            for i in range(MRI_shape[0]):
                if self.output_dim > 1:
                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, mri_scan)
                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, pet_scan)
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = multislice_data_minimal_process(self.output_dim, self.resolution, _mri_data, data_transform)
                    _pet_data = multislice_data_minimal_process(self.output_dim, self.resolution, _pet_data, data_transform)
                else:
                    _pet_data = pet_scan[i, :, :]
                    _mri_data = mri_scan[i, :, :]
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = data_minimal_process(self.resolution, _mri_data, data_transform)
                    _pet_data = data_minimal_process(self.resolution, _pet_data, data_transform)

                mri_scan_list.append(_mri_data)
                pet_scan_list.append(_pet_data)

        
        elif self.direction == 'axial':
            for i in range(MRI_shape[2]):
                if self.output_dim > 1:
                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, mri_scan)
                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, pet_scan)
                    
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = multislice_data_minimal_process(self.output_dim, self.resolution, _mri_data, data_transform)
                    _pet_data = multislice_data_minimal_process(self.output_dim, self.resolution, _pet_data, data_transform)
                else:
                    _pet_data = pet_scan[:, :, i]
                    _mri_data = mri_scan[:, :, i]

                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = data_minimal_process(self.resolution, _mri_data, data_transform)
                    _pet_data = data_minimal_process(self.resolution, _pet_data, data_transform)

                mri_scan_list.append(_mri_data)
                pet_scan_list.append(_pet_data)

        label = self._diagnosis[idx]
       
        return mri_scan_list, pet_scan_list, label, tabular_data, mri_uid


class MRI2PET_2_5D_Dataset(Dataset):
    def __init__(
        self,
        resolution = None,
        mri_root_path = '/path/to/mri',
        pet_root_path = '/path/to/pet',
        data_path = None,
        output_dim = 3,
        direction = 'axial',
        num_slices = 'all',
        standardized_tabular = True,
        classes=None, # 'binary' or 'multi' (with MCI)
        random_crop=False,
        random_flip=False,
        random_affine=False,
        resample_mri=False,
        ROI_mask = None,
        dx_labels = ['CN', 'Dementia', 'MCI'],
    ):
        super().__init__()
        self.resolution = resolution
        self.mri_root_path = mri_root_path
        self.pet_root_path = pet_root_path
        self.data_path = data_path
        self.output_dim = output_dim
        self.direction = direction
        self.num_slices = num_slices
        self.standardized_tabular = standardized_tabular
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_affine = random_affine
        self.with_label = classes
        self.resample_mri = resample_mri

        self.ROI_mask = ROI_mask
        self.dx_labels = dx_labels

        self._load()

    def _load(self):
        """Lazy-load version: only build an index of (subject_key, slice_position).
        Actual volume data is read on-demand in __getitem__().
        """
        diagnosis = []
        tabular_data = []
        slice_index = []
        mri_uid = []

        PET_shape = (113, 137, 113)
        MRI_shape = (113, 137, 113)

        # Determine max_slice_index and slice positions based on direction
        dir_to_axis = {'coronal': 1, 'sagittal': 0, 'axial': 2}
        axis = dir_to_axis[self.direction]
        max_slice_index = PET_shape[axis] - 1
        center = PET_shape[axis] // 2 + 1

        if 'h5' not in self.data_path:
            raise NotImplementedError

        print('lazy-loading index from h5 file')

        # We need to pre-scan for 'all' mode to filter empty slices.
        # For fixed num_slices (int), slices around center are unlikely empty, skip pre-filter.
        need_prescan = (self.num_slices == 'all')

        # --- Index mapping: list of (h5_key, absolute_slice_pos) ---
        self._index_map = []

        with h5py.File(self.data_path, mode='r') as file:
            for name, group in tqdm(file.items(), total=len(file), desc='indexing'):

                if name == "stats":
                    self.tabular_mean = group["tabular/mean"][:]
                    self.tabular_std = group["tabular/stddev"][:]
                    continue

                if group.attrs['DX'] not in self.dx_labels:
                    continue

                _tabular_data = group['tabular'][:]
                _diagnosis = group.attrs['DX']

                if self.num_slices == 1:
                    positions = [center]
                elif self.num_slices == 'all':
                    positions = list(range(PET_shape[axis]))
                else:
                    positions = [center + i for i in range(-self.num_slices // 2, self.num_slices // 2)]

                if need_prescan:
                    # Read volumes once to filter empty slices, then discard
                    if self.resample_mri:
                        _raw_mri = group['MRI/T1/data'][:]
                        input_mri = resample_and_reshape(_raw_mri, (1.5, 1.5, 1.5), PET_shape)
                        input_pet = group['PET/FDG/data'][:]
                    else:
                        input_mri = group['MRI/T1/data'][:]
                        input_pet = group['PET/FDG/data'][:]
                    input_mri = rescale_intensity_3D(input_mri)
                    input_pet = rescale_intensity_3D(input_pet)

                    for pos in positions:
                        _mri_sl = get_neighboring_slices(self.output_dim, self.direction, pos, input_mri)
                        _pet_sl = get_neighboring_slices(self.output_dim, self.direction, pos, input_pet)
                        _pet_sl = np.nan_to_num(_pet_sl, copy=False)
                        if not np.any(_pet_sl) or not np.any(_mri_sl):
                            continue
                        self._index_map.append((name, pos))
                        tabular_data.append(_tabular_data)
                        diagnosis.append(_diagnosis)
                        mri_uid.append(name)
                        slice_index.append(pos)
                    del input_mri, input_pet
                else:
                    for pos in positions:
                        self._index_map.append((name, pos))
                        tabular_data.append(_tabular_data)
                        diagnosis.append(_diagnosis)
                        mri_uid.append(name)
                        slice_index.append(pos)

        self.len_data = len(self._index_map)
        self._tabular_data = tabular_data
        self._slice_index = [float(i / max_slice_index) for i in slice_index]
        self._max_slice_index = max_slice_index
        self._mri_uid = mri_uid
        self._PET_shape = PET_shape

        # Per-worker h5 file handle (lazy-opened in __getitem__)
        self._h5_file = None

        LOG.info("DATASET: %s", self.data_path)
        LOG.info("SAMPLES: %d", self.len_data)

        labels, counts = np.unique(diagnosis, return_counts=True)
        LOG.info("Classes: %s", pd.Series(counts, index=labels))

        if self.with_label == 'binary':
            self._diagnosis = [DIAGNOSIS_MAP_binary[d] for d in diagnosis]
        elif self.with_label == 'multi':
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]
        else:
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]

        if self.ROI_mask is not None:
            self._ROI_mask = nib.load(self.ROI_mask).get_fdata()
            print('Loaded ROI mask shape: ', self._ROI_mask.shape)
            assert PET_shape == self._ROI_mask.shape, ('ROI mask shape is not Input data shape', self._ROI_mask.shape)
        else:
            self._ROI_mask = None

    def _get_h5(self):
        """Return an h5py File handle, opening lazily (safe for multi-worker DataLoader)."""
        if self._h5_file is None:
            try:
                self._h5_file = h5py.File(self.data_path, mode='r', swmr=True)
            except Exception:
                self._h5_file = h5py.File(self.data_path, mode='r')
        return self._h5_file

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        h5_key, slice_pos = self._index_map[idx]
        tabular_data = self._tabular_data[idx]
        slice_index = self._slice_index[idx]
        assert 0 <= slice_index <= 1, 'slice index should be normalized to [0, 1]'

        # --- Read 3D volume on demand ---
        f = self._get_h5()
        group = f[h5_key]
        if self.resample_mri:
            input_mri = resample_and_reshape(group['MRI/T1/data'][:], (1.5, 1.5, 1.5), self._PET_shape)
            input_pet = group['PET/FDG/data'][:]
        else:
            input_mri = group['MRI/T1/data'][:]
            input_pet = group['PET/FDG/data'][:]

        input_mri = rescale_intensity_3D(input_mri)
        input_pet = rescale_intensity_3D(input_pet)

        mri_scan = get_neighboring_slices(self.output_dim, self.direction, slice_pos, input_mri)
        pet_scan = get_neighboring_slices(self.output_dim, self.direction, slice_pos, input_pet)
        pet_scan = np.nan_to_num(pet_scan, copy=False)

        # Free 3D volumes immediately
        del input_mri, input_pet

        # --- ROI mask ---
        if self._ROI_mask is not None:
            roi_mask = get_neighboring_slices(self.output_dim, self.direction, int(slice_index * self._max_slice_index), self._ROI_mask)
            assert roi_mask.shape == mri_scan.shape, ('roi mask shape is not Input scan shape', roi_mask.shape, mri_scan.shape)
            loss_weight_mask = roi_mask.copy()
            loss_weight_mask[roi_mask == 0] = 1
            loss_weight_mask[roi_mask == 1] = 10
        else:
            loss_weight_mask = np.ones(mri_scan.shape)

        # --- Tabular ---
        if self.standardized_tabular:
            tabular_data = (tabular_data - self.tabular_mean) / self.tabular_std
        tabular_data = process_tabular_data(tabular_data)

        # --- Per-slice transform ---
        data_transform = T.Compose([
            T.ToTensor(),
            T.CenterCrop((self.resolution[0], self.resolution[1])) if self.resolution is not None else nn.Identity(),
            T.RandomVerticalFlip() if self.random_flip else nn.Identity(),
            T.RandomAffine(180, translate=(0.3, 0.3)) if self.random_affine else nn.Identity(),
        ])

        processed_mri = np.zeros((self.output_dim, self.resolution[0], self.resolution[1]), dtype=np.float32)
        processed_pet = np.zeros((self.output_dim, self.resolution[0], self.resolution[1]), dtype=np.float32)
        processed_loss_weight_mask = np.zeros((self.output_dim, self.resolution[0], self.resolution[1]), dtype=np.float32)

        for i in range(mri_scan.shape[0]):
            processed_mri[i] = data_minimal_process(self.resolution, mri_scan[i], data_transform)
            processed_pet[i] = data_minimal_process(self.resolution, pet_scan[i], data_transform)
            processed_loss_weight_mask[i] = data_minimal_process(self.resolution, loss_weight_mask[i], data_transform)

        mri_scan = processed_mri
        pet_scan = processed_pet
        loss_weight_mask = processed_loss_weight_mask

        label = self._diagnosis[idx]
        mri_uid = self._mri_uid[idx]

        assert mri_scan.shape == (self.output_dim, self.resolution[0], self.resolution[1]), 'mri scan shape is not correct'
        assert pet_scan.shape == (self.output_dim, self.resolution[0], self.resolution[1]), 'pet scan shape is not correct'

        return mri_scan, pet_scan, label, tabular_data, slice_index, loss_weight_mask, mri_uid