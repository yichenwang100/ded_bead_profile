import os, shutil, sys, re
import time

from tqdm import tqdm

import torch, torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
from PIL import Image
from scipy.ndimage import center_of_mass
from skimage.measure import shannon_entropy

from torchvision import transforms
import matplotlib.pyplot as plt

from util import *

'''***********************************************************************'''
'''Project related settings'''
'''***********************************************************************'''

dataset_name_list = ['Low_const_const_1', 'Low_const_const_2', 'Low_const_tooth_1', 'Low_const_tooth_2', 'Low_const_sin_1', 'Low_const_sin_2', 'Low_const_square_1', 'Low_const_square_2', 'Low_tooth_const_1', 'Low_tooth_const_2', 'Low_tooth_tooth_1', 'Low_tooth_tooth_2', 'Low_tooth_sin_1', 'Low_tooth_sin_2', 'Low_tooth_square_1', 'Low_tooth_square_2', 'Low_tooth_noise_1', 'Low_tooth_noise_2', 'Low_sin_const_1', 'Low_sin_const_2', 'Low_sin_tooth_1', 'Low_sin_tooth_2', 'Low_sin_sin_1', 'Low_sin_sin_2', 'Low_sin_square_1', 'Low_sin_square_2', 'Low_square_const_1', 'Low_square_const_2', 'Low_square_tooth_1', 'Low_square_tooth_2', 'Low_square_sin_1', 'Low_square_sin_2', 'Low_square_square_1', 'Low_square_square_2', 'Low_noise_tooth_1', 'Low_noise_tooth_2', 'Low_noise_noise_1', 'Low_noise_noise_2', 'High_const_tooth_1', 'High_const_tooth_2', 'High_const_sin_1', 'High_const_sin_2', 'High_const_square_1', 'High_const_square_2', 'High_tooth_const_1', 'High_tooth_const_2', 'High_tooth_tooth_1', 'High_tooth_tooth_2', 'High_tooth_sin_1', 'High_tooth_sin_2', 'High_tooth_square_1', 'High_tooth_square_2', 'High_sin_const_1', 'High_sin_const_2', 'High_sin_tooth_1', 'High_sin_tooth_2', 'High_sin_sin_1', 'High_sin_sin_2', 'High_sin_square_1', 'High_sin_square_2', 'High_square_const_1', 'High_square_const_2', 'High_square_tooth_1', 'High_square_tooth_2', 'High_square_sin_1', 'High_square_sin_2', 'High_square_square_1', 'High_square_square_2', 'CADPrint_Drawing3', 'CADPrint_Drawing4', 'CADPrint_Drawing6']
dataset_exclude_for_deploy = ['Low_noise_noise_1', 'Low_noise_noise_2',
                              'Low_const_const_1', 'Low_const_const_2',
                              'High_sin_tooth_1', 'High_sin_tooth_2']
map_dataset_name_to_raw_id = bidict({'Low_const_const_1': 1, 'Low_const_const_2': 2, 'Low_const_tooth_1': 3, 'Low_const_tooth_2': 4, 'Low_const_sin_1': 5, 'Low_const_sin_2': 6, 'Low_const_square_1': 7, 'Low_const_square_2': 8, 'Low_tooth_const_1': 11, 'Low_tooth_const_2': 12, 'Low_tooth_tooth_1': 13, 'Low_tooth_tooth_2': 14, 'Low_tooth_sin_1': 15, 'Low_tooth_sin_2': 16, 'Low_tooth_square_1': 17, 'Low_tooth_square_2': 18, 'Low_tooth_noise_1': 19, 'Low_tooth_noise_2': 20, 'Low_sin_const_1': 21, 'Low_sin_const_2': 22, 'Low_sin_tooth_1': 23, 'Low_sin_tooth_2': 24, 'Low_sin_sin_1': 25, 'Low_sin_sin_2': 26, 'Low_sin_square_1': 27, 'Low_sin_square_2': 28, 'Low_square_const_1': 31, 'Low_square_const_2': 32, 'Low_square_tooth_1': 33, 'Low_square_tooth_2': 34, 'Low_square_sin_1': 35, 'Low_square_sin_2': 36, 'Low_square_square_1': 37, 'Low_square_square_2': 38, 'Low_noise_tooth_1': 43, 'Low_noise_tooth_2': 44, 'Low_noise_noise_1': 49, 'Low_noise_noise_2': 50, 'High_const_tooth_1': 53, 'High_const_tooth_2': 54, 'High_const_sin_1': 55, 'High_const_sin_2': 56, 'High_const_square_1': 57, 'High_const_square_2': 58, 'High_tooth_const_1': 61, 'High_tooth_const_2': 62, 'High_tooth_tooth_1': 63, 'High_tooth_tooth_2': 64, 'High_tooth_sin_1': 65, 'High_tooth_sin_2': 66, 'High_tooth_square_1': 67, 'High_tooth_square_2': 68, 'High_sin_const_1': 71, 'High_sin_const_2': 72, 'High_sin_tooth_1': 73, 'High_sin_tooth_2': 74, 'High_sin_sin_1': 75, 'High_sin_sin_2': 76, 'High_sin_square_1': 77, 'High_sin_square_2': 78, 'High_square_const_1': 81, 'High_square_const_2': 82, 'High_square_tooth_1': 83, 'High_square_tooth_2': 84, 'High_square_sin_1': 85, 'High_square_sin_2': 86, 'High_square_square_1': 87, 'High_square_square_2': 88, 'CADPrint_Drawing3': 103, 'CADPrint_Drawing4': 104, 'CADPrint_Drawing6': 106})
map_dataset_name_to_pub_id = bidict({'Low_const_const_1': 1, 'Low_const_const_2': 2, 'Low_const_tooth_1': 3, 'Low_const_tooth_2': 4, 'Low_const_sin_1': 5, 'Low_const_sin_2': 6, 'Low_const_square_1': 7, 'Low_const_square_2': 8, 'Low_tooth_const_1': 9, 'Low_tooth_const_2': 10, 'Low_tooth_tooth_1': 11, 'Low_tooth_tooth_2': 12, 'Low_tooth_sin_1': 13, 'Low_tooth_sin_2': 14, 'Low_tooth_square_1': 15, 'Low_tooth_square_2': 16, 'Low_tooth_noise_1': 17, 'Low_tooth_noise_2': 18, 'Low_sin_const_1': 19, 'Low_sin_const_2': 20, 'Low_sin_tooth_1': 21, 'Low_sin_tooth_2': 22, 'Low_sin_sin_1': 23, 'Low_sin_sin_2': 24, 'Low_sin_square_1': 25, 'Low_sin_square_2': 26, 'Low_square_const_1': 27, 'Low_square_const_2': 28, 'Low_square_tooth_1': 29, 'Low_square_tooth_2': 30, 'Low_square_sin_1': 31, 'Low_square_sin_2': 32, 'Low_square_square_1': 33, 'Low_square_square_2': 34, 'Low_noise_tooth_1': 35, 'Low_noise_tooth_2': 36, 'Low_noise_noise_1': 37, 'Low_noise_noise_2': 38, 'High_const_tooth_1': 39, 'High_const_tooth_2': 40, 'High_const_sin_1': 41, 'High_const_sin_2': 42, 'High_const_square_1': 43, 'High_const_square_2': 44, 'High_tooth_const_1': 45, 'High_tooth_const_2': 46, 'High_tooth_tooth_1': 47, 'High_tooth_tooth_2': 48, 'High_tooth_sin_1': 49, 'High_tooth_sin_2': 50, 'High_tooth_square_1': 51, 'High_tooth_square_2': 52, 'High_sin_const_1': 53, 'High_sin_const_2': 54, 'High_sin_tooth_1': 55, 'High_sin_tooth_2': 56, 'High_sin_sin_1': 57, 'High_sin_sin_2': 58, 'High_sin_square_1': 59, 'High_sin_square_2': 60, 'High_square_const_1': 61, 'High_square_const_2': 62, 'High_square_tooth_1': 63, 'High_square_tooth_2': 64, 'High_square_sin_1': 65, 'High_square_sin_2': 66, 'High_square_square_1': 67, 'High_square_square_2': 68, 'CADPrint_Drawing3': 69, 'CADPrint_Drawing4': 70, 'CADPrint_Drawing6': 71})

param_str_list = [
    "EXP_ID",  #            0
    "POINT_ID",  #          1
    "FREQUENCY",  #         2
    "POWER_PATTERN",  #     3
    "FEEDRATE_PATTERN",  #  4
    "LINEIDX",  #           5
    "RTCP",  #              6   # 0 # used
    "CLOCKWISE",  #         7   # 1 # used
    "CURVATURE",  #         8   # 2 # used
    "POWER",  #             9   # 3 # used
    "FEEDRATE",  #          10  # 4 # used
    "POWER_DIFF",  #        11  # 5 # excluded
    "FEEDRATE_DIFF"  #      12  # 6 # excluded
]

map_param_str_to_show_name = {
    "EXP_ID": 'Exp_ID \n',   # 0
    "POINT_ID": 'Point_ID \n',   #, 1
    "FREQUENCY": 'Pattern \nFrequency', # 2
    "POWER_PATTERN": 'Power \nPattern',  # 3
    "FEEDRATE_PATTERN": 'Speed \nPattern',  # 4
    "LINEIDX": 'Line \nIndex', # 5
    "RTCP": 'RTCP \n',  # 6   # 0 # used
    "CLOCKWISE": 'Direction \n', # 7   # 1 # used
    "CURVATURE":  'Curvature \n(1/mm)', # 8   # 2 # used
    "POWER": 'Power \n(W)', # 9   # 3 # used
    "FEEDRATE": 'Speed \n(mm/min)',  # 10  # 4 # used
    "POWER_DIFF": 'Power Change \nRate (W/s)',  # 11  # 5 # excluded
    "FEEDRATE_DIFF": 'Speed Change \nRate (mm/s^2)',  # 12  # 6 # excluded

    'EDR': 'EDR \n(J/mm)',

    "AXIS_X": 'Pos X \n(mm)',
    "AXIS_Y": 'Pos Y \n(mm)',
    "AXIS_Z": 'Pos Z \n(mm)',
    "AXIS_C": 'Pos C \n(deg)',

    "WCS_AXIS_X": 'WCS Pos X \n(mm)',
    "WCS_AXIS_Y": 'WCS Pos Y \n(mm)',
    "WCS_AXIS_Z": 'WCS Pos Z \n(mm)',
    "WCS_AXIS_C": 'WCS Pos C \n(deg)',

    "VEL_X": 'Vel X \n(mm/s)',
    "VEL_Y": 'Vel Y \n(mm/s)',
    "VEL_Z": 'Vel Z \n(mm/s)',
    "VEL_C": 'Vel C \n(deg/s)',

    "ACC_X": 'Acc X \n(mm/s^2)',
    "ACC_Y": 'Acc Y \n(mm/s^2)',
    "ACC_Z": 'Acc Z \n(mm/s^2)',
    "ACC_C": 'Acc C \n(deg/s^2)',

    'PEAK_POS': 'Peak Pos \n(mm)',
    'ABS(PEAK_POS)': 'Abs Peak Pos \n(mm)',

    'DISTANCE': 'Distance \n(mm)',

    'WIDTH': 'Width \n(mm)',
    'HEIGHT': 'Height \n(mm)',
    'AREA': 'Area \n(mm^2)',

    'W1': 'Width \n(mm)',
    'H': 'Height \n(mm)',
    'A': 'Area \n(mm^2)',

    'IoU': 'IoU',

    'img_area': 'Image Area',
    'img_center_x': 'Image CoM X',
    'img_center_y': 'Image CoM Y',
    'img_entropy': 'Image Entropy',
    'img_mean_value': 'Image Mean',
}

def param_id_to_str(id):
    return param_str_list[id]


pos_str_list = [
    "DISTANCE",  #      0
    "TIME",  #          1
    "AXIS_X",  #        2
    "AXIS_Y",  #        3
    "WCS_AXIS_X",  #    4
    "WCS_AXIS_Y",  #    5
    "AXIS_C",  #        6
    "VEL_X",  #         7
    "VEL_Y",  #         8
    "VEL_C",  #         9
    "ANGLE_WCS_AXIS",  #10
    "ANGLE_AXIS",  #    11
    "ACC_X",  #         12
    "ACC_Y",  #         13
    "ACC_C"  #          14
]


def pos_id_to_str(id):
    return pos_str_list[id]


excel_headers = [
    "EXP_ID", "POINT_ID",
    "FREQUENCY", "POWER_PATTERN", "FEEDRATE_PATTERN", "LINEIDX",
    "RTCP", "CLOCKWISE", "CURVATURE",
    "POWER", "FEEDRATE", "POWER_DIFF", "FEEDRATE_DIFF",

    "DISTANCE", "TIME",
    "AXIS_X", "AXIS_Y", "WCS_AXIS_X",
    "WCS_AXIS_Y", "AXIS_C", "VEL_X",
    "VEL_Y", "VEL_C",
    "ANGLE_WCS_AXIS", "ANGLE_AXIS",
    "ACC_X", "ACC_Y", "ACC_C",

    "W1", "W2", "W3", "W4", "H", "A",
    "SIG_PARA1", "SIG_PARA2", "SIG_PARA3", "SIG_PARA4", "SIG_PARA5", "SIG_PARA6", "SIG_PARA7",

    "REAL_PROFILE"
]


'''***********************************************************************'''
'''Dataset update for incremental learning '''
'''***********************************************************************'''

# for incremental learning
dataset_name_list_by_type = dataset_name_list + dataset_exclude_for_deploy

# De-duplicate while preserving order (deterministic)
dataset_name_list_by_type = list(dict.fromkeys(dataset_name_list_by_type))

_DOMAIN_PRIORITY = ("noise", "tooth", "sin", "square", "const")

def build_domains_from_dataset_names(
    names: list[str],
    *,
    priority: tuple[str, ...] = _DOMAIN_PRIORITY
) -> dict[str, list[str]]:
    """
    Split datasets into semantic 'domains' based on pattern tokens in dataset name.
    Names are expected like: <Low|High>_<patternA>_<patternB>_<id>
    """
    domains = {k: [] for k in priority}
    file_list = []
    for n in names:
        n_split = n.split("_")
        if len(n_split) > 2:
            target_type = n_split[-2]       # CURRENTLY: Only study the speed pattern
            for k in priority:
                if k == target_type:
                    file_name = f"{n}_dataset.pt"
                    domains[k].append(file_name)
                    file_list.append(file_name)
                    break
            else:
                print(f"Cannot assign dataset '{n}' to any domain in {priority}")
        else:
            print(f"Cannot assign dataset '{n}' to any domain in {priority} (illegal pattern)")
    return domains, file_list


def _dataset_names_to_pt_files(dataset_names: list[str]) -> list[str]:
    """
    Convert dataset base names into actual .pt filenames.
    Accepts either 'XXX' or 'XXX.pt'.
    """
    return [n if n.endswith(".pt") else f"{n}.pt" for n in dataset_names]

'''***********************************************************************'''
'''Dataset and Dataloader'''
'''***********************************************************************'''

''' 
Dataset Decomposition (Total: M+N dataset)
 - Development (M dataset)
    - Train (80%)
    - Val   (10%)
    - Test  (10%)
 - Deployment (N dataset)
    - Deploy
'''


class MyDataset(Dataset):

    def __init__(self, config):
        self.config = config

        ''' Load Data'''
        # load data from hard disk into tensor
        if config.dataset_name == 'simu_data':
            data_hidden_dim = (config.img_start_idx + config.img_embed_dim
                               + config.param_start_idx + config.param_size
                               + config.pos_start_idx + config.pos_size
                               + config.label_size + 1)  # last 1 for mask
            data_tensor = torch.randn((9999, data_hidden_dim))
        else:
            dataset_path = os.path.join(config.machine_dataset_dir, config.dataset_name)
            data_tensor = torch.load(dataset_path, weights_only=True)

        self.data_tensor = data_tensor  # into memory (not gpu memory)

        ''' Data extraction '''
        idx_lf, idx_rt = 0, 0

        # img
        idx_lf, idx_rt = idx_rt + config.img_start_idx, idx_rt + config.img_start_idx + config.img_embed_dim
        self.data_img = data_tensor[:, idx_lf:idx_rt].float().to(config.device)

        # param
        idx_lf, idx_rt = idx_rt + config.param_start_idx, idx_rt + config.param_start_idx + config.param_size
        self.data_param = data_tensor[:, idx_lf:idx_rt].float().to(config.device)

        # pos
        idx_lf, idx_rt = idx_rt + config.pos_start_idx, idx_rt + config.pos_start_idx + config.pos_size
        self.data_pos = data_tensor[:, idx_lf:idx_rt].float().to(config.device)

        # label
        idx_lf, idx_rt = idx_rt + config.label_start_index, idx_rt + config.label_start_index + config.label_crop_size
        label_index = torch.linspace(idx_lf, idx_rt - 1, config.label_size).long()
        self.data_label = data_tensor[:, label_index].float().to(config.device)

        ''' Get indexable data '''
        # | sequence look-back          | ego   | sequence look-ahead   |
        # | x[-B], ..., x[-2], x[-1],   | x[0], | x[1], y[2], ..., x[A] |
        # | y[-B], ..., y[-2], y[-1],   | y[0], | y[1], y[2], ..., y[A] |
        self.n_seq_enc_look_back = config.n_seq_enc_look_back
        self.n_seq_enc_look_ahead = config.n_seq_enc_look_ahead
        if config.n_seq_enc_total != self.n_seq_enc_look_back + self.n_seq_enc_look_ahead + 1:
            raise RuntimeError("self.n_seq_enc_total != self.n_seq_enc_look_back + self.n_seq_enc_look_ahead + 1")

        self.n_seq_dec_pool = config.n_seq_dec_pool
        if self.n_seq_dec_pool < 1:
            raise RuntimeError("self.n_seq_dec_pool < 1")

        '''Mask data'''
        # last column: mask for valid profiles
        valid_data_mask = data_tensor[:, -1].to(config.device) > 0

        # RTCP on/off, separate to check its effects
        rtcp_mask = self.data_param[:, 4] > 0
        if 'enable_rtcp' in config and config.enable_rtcp == 'on_only':
            data_mask = valid_data_mask & rtcp_mask
        elif 'enable_rtcp' in config and config.enable_rtcp == 'off_only':
            data_mask = valid_data_mask & ~rtcp_mask
        else:
            data_mask = valid_data_mask

        self.raw_data_index = (data_mask[self.n_seq_enc_look_back:
                                         len(data_mask) - self.n_seq_enc_look_ahead - (self.n_seq_dec_pool - 1)]
                               .nonzero().squeeze())
        self.raw_data_index += self.n_seq_enc_look_back
        self.data_len = len(self.raw_data_index)

        ''' Sub-sampling '''
        self.sys_sampling_interval = config.sys_sampling_interval
        self.data_len = self.data_len // self.sys_sampling_interval

    def apply_standardization(self, config):
        self.param_mean = config.param_mean
        self.param_std = config.param_std
        self.data_param = standardize_tensor(self.data_param, self.param_mean, self.param_std)

        self.pos_mean = config.pos_mean
        self.pos_std = config.pos_std
        self.data_pos = standardize_tensor(self.data_pos, self.pos_mean, self.pos_std)

    def apply_exclusion(self, config):
        param_idx = [i for i in range(self.data_param.shape[1]) if i not in config.param_exclude]
        self.data_param = self.data_param[:, param_idx]

        pos_idx = [i for i in range(self.data_pos.shape[1]) if i not in config.pos_exclude]
        self.data_pos = self.data_pos[:, pos_idx]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        raw_index = index * self.sys_sampling_interval
        idx_ego = self.raw_data_index[raw_index]
        idx_lf = idx_ego - self.n_seq_enc_look_back
        idx_rt = idx_ego + self.n_seq_enc_look_ahead + self.n_seq_dec_pool
        return (index,
                self.data_img[idx_lf: idx_rt],
                self.data_param[idx_lf: idx_rt],
                self.data_pos[idx_lf: idx_rt],
                self.data_label[idx_lf: idx_rt])

    def get_raw_data(self, index, index_shift):
        raw_index = index * self.sys_sampling_interval
        idx_ego = self.raw_data_index[raw_index].cpu() + index_shift
        return self.data_tensor[idx_ego]

    def clear_mem(self):
        self.data_img.to('cpu')
        del self.data_img

        self.data_param.to('cpu')
        del self.data_param

        self.data_pos.to('cpu')
        del self.data_pos

        self.data_label.to('cpu')
        del self.data_label

        del self.data_tensor


class MyCombinedDataset(Dataset):
    def __init__(self,
                 config,
                 dataset_names: list[str] | None = None,
                 file_list_override: list[str] | None = None):
        """
        If dataset_names or file_list_override is provided, only those datasets are loaded.
        - dataset_names: list of base dataset names (without .pt) OR filenames (with .pt)
        - file_list_override: explicit list of filenames (with .pt)
        """
        if file_list_override is not None:
            file_list = list(file_list_override)
        elif dataset_names is not None:
            file_list = _dataset_names_to_pt_files(dataset_names)
        else:
            if not config.enable_deploy_dataset:
                # iterate all dataset in the folder
                file_list = [
                    file for file in os.listdir(config.machine_dataset_dir)
                    if file.endswith('.pt') and file not in config.dataset_exclude_for_deploy
                ]
            else:
                file_list = [
                    file for file in os.listdir(config.machine_dataset_dir)
                    if file.endswith('.pt') and file in config.dataset_exclude_for_deploy
                ]

        # Deterministic ordering is important for reproducibility
        file_list = sorted(file_list)

        # Keep existing "iterate ratio" behavior
        file_num = max(1, int(len(file_list) * config.dataset_iterate_ratio))
        file_list = file_list[:file_num] if file_num > 0 else []

        self.dataset_num = file_num
        self.datasets = []
        self.dataset_len = []
        self.dataset_bytes = []
        print(f"> starting to load datasets (total number: [{file_num}])")
        for i_file, file_name in enumerate(tqdm(file_list)):
            config.dataset_name = file_name
            dataset = MyDataset(config)
            self.datasets.append(dataset)
            self.dataset_len.append(dataset.__len__())
            self.dataset_bytes.append(calculate_dataset_size(dataset))

        self.cumulative_sizes = np.cumsum(self.dataset_len)
        self.total_len = self.cumulative_sizes[-1]
        self.total_bytes = np.sum(np.array(self.dataset_bytes, dtype=np.int64))

    def apply_exclusion(self, config):
        for dataset in self.datasets:
            dataset.apply_exclusion(config)

    def apply_standardization(self, config):
        for dataset in self.datasets:
            dataset.apply_standardization(config)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if index >= self.total_len:
            raise RuntimeError(f"Index {index} is out of range")

        for i_dataset, cum_size in enumerate(self.cumulative_sizes):
            if index < cum_size:
                if i_dataset == 0:
                    dataset_index = index
                else:
                    dataset_index = index - self.cumulative_sizes[i_dataset - 1]
                item = self.datasets[i_dataset].__getitem__(dataset_index)
                return (index, item[1], item[2], item[3], item[4])

    def get_raw_data(self, index):
        if index >= self.total_len:
            raise RuntimeError(f"Index {index} is out of range")

        for i_dataset, cum_size in enumerate(self.cumulative_sizes):
            if index < cum_size:
                if i_dataset == 0:
                    dataset_index = index
                else:
                    dataset_index = index - self.cumulative_sizes[i_dataset - 1]
                return self.datasets[i_dataset].get_raw_data(dataset_index)

    def clear_mem(self):
        for dataset in self.datasets:
            dataset.clear_mem()

def split_dataset(dataset, config, shuffle=True):
    # train, val, test split
    assert sum(config.train_val_test_ratio) == 1.0, "The train_val_test_ratio must sum to 1.0"

    train_size = int(config.train_val_test_ratio[0] * len(dataset))
    val_size = int(config.train_val_test_ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    if shuffle:
        train_dataset, val_dataset, test_dataset = (
            random_split(dataset,
                         lengths=[train_size, val_size, test_size],
                         generator=torch.Generator().manual_seed(config.seed)))
    else:
        indices = list(range(len(dataset)))
        train_dataset = Subset(dataset, indices[:train_size]) if train_size > 0 else None
        val_dataset = Subset(dataset, indices[train_size:train_size + val_size]) if val_size > 0 else None
        test_dataset = Subset(dataset, indices[train_size + val_size:]) if test_size > 0 else None

    return (train_dataset, val_dataset, test_dataset,
            train_size, val_size, test_size)


def calculate_standardization(dataset, config):
    (train_dataset, val_dataset, test_dataset,
     train_size, val_size, test_size) = split_dataset(dataset, config, shuffle=False)

    # calculate_mean_std
    param_data_list = []
    pos_data_list = []
    print('> calculate standardization mean/std based on the train_dataset...')
    n_seq_enc_look_back = config.n_seq_enc_look_back
    for idx in tqdm(train_dataset.indices):
        # (index, item[1], item[2], item[3], item[4])
        items = train_dataset.dataset[idx]
        param_data_list.append(items[2][n_seq_enc_look_back].unsqueeze(0))  # add batch dimension
        pos_data_list.append(items[3][n_seq_enc_look_back].unsqueeze(0))

    param_data = torch.cat(param_data_list, dim=0)
    config.param_mean = param_data.mean(axis=0).cpu().tolist()
    config.param_std = param_data.std(axis=0).cpu().tolist()

    pos_data = torch.cat(pos_data_list, dim=0)
    config.pos_mean = pos_data.mean(axis=0).cpu().tolist()
    config.pos_std = pos_data.std(axis=0).cpu().tolist()


def get_dataloaders(dataset, config, shuffle=True):
    (train_dataset, val_dataset, test_dataset,
     train_size, val_size, test_size) = split_dataset(dataset, config, shuffle)

    train_sampler = None
    val_sampler = None
    test_sampler = None
    if 'enable_ddp' in config and config.enable_ddp:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=config.ddp_world_size,
                                           rank=config.ddp_local_rank) if train_size > 0 else None

        val_sampler = DistributedSampler(val_dataset,
                                         num_replicas=config.ddp_world_size,
                                         rank=config.ddp_local_rank) if train_size > 0 else None

        test_sampler = DistributedSampler(test_dataset,
                                          num_replicas=config.ddp_world_size,
                                          rank=config.ddp_local_rank) if train_size > 0 else None

    # prep dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True if shuffle else False,
                              num_workers=config.num_workers,
                              sampler=train_sampler if train_sampler is not None else None,
                              # pin_memory=True, # no need if already in gpu
                              drop_last=True,
                              ) if train_size > 0 else None
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers,
                            sampler=val_sampler if val_sampler is not None else None,
                            # pin_memory=True,
                            drop_last=True,
                            ) if val_size > 0 else None
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             sampler=test_sampler if test_sampler is not None else None,
                             # pin_memory=True,
                             drop_last=True,
                             ) if test_size > 0 else None

    return train_loader, val_loader, test_loader


def test_dataset():
    config = load_config()  # load default config

    print('\n> Testing Dataset...')
    print(f"> config.enable_iterate_dataset: [{config.enable_iterate_dataset}]")
    if config.enable_iterate_dataset:
        dataset = MyCombinedDataset(config)
        print('> dataset bytes (GB):', dataset.total_bytes / 1e9)
    else:
        dataset = MyDataset(config)
        calculate_dataset_size(dataset, print_size=True)

    print('> dataset length:', dataset.__len__())

    def print_data_item(data_item):
        for i_elem, element in enumerate(data_item):
            if isinstance(element, torch.Tensor):
                print(f"> element[{i_elem}] Tensor: "
                      f"  shape: {element.shape}, device: {element.device}"
                      f"  \tmean: {element.float().mean().item():.6f}"
                      f"  \tmax: {element.max().item():.6f}"
                      f"  \tstd: {element.float().std().item():.6f}")
            elif isinstance(element, (int, float)):
                print(f"> element[{i_elem}] Int/Long/Float: "
                      f"  value: {element}")

    print('\n> dataset first item:')
    data_item = dataset[0]
    print_data_item(data_item)

    print('\n> testing dataloader...')
    train_loader, val_loader, test_loader = get_dataloaders(dataset, config)
    print(f"> train loader size: {len(train_loader)} ")
    print(f"> val loader size: {len(val_loader)} ")
    print(f"> test loader size: {len(test_loader)} ")

    print('\n> train loader first item:')
    for data_item in train_loader:
        break
    print_data_item(data_item)


def create_dataset(data_file_path, img_root_dir, output_dir):
    print(f"> create_dataset_from_file: {data_file_path}")

    config = load_config()

    # Preprocess the image using ImageNet mean and std values adapted for grayscale
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    grey_scale_mean = sum(imagenet_mean) / len(imagenet_mean)  # average mean
    grey_scale_std = sum(imagenet_std) / len(imagenet_std)  # average std

    # Preprocess the image
    image_transform = transforms.Compose([
        transforms.CenterCrop(config.img_crop_pixel),
        transforms.Resize(config.img_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[grey_scale_mean], std=[grey_scale_std]),
    ])

    # Load the pre-trained ResNet-50 model with the updated argument
    import torchvision.models as models
    weights = models.ResNet18_Weights.DEFAULT
    cnn_model = models.resnet18(weights=weights)

    # Modify the first convolutional layer to accept a single channel
    # and average the weights across the three color channels
    original_conv1 = cnn_model.conv1
    cnn_model.conv1 = nn.Conv2d(1, 64,
                                kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    with torch.no_grad():
        cnn_model.conv1.weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))

    # Replace the final linear layer with nn.Identity
    cnn_model.fc = nn.Identity()

    # Freeze the parameters
    for param in cnn_model.parameters():
        param.requires_grad = False

    # Move model to GPU if available
    setup_local_device(config)
    cnn_model = cnn_model.to(config.device)
    cnn_model.eval()

    # Read the Excel file
    dataset_name = os.path.splitext(os.path.basename(data_file_path))[0]
    print(f"> loading csv table [{dataset_name}], pls wait...")
    t_temp = time.time()
    df = pd.read_csv(data_file_path)
    print(f"> table loaded, used time: {time.time() - t_temp:.3f} s")

    # Creating pytroch tensor
    tensors = []
    t_start = time.time()
    for index, row in df.iterrows():
        if index % 2000 == 0:
            t_elapsed = time.time() - t_start
            print(f'> dataset [{dataset_name}]'
                  f' | index[{index}]/{len(df)}'
                  f' | elapsed: {t_elapsed:.3f}s'
                  f' | remaining: {t_elapsed / (index + 1) * (len(df) - index):.3f}s'
                  f' | speed: {t_elapsed/(index+1)*1000}ms/frame')
        img_filename = row['IMG']
        img_path = os.path.join(img_root_dir, img_filename)

        # Load and preprocess image
        image = Image.open(img_path)
        if image.mode != 'L':
            image = image.convert('L')
        image = image_transform(image).unsqueeze(0).to(config.device)  # add a batch dimension
        cnn_features = cnn_model(image).cpu().squeeze(0)  # remove the batch dimension

        # Control parameters (columns B=2 to N=14)
        control_params = torch.tensor(row.iloc[1:14].values.astype(np.float32), dtype=torch.float32)

        # Position data (columns O=15 to AC=29)
        position_data = torch.tensor(row.iloc[14:29].values.astype(np.float32), dtype=torch.float32)

        # Label data (columns AD=30 to XS=643)
        label_data = torch.tensor(row.iloc[29:643].values.astype(np.float32), dtype=torch.float32)

        # Drop data if width (column AD=30) is 0 or negative
        if label_data[0] <= 0:
            continue

        # make all negative value (due to camera error) to 0
        label_data[label_data < 0] = 0

        # Create img mask (columns AD=30 to last)
        if (row.iloc[29:-1] <= 0).all():
            profile_mask = torch.tensor([0])
        else:
            profile_mask = torch.tensor([1])

        # === Image Statistics ===
        def extract_map_stats(img_np: np.ndarray, threshold=0.01):
            """Extract statistics from a grayscale image or Grad-CAM heatmap."""
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            bin_mask = img_norm > threshold
            area = np.sum(bin_mask) / img_norm.size
            entropy = shannon_entropy(img_norm)
            cy, cx = center_of_mass(img_norm)
            mean_val = np.mean(img_norm)
            std_val = np.std(img_norm)
            max_val = np.max(img_norm)
            min_val = np.min(img_norm)
            return [area, cx, cy, entropy, mean_val, std_val, max_val, min_val]

        img_np = image.squeeze().cpu().numpy()
        img_stats = torch.tensor(extract_map_stats(img_np), dtype=torch.float32)

        # Concatenate all data
        row_tensor = torch.cat((cnn_features, control_params, position_data, label_data, img_stats, profile_mask))
        tensors.append(row_tensor)

    # Stack all row tensors to create the final dataset tensor
    dataset_tensor = torch.stack(tensors)

    # profile_mask
    mask_num = len(tensors) - dataset_tensor[:, -1].sum()
    mask_ratio = mask_num / len(tensors)
    print(f"> mask_num: {mask_num}, total_len: {len(tensors)}, mask_ratio: {mask_ratio}")

    # Save the tensor
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tensor_save_path = os.path.join(output_dir, f"{dataset_name}_dataset.pt")
    torch.save(dataset_tensor, tensor_save_path)
    print(f"> dataset tensor saved at: {tensor_save_path} with size {dataset_tensor.size()}")


def create_all_dataset_in_parallel(data_root_dir, img_root_dir, output_dir, num_worker=1):
    dir_list = [os.path.join(data_root_dir, entry.name) for entry in os.scandir(data_root_dir)
                if entry.is_file()
                and entry.name.endswith('.csv')
                ]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        for i, dir in enumerate(dir_list):
            executor.submit(create_dataset, dir, img_root_dir, output_dir)


if __name__ == '__main__':
    # test_dataset()
    #
    img_root_dir = r'C:\mydata\dataset\p2_ded_bead_profile'
    # data_root_dir = r'C:\mydata\dataset\p2_ded_bead_profile\Post_Data_20241225'
    data_root_dir = r'C:\mydata\dataset\p2_ded_bead_profile\Post_Data_20240919'
    output_dir = r'C:\mydata\dataset\p2_ded_bead_profile\20250810'
    create_dataset(os.path.join(data_root_dir, 'High_sin_tooth_1.csv'), img_root_dir, output_dir)
    # create_all_dataset_in_parallel(data_root_dir, img_root_dir, output_dir, num_worker=8)
