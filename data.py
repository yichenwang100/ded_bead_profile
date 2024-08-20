import os, shutil, sys, re
from tqdm import tqdm

import torch, torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms

import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from util import *

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

        '''Prep Data Here'''
        # load data from hard disk into tensor
        if config.dataset_name == 'simu_data':
            data_hidden_dim = (config.img_start_idx + config.img_embed_dim
                               + config.param_start_idx + config.param_size
                               + config.pos_start_idx + config.pos_size
                               + config.output_size + 1)  # last 1 for mask
            data_tensor = torch.randn((9999, data_hidden_dim))
        else:
            dataset_path = os.path.join(config.dataset_dir, config.dataset_name)
            data_tensor = torch.load(dataset_path)

        self.data_tensor = data_tensor  # into memory (not gpu memory)
        self.param_index_lf = config.img_start_idx + config.img_embed_dim

        idx_lf, idx_rt = config.img_start_idx, config.img_start_idx + config.img_embed_dim
        self.data_img = data_tensor[:, idx_lf:idx_rt].to(config.device)

        idx_lf, idx_rt = idx_rt + config.param_start_idx, idx_rt + config.param_start_idx + config.param_size
        self.data_param = data_tensor[:, idx_lf:idx_rt].to(config.device)

        idx_lf, idx_rt = idx_rt + config.pos_start_idx, idx_rt + config.pos_start_idx + config.pos_size
        self.data_pos = data_tensor[:, idx_lf:idx_rt].to(config.device)

        idx_lf, idx_rt = idx_rt + config.output_start_index, idx_rt + config.output_start_index + config.output_size
        self.data_y = data_tensor[:, idx_lf:idx_rt].to(config.device)

        ''' Get indexable data '''
        # | sequence look-back          | ego   | sequence look-ahead   |
        # | x[-B], ..., x[-2], x[-1],   | x[0], | x[1], y[2], ..., x[A] |
        # | y[-B], ..., y[-2], y[-1],   | y[0], | y[1], y[2], ..., y[A] |
        self.n_seq_enc_look_back = config.n_seq_enc_look_back
        self.n_seq_enc_look_ahead = config.n_seq_enc_look_ahead
        if config.n_seq_enc_total != self.n_seq_enc_look_back + self.n_seq_enc_look_ahead + 1:
            raise RuntimeError("self.n_seq != self.n_seq_before + self.n_seq_after + 1")

        self.n_seq_dec_pool = config.n_seq_dec_pool

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
                                         -1 - self.n_seq_enc_look_ahead - self.n_seq_dec_pool]
                               .nonzero().squeeze())
        self.raw_data_index += self.n_seq_enc_look_back
        self.data_len = len(self.raw_data_index)

        ''' Use stride for sub-sampling '''
        self.sys_sampling_interval = config.sys_sampling_interval
        self.data_len = self.data_len // self.sys_sampling_interval

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        raw_index = index * self.sys_sampling_interval
        idx_ego = self.raw_data_index[raw_index]
        idx_lf = idx_ego - self.n_seq_enc_look_back
        idx_rt = idx_ego + self.n_seq_enc_look_ahead + self.n_seq_dec_pool + 1
        return (index,
                self.data_img[idx_lf: idx_rt],
                self.data_param[idx_lf: idx_rt],
                self.data_pos[idx_lf: idx_rt],
                self.data_y[idx_lf: idx_rt])

    def get_raw_data(self, index, index_shift):
        raw_index = index * self.sys_sampling_interval
        idx_ego = self.raw_data_index[raw_index].cpu() + index_shift
        return self.data_tensor[idx_ego, self.param_index_lf:]


class MyCombinedDataset(Dataset):
    def __init__(self, config):
        if not config.enable_deploy_dataset:
            # iterate all dataset in the folder
            file_list = [file for file in os.listdir(config.dataset_dir)
                         if file.endswith('.pt')
                         and file not in config.dataset_exclude]
            file_num = int(len(file_list) * config.dataset_iterate_ratio)
            file_list = file_list[:file_num]
        else:
            file_list = [file for file in os.listdir(config.dataset_dir)
                         if file.endswith('.pt')
                         and file in config.dataset_exclude]
            file_num = len(file_list)

        self.dataset_num = file_num
        self.datasets = []
        self.dataset_len = []
        self.dataset_bytes = []
        print(f"> Starting to load datasets (total number: [{file_num}])")
        for i_file, file_name in enumerate(tqdm(file_list)):
            config.dataset_name = file_name
            dataset = MyDataset(config)
            self.datasets.append(dataset)
            self.dataset_len.append(dataset.__len__())
            self.dataset_bytes.append(calculate_dataset_size(dataset))

        self.cumulative_sizes = np.cumsum(self.dataset_len)
        self.total_len = self.cumulative_sizes[-1]
        self.total_bytes = np.sum(np.array(self.dataset_bytes, dtype=np.int64))

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


def get_dataloaders(dataset, config, shuffle=True):
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

    # prep dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True if shuffle else False,
                              num_workers=config.num_workers,
                              # pin_memory=True, # no need if already in gpu
                              drop_last=True,
                              ) if train_size > 0 else None
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers,
                            # pin_memory=True,
                            drop_last=True,
                            ) if val_size > 0 else None
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
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

    print('\n> Dataset first item:')
    data_item = dataset[0]
    print_data_item(data_item)

    print('\n> Testing Dataloader...')
    train_loader, val_loader, test_loader = get_dataloaders(dataset, config)
    print(f"> train loader size: {len(train_loader)} ")
    print(f"> val loader size: {len(val_loader)} ")
    print(f"> test loader size: {len(test_loader)} ")

    print('\n> Train loader first item:')
    for data_item in train_loader:
        break
    print_data_item(data_item)


def create_dataset(xlsx_path, img_root_dir, output_dir):
    print(f"> create_dataset_from_xlsx: {xlsx_path}")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = cnn_model.to(device)

    # Read the Excel file
    dataset_name = os.path.splitext(os.path.basename(xlsx_path))[0]
    print(f"> loading excels table [{dataset_name}], pls wait...")
    df = pd.read_excel(xlsx_path)

    # Creating pytroch tensor
    tensors = []
    t_start = time.time()
    for index, row in df.iterrows():
        if index % 2000 == 0:
            print(f'> dataset [{dataset_name}] | index[{index}]/{len(df)} | elapsed: {time.time() - t_start}s')
        img_filename = row['IMG']
        img_path = os.path.join(img_root_dir, img_filename)

        # Load and preprocess image
        image = Image.open(img_path)
        if image.mode != 'L':
            image = image.convert('L')
        image = image_transform(image).unsqueeze(0).to(device)  # add a batch dimension
        # TODO
        if cnn_model.training():
            print('cnn_model training')
        else:
            print('cnn_model evaluating')
        cnn_features = cnn_model(image).cpu().squeeze(0)  # remove the batch dimension

        # Control parameters (columns B=1 to N=13)
        control_params = torch.tensor(row.iloc[1:14].values.astype(np.float32), dtype=torch.float32)

        # Position data (columns O=14 to AC=28)
        position_data = torch.tensor(row.iloc[14:29].values.astype(np.float32), dtype=torch.float32)

        # Label data (columns AD=29 to DR=121)
        label_data = torch.tensor(row.iloc[29:122].values.astype(np.float32), dtype=torch.float32)

        # make all negative value (due to camera error) to 0
        label_data[label_data < 0] = 0

        # Create img mask (columns AQ=42 to CD=81)
        if (row.iloc[42:82] == 0).all():
            profile_mask = torch.tensor([0])
        else:
            profile_mask = torch.tensor([1])

        # Concatenate all data
        row_tensor = torch.cat((cnn_features, control_params, position_data, label_data, profile_mask))
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
    print(f"> Dataset tensor saved at: {tensor_save_path} with size {dataset_tensor.size()}")


def create_all_dataset_in_parallel(xlsx_root_dir, img_root_dir, output_dir, num_worker=1):
    dir_list = [os.path.join(xlsx_root_dir, entry.name) for entry in os.scandir(xlsx_root_dir)
                if entry.is_file()
                and entry.name.endswith('.xlsx')
                ]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        for i, dir in enumerate(dir_list):
            executor.submit(create_dataset, dir, img_root_dir, output_dir)


if __name__ == '__main__':
    test_dataset()

    # img_root_dir = r'C:\mydata\dataset\p2_ded_bead_profile'
    # excel_root_dir = r'C:\mydata\dataset\p2_ded_bead_profile\Post_Data_20240730'
    # output_dir = r'C:\mydata\dataset\p2_ded_bead_profile\20240730'
    # create_dataset(os.path.join(excel_root_dir, 'High_const_sin_1.xlsx'), img_root_dir, output_dir)
    # create_all_dataset_in_parallel(excel_root_dir, img_root_dir, output_dir, num_worker=1)
