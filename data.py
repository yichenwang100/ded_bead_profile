import os, shutil, sys, re
from tqdm import tqdm

import torch, torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from util import *

'''***********************************************************************'''
'''Project related settings'''
'''***********************************************************************'''

param_str_list = [
    "EXP_ID",
    "POINT_ID",
    "FREQUENCY",
    "POWER_PATTERN",
    "FEEDRATE_PATTERN",
    "LINEIDX",
    "RTCP",
    "CLOCKWISE",
    "CURVATURE",
    "POWER",
    "FEEDRATE",
    "POWER_DIFF",
    "FEEDRATE_DIFF"
]


def param_id_to_str(id):
    return param_str_list[id]


pos_str_list = [
    "DISTANCE",
    "TIME",
    "AXIS_X",
    "AXIS_Y",
    "WCS_AXIS_X",
    "WCS_AXIS_Y",
    "AXIS_C",
    "VEL_X",
    "VEL_Y",
    "VEL_C",
    "ANGLE_WCS_AXIS",
    "ANGLE_AXIS",
    "ACC_X",
    "ACC_Y",
    "ACC_C"
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

        '''Prep Data Here'''
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

        idx_lf, idx_rt = 0, 0

        idx_lf, idx_rt = idx_rt + config.img_start_idx, idx_rt + config.img_start_idx + config.img_embed_dim
        self.data_img = data_tensor[:, idx_lf:idx_rt].to(config.device)

        idx_lf, idx_rt = idx_rt + config.param_start_idx, idx_rt + config.param_start_idx + config.param_size
        self.data_param = data_tensor[:, idx_lf:idx_rt].to(config.device)

        idx_lf, idx_rt = idx_rt + config.pos_start_idx, idx_rt + config.pos_start_idx + config.pos_size
        self.data_pos = data_tensor[:, idx_lf:idx_rt].to(config.device)

        idx_lf, idx_rt = idx_rt + config.label_start_index, idx_rt + config.label_start_index + config.label_crop_size
        label_index = torch.linspace(idx_lf, idx_rt - 1, config.label_size).long()
        self.data_label = data_tensor[:, label_index].to(config.device)

        ''' Get indexable data '''
        # | sequence look-back          | ego   | sequence look-ahead   |
        # | x[-B], ..., x[-2], x[-1],   | x[0], | x[1], y[2], ..., x[A] |
        # | y[-B], ..., y[-2], y[-1],   | y[0], | y[1], y[2], ..., y[A] |
        self.n_seq_enc_look_back = config.n_seq_enc_look_back
        self.n_seq_enc_look_ahead = config.n_seq_enc_look_ahead
        if config.n_seq_enc_total != self.n_seq_enc_look_back + self.n_seq_enc_look_ahead + 1:
            raise RuntimeError("self.n_seq != self.n_seq_before + self.n_seq_after + 1")

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
                                         len(data_mask) - self.n_seq_enc_look_ahead - (self.n_seq_dec_pool-1)]
                               .nonzero().squeeze())
        self.raw_data_index += self.n_seq_enc_look_back
        self.data_len = len(self.raw_data_index)

        ''' Sub-sampling '''
        self.sys_sampling_interval = config.sys_sampling_interval
        self.data_len = self.data_len // self.sys_sampling_interval

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


class MyCombinedDataset(Dataset):
    def __init__(self, config):
        if not config.enable_deploy_dataset:
            # iterate all dataset in the folder
            file_list = [file for file in os.listdir(config.machine_dataset_dir)
                         if file.endswith('.pt')
                         and file not in config.dataset_exclude]
            file_num = int(len(file_list) * config.dataset_iterate_ratio)
            file_list = file_list[:file_num]
        else:
            file_list = [file for file in os.listdir(config.machine_dataset_dir)
                         if file.endswith('.pt')
                         and file in config.dataset_exclude]
            file_num = len(file_list)

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


def convert_xlsx_to_csv(folder_path):
    # List all files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.xlsx'):
            # Construct full file path
            file_path = os.path.join(folder_path, filename)

            # Read the Excel file
            df = pd.read_excel(file_path, engine='openpyxl')

            # Create the CSV file path
            csv_filename = filename.replace('.xlsx', '.csv')
            csv_file_path = os.path.join(folder_path, csv_filename)

            # Save the DataFrame to a CSV file
            df.to_csv(csv_file_path, index=False)

            print(f"Converted {filename} to {csv_filename}")


def calculate_statistics(csv_path):
    df = pd.read_csv(csv_path)
    stats = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            stats[col] = {stat: np.nan for stat in
                          ['mean', 'median', 'std', 'min', 'max', 'num_zeros', 'num_negative', 'num_positive',
                           'pct_zeros']}
        else:
            total_count = len(df[col])
            zero_count = (df[col] == 0).sum()
            precision = 5
            stats[col] = {
                'mean': round(df[col].mean(), precision),
                'median': round(df[col].median(), precision),
                'std': round(df[col].std(), precision),
                'min': round(df[col].min(), precision),
                'max': round(df[col].max(), precision),
                'num_zeros': zero_count,
                'pct_zeros': round((zero_count / total_count) * 100, precision),
                'num_negative': (df[col] < 0).sum(),
                'num_positive': (df[col] > 0).sum()
            }
    return stats, len(df), df.columns  # Return stats, number of rows, and the column order


def compute_stats_for_all_csv(directory_path):
    results = []
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        stats, data_size, columns = calculate_statistics(os.path.join(directory_path, csv_file))
        for stat_name in ['mean', 'median', 'std', 'min', 'max', 'num_zeros', 'pct_zeros', 'num_negative',
                          'num_positive']:
            row = {
                'Dataset': csv_file,
                'Data Size': data_size,
                'Statistic Type': stat_name.capitalize()
            }
            for col in columns:
                row[col] = stats[col][stat_name]
            results.append(row)

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Sort the DataFrame by the 'Statistic Type' column and then by the 'EXP_ID' column
    results_df = results_df.sort_values(by=['Statistic Type', 'EXP_ID'])

    # Determine the output file path
    output_csv = os.path.join(os.path.dirname(directory_path), f"{os.path.basename(directory_path)}_stats.csv")

    # Save the sorted DataFrame to the output CSV file
    results_df.to_csv(output_csv, index=False, float_format='%.5f')
    print(f"Statistics saved to {output_csv}")


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
            print(f'> dataset [{dataset_name}] '
                  f'| index[{index}]/{len(df)} '
                  f'| elapsed: {time.time() - t_start:.3f}s'
                  f'| remaining: {(time.time() - t_start) / (index + 1) * (len(df) - index):.3f}s')
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
    data_root_dir = r'C:\mydata\dataset\p2_ded_bead_profile\Post_Data_20240911'
    output_dir = r'C:\mydata\dataset\p2_ded_bead_profile\20240911'
    # convert_xlsx_to_csv(data_root_dir)
    # compute_stats_for_all_csv(data_root_dir)
    # create_dataset(os.path.join(data_root_dir, 'High_const_sin_2.csv'), img_root_dir, output_dir)
    create_all_dataset_in_parallel(data_root_dir, img_root_dir, output_dir, num_worker=16)
