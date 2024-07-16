import os
from tqdm import tqdm
import torch, torchvision
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from util import *


class LightDataset(Dataset):

    def __init__(self, config, data_mean=None, data_std=None):
        self.config = config

        '''The sequence length'''
        # | sequence before | this  | sequence after|
        # | x, x, ..., x,   | x,    | x, x, ..., x, |
        self.n_seq_before = config.n_seq_before
        self.n_seq_after = config.n_seq_after
        self.n_seq = self.n_seq_before + self.n_seq_after + 1
        self.input_size = config.resnet_fc_size
        self.output_size = config.output_size
        self.param_size = config.param_size

        '''Prep Data Here'''
        # load data from hard disk into tensor
        if config.enable_light_dataset:
            dataset_path = os.path.join(config.dataset_dir, config.dataset_name) + ".pt"
            data_raw = torch.load(dataset_path)

        else:  # use random dataset
            total_data_size = 50000
            data_raw = torch.randn((total_data_size,
                                         self.n_seq,
                                         self.input_size + self.output_size + self.param_size))

        self.data_len = len(data_raw)
        self.data_x = data_raw[:, :, 0:self.input_size].to(config.device)
        self.data_y_origin = data_raw[:, 0, self.input_size: self.input_size + self.output_size].to(config.device)
        self.data_y = self.data_y_origin.clone()
        self.data_p = data_raw[:, :, self.input_size + self.output_size:].to(config.device)

        ''' standardization of y lables '''
        if config.enable_standardization:
            if data_mean is None:
                self.data_mean = self.data_y_origin.view(-1, config.output_size).mean(0)
            else:
                self.data_mean = torch.tensor(data_mean).to(config.device)

            if data_std is None:
                self.data_std = self.data_y_origin.view(-1, config.output_size).std(0)
            else:
                self.data_std = torch.tensor(data_std).to(config.device)

            self.data_y = (self.data_y - self.data_mean) / self.data_std

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], self.data_p[index], self.data_y_origin[index]


def get_dataloaders(dataset, config, shuffle=True):
    # train, val, test split
    if 'train_val_test_ratio' not in config:
        config.train_val_test_ratio = [0.8, 0.1, 0.1]

    assert sum(config.train_val_test_ratio) == 1.0, "The train_val_test_ratio must sum to 1.0"

    train_size = int(config.train_val_test_ratio[0] * len(dataset))
    val_size = int(config.train_val_test_ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    if shuffle:
        train_dataset, val_dataset, test_dataset = (
            random_split(dataset, lengths=[train_size, val_size, test_size]))
    else:
        indices = list(range(len(dataset)))
        train_dataset = Subset(dataset, indices[:train_size])
        val_dataset = Subset(dataset, indices[train_size:train_size + val_size])
        test_dataset = Subset(dataset, indices[train_size + val_size:])

    if config.enable_standardization:
        val_dataset.dataset.data_mean = train_dataset.dataset.data_mean
        val_dataset.dataset.data_std = train_dataset.dataset.data_std
        test_dataset.dataset.data_mean = test_dataset.dataset.data_mean
        test_dataset.dataset.data_std = test_dataset.dataset.data_std


    # prep dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True if shuffle else False,
                              num_workers=config.num_workers,
                              drop_last=True,
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=config.num_workers,
                            drop_last=True,
                            )
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=config.num_workers,
                             drop_last=True,
                             )

    return train_loader, val_loader, test_loader


def TestDataTool():
    config = load_config()  # load default config

    dataset = LightDataset(config)
    x, y = dataset[0]
    print('\ndataset size:', len(dataset))
    print('dataset in MB: ', x.element_size() * x.numel() * len(dataset) // (1024 ** 2))
    print('dataset device: ', x.device)
    print('x shape: ', x.shape)
    print('y shape: ', y.shape)
    print('x device', x.device)
    print('y device', y.device)

    train_loader, val_loader, test_loader = get_dataloaders(dataset, config)
    for x, y in train_loader:
        break

    print(f"\ntrain loader size: {len(train_loader)} mean: {train_loader.dataset.dataset.data_mean} std: {train_loader.dataset.dataset.data_std}")
    print(f"val loader size: {len(val_loader)} mean: {val_loader.dataset.dataset.data_mean} std: {val_loader.dataset.dataset.data_std}")
    print(f"test loader size: {len(test_loader)} mean: {test_loader.dataset.dataset.data_mean} std: {test_loader.dataset.dataset.data_std}")
    print('x shape: ', x.shape)
    print('y shape: ', y.shape)
    print('x device', x.device)
    print('y device', y.device)


def compress_data_to_tensor(input_dir):
    config = load_config()

    def get_keys(filename):
        # for folder name like '1_0.8_1_10' [frequency, amplitude, channel_id, sample_id]
        parts = filename.split('_')
        frequency, amplitude, channel_id, sample_id = [float(parts[i]) for i in range(len(parts))]
        return frequency, amplitude, channel_id, sample_id

    def sort_key(filename):
        # for folder name like '1_0.8_1_10'
        a, b, c, d = get_keys(filename)
        number = a * 1e9 + b * 1e6 + c * 1e3 + d
        return int(number)

    dir_list = [entry.name for entry in os.scandir(input_dir) if entry.is_dir()]
    dir_list = sorted(dir_list, key=sort_key)
    data_len = len(dir_list)
    print(f"> data_len is [{data_len}] in folder {input_dir} ")

    n_seq_before = config.n_seq_before
    n_seq_after = config.n_seq_after
    n_seq = n_seq_before + n_seq_after + 1

    config.resnet_fc_size = 512     # the default features
    config.param_size = (4      # from parameters
                         + 2    # from overlap
                         + 1    # from spatio
                         + 4    # from file index
                         )

    output_x = torch.zeros((data_len, n_seq, config.resnet_fc_size))
    output_y = torch.zeros((data_len, n_seq, config.output_size))
    output_p = torch.zeros((data_len, n_seq, config.param_size))

    for i_dir, dir in enumerate(dir_list):
        # load x (input)
        for i_seq, seq in enumerate(range(-n_seq_before, config.n_seq_before + 1)):
            temp_file_path = os.path.join(input_dir, dir + '/T_' + str(seq) + '.fea')
            temp_x = torch.load(temp_file_path)
            output_x[i_dir, i_seq] = temp_x

        # load y (output label)
        temp_file_path = os.path.join(input_dir, dir + '/results.txt')
        with open(temp_file_path, 'r') as file:
            for line in file:
                # Custom parsing logic here
                line_data = line.strip().split(',')
                line_data = [float(num) for num in line_data]
                temp_y = torch.tensor(line_data)
            output_y[i_dir, :, :] = torch.stack([temp_y] * n_seq, dim=0)

        # load p (parameter)
        p_index = 0
        temp_file_path = os.path.join(input_dir, dir + '/parameters.txt')
        with open(temp_file_path, 'r') as file:
            i_line = 0
            for line in file:
                # Custom parsing logic here
                line_data = line.strip().split(',')
                line_data = [float(num) for num in line_data]
                output_p[i_dir, i_line, p_index:p_index+4] = torch.tensor(line_data)
                i_line += 1
        p_index += 4

        # load p (overlap)
        temp_file_path = os.path.join(input_dir, dir + '/overlap.txt')
        with open(temp_file_path, 'r') as file:
            i_line = 0
            for line in file:
                # Custom parsing logic here
                line_data = line.strip().split(',')
                line_data = [float(num) for num in line_data]
                output_p[i_dir, :, p_index] = torch.tensor(line_data)
                p_index += 1

        # load p (spatio)
        temp_file_path = os.path.join(input_dir, dir + '/spatio.txt')
        with open(temp_file_path, 'r') as file:
            i_line = 0
            for line in file:
                # Custom parsing logic here
                line_data = line.strip().split(',')
                line_data = [float(num) for num in line_data]
                output_p[i_dir, :, p_index] = torch.tensor(line_data)
                p_index += 1

        # add keys from filename
        output_p[i_dir, :, p_index:p_index+4] = torch.tensor(get_keys(dir))

    output = torch.cat((output_x, output_y, output_p), dim=2)
    output_path = input_dir.rstrip('/') + '.pt'
    torch.save(output, output_path)


def clean_data(path):
    config = load_config()

    data = torch.load(path)

    # check the output columns
    # output_x = torch.zeros((data_len, n_seq, config.resnet_fc_size))
    # output_y = torch.zeros((data_len, n_seq, config.output_size))
    # output_p = torch.zeros((data_len, n_seq, config.param_size))

    data_len = data.shape[0]
    output_y = data[:, :, config.resnet_fc_size:config.resnet_fc_size + config.output_size]

    # Identify rows where all values are zero
    non_zero_index = output_y.any(dim=-1).any(dim=-1)
    null_data_count = (data_len - torch.sum(non_zero_index)).numpy()
    print(f"> path: {path}"
          f" | null data: [{null_data_count}/{data_len}={null_data_count / data_len * 100:.3f}%]")
    data_filtered = data[non_zero_index]

    # save data
    new_file_name = os.path.join(os.path.dirname(path), f"clean_{os.path.basename(path)}")
    torch.save(data_filtered, new_file_name)

    # prep data for standardization
    data_output = data_filtered[:, :, config.resnet_fc_size:config.resnet_fc_size + config.output_size]
    output_mean = torch.mean(data_output.view(-1, config.output_size), dim=-0)
    output_std = torch.std(data_output.view(-1, config.output_size), dim=0)
    output_size = len(output_mean)
    return output_mean, output_std, output_size


def compress_all_data(root_dir, num_worker=1):
    dir_list = [os.path.join(root_dir, entry.name) for entry in os.scandir(root_dir)
                if entry.is_dir()
                and entry.name.startswith('interval')]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        for i, dir in enumerate(dir_list):
            executor.submit(compress_data_to_tensor, dir)


def clean_all_data(root_dir, num_worker=1):
    dir_list = [os.path.join(root_dir, entry.name) for entry in os.scandir(root_dir)
                if entry.is_file() and entry.name.endswith('.pt')
                and entry.name.startswith('interval')]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        for i, dir in enumerate(dir_list):
            executor.submit(clean_data, dir)


if __name__ == '__main__':
    # TestDataTool()

    #  test: convert single folder
    # parent_dir = r'C:\Users\wangy\Downloads\dataset\1_0.8_1_1'
    # compress_data_to_tensor(parent_dir)

    # # convert multiple folder in the root dir
    # root_dir = r'C:\Users\wangy\Downloads\dataset'
    # compress_all_data(root_dir, num_worker=8)

    # # clean all data in single folder
    root_dir = r'C:\mydata\dataset\proj_melt_pool_pred\20240625_single_line'
    clean_all_data(root_dir, num_worker=8)
