# src/utils/config.py
import os, shutil, time, copy, inspect, re
import argparse, yaml
from pprint import pprint
from datetime import datetime

'''***********************************************************************'''
''' Torch and tensor'''
'''***********************************************************************'''

import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


# seed os, random, numpy, torch, etc.
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_dataset_size(obj, print_size=False):
    total_size = 0
    for attr_name in dir(obj):
        # Skip private attributes and methods
        if attr_name.startswith('_'):
            continue

        attr_value = getattr(obj, attr_name)
        if isinstance(attr_value, torch.Tensor):
            tensor_size = get_tensor_size(attr_value)

            # Print the attribute name and value
            if print_size:
                print(f'{attr_name} has size {tensor_size / 1e6} Mb')

            total_size += tensor_size

    if print_size:
        print(f"total size: {total_size / 1e6} Mb")
    return total_size


def get_tensor_size(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.nelement()
    return 0


def get_model_parameter_num(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_neuron_num(model):
    total_neurons = 0

    def count_neurons(layer):
        nonlocal total_neurons
        if isinstance(layer, nn.Linear):
            total_neurons += layer.out_features
        elif isinstance(layer, nn.Conv2d):
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size[0] * layer.kernel_size[1]  # Assuming square kernels
            out_neurons = out_channels * kernel_size
            total_neurons += out_neurons
        elif isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
            num_directions = 2 if layer.bidirectional else 1
            total_neurons += layer.hidden_size * num_directions
        elif isinstance(layer, nn.Module):
            for sublayer in layer.children():
                count_neurons(sublayer)

    count_neurons(model)
    return total_neurons


def init_model_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            elif 'weight_hh' in name:
                torch.nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            elif 'bias' in name:
                param.data.fill_(0)
    elif isinstance(m, nn.MultiheadAttention):
        torch.nn.init.xavier_uniform_(m.in_proj_weight)
        if m.in_proj_bias is not None:
            torch.nn.init.zeros_(m.in_proj_bias)
    elif isinstance(m, nn.Parameter):
        torch.nn.init.uniform_(m, a=0.0, b=1.0)

import subprocess
def get_gpu_memory():
    # Using `nvidia-smi` to get the memory usage of GPUs
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], encoding='utf-8'
    )
    gpu_memory = [tuple(map(int, x.split(', '))) for x in result.strip().split('\n')]
    return gpu_memory

def get_least_used_gpu():
    gpu_memory = get_gpu_memory()
    gpu_memory_usage = [used/total for used, total in gpu_memory]
    least_used_gpu = gpu_memory_usage.index(min(gpu_memory_usage))
    return least_used_gpu

def ddp_setup(local_rank, world_size):
    dist.init_process_group('ncll',
                            init_method='env://',
                            rank=local_rank,
                            world_size=world_size)


def ddp_cleanup():
    dist.destroy_process_group()


'''***********************************************************************'''
'''Formatting and displaying '''
'''***********************************************************************'''


def elapsed_time_to_HHMMSS(elapsed_time):
    hours, rem = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def curr_time_to_str(format="%y%m%d-%H%M%S"):
    # Get the current date and time
    now = datetime.now()

    # Format the date and time
    formatted_date_time = now.strftime(format)

    return formatted_date_time


def function_name():
    # Get the current frame
    current_frame = inspect.currentframe()
    # Get the calling frame
    calling_frame = inspect.getouterframes(current_frame, 2)
    return f"{calling_frame[1].function}"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def int_split_by_comma(number):
    return re.sub(r'(?<!^)(?=(\d{3})+$)', ',', str(number))


'''***********************************************************************'''
''' Config, yaml, and file directories'''
'''***********************************************************************'''


# dict could be referenced by '.', for example, some_dict.key
class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __deepcopy__(self, memo):
        new_dict = AttributeDict()
        # Add a deepcopy of each item in the original dict to the new dict
        for key, value in self.items():
            new_dict[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return new_dict


def load_attribute_dict(path):
    with open(path, 'r') as file:
        attribute_dict = yaml.safe_load(file)
        return AttributeDict(attribute_dict)


# load config by its file name
def load_config(config_path='config.yaml'):
    # load the main config
    main_config = load_attribute_dict(config_path)

    # load machine.config
    machine_config = load_attribute_dict(main_config.machine_config_path)

    # merge machine config into the main config
    merge_dicts(main_config, machine_config)
    return main_config


def save_config(config, config_path='config.yaml'):
    with open(config_path, 'w') as file:
        config_save = copy.deepcopy(dict(config))
        config_save["device"] = None
        yaml.dump(config_save, file)


def merge_dicts(dict1, dict2):
    """Recursively merge two dictionaries."""
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = value


# convert config with parser of args from the shell command
def get_config_from_cmd(parser):
    '''
    1.load the default config file
    2.load the user-input config file in command line, if applies
    3.update the user-input kay values in command line, if applies
    '''

    # load the default config as a dict()
    default_config_path = 'config.yaml'
    proj_config = load_config(default_config_path)

    # auto add all config items into the parser
    def add_arguments(parser, config, prefix=''):
        for key, value in config.items():
            if isinstance(value, dict):
                add_arguments(parser, value, prefix + key + '.')
            else:
                arg_type = type(value)
                arg_type = str2bool if isinstance(value, bool) else type(value)  # process bool type
                parser.add_argument(f'--{prefix}{key}', type=arg_type, default=None, help=f'Default: {value}')

    # get the user input args
    add_arguments(parser, proj_config)
    args = parser.parse_args()

    # update the user config file
    if args.config_path is not None:
        print(f"\n> loading use-input config file: {args.config_path}...")
        user_config = load_config(args.config_path)
        proj_config.update(user_config)
    else:
        print(f"\n> applying system-default config file: {default_config_path}...")

    # update the user input args
    new_args_dict = {key: value for key, value in vars(args).items() if value is not None}
    proj_config.update(new_args_dict)

    return proj_config


def setup_local_device(config):
    if config.enable_gpu and torch.cuda.is_available():
        # auto get core index with the min memory used
        if config.dp_core_idx < 0:
            config.dp_core_idx = get_least_used_gpu()

        dev_name = f'cuda:{config.dp_core_idx}'

    else:
        dev_name = "cpu"

    config.device = torch.device(dev_name)
    print('> dev: ', dev_name)

    # set up ddp
    if 'enable_ddp' in config and config.enable_ddp:
        ddp_cleanup()

        if config.ddp_local_rank < 0:
            config.ddp_local_rank = 0

        if config.ddp_world_size < 0:
            config.ddp_world_size = torch.cuda.device_count()

        ddp_setup(local_rank=config.ddp_local_rank, world_size=config.ddp_world_size)


# prep all local-machine related dir, naming, etc.
def setup_local_config(config):
    ''' setup the basic environment from config
    - project naming
    - disk directory
    - gpu and multi tasking
    '''

    ''' project naming '''
    if config.enable_uuid_naming:
        # get a shortened UUID
        # def get_uuid(length=-1):
        #     import uuid, base64
        #     # Generate a UUID
        #     uuid_obj = uuid.uuid4()
        #
        #     # Convert UUID to bytes
        #     uuid_bytes = uuid_obj.bytes
        #
        #     # Encode bytes to Base64
        #     short_uuid = base64.urlsafe_b64encode(uuid_bytes).rstrip(b'=').decode('utf-8')
        #
        #     if length > 0:
        #         return short_uuid[0:length]
        #
        #     else:
        #         return short_uuid

        # Get the high resolution performance counter of CPU
        high_prec_time = time.perf_counter()
        decimal_part = f"{high_prec_time - int(high_prec_time):.4f}".split('.')[1]
        task_name = f"{curr_time_to_str()}.{decimal_part}"

    else:
        task_name = (f"{config.dataset_name}"
                     f" {config.model}"
                     f" b={config.batch_size}"
                     f" lr={config.lr:.1E}"
                     )
        if config.enable_adaptive_lr:
            task_name += f"adap"
        if config.enable_weight_decay:
            task_name += f" wd={config.weight_decay:.1E}"
        if config.enable_dropout:
            task_name += f" drop={config.dropout}"

    task_name += f".{config.extra_name}" if config.extra_name is not None else ""
    print("> task_name: ", task_name)

    ''' disk directory '''
    # dataset
    config.machine_dataset_dir = os.path.join(config.data_root_dir, config.dataset_dir)
    print("> machine_dataset_dir: ", os.path.abspath(config.machine_dataset_dir))

    # output
    config.machine_output_dir = os.path.join(config.data_root_dir, config.output_dir, task_name)

    if config.enable_rewrite_output_dir:  # remove the project dir if it exists
        if os.path.exists(config.machine_output_dir):
            shutil.rmtree(config.machine_output_dir)
            time.sleep(0.2)
            print("! machine_output_dir removed:\t", os.path.abspath(config.machine_output_dir))

    os.makedirs(config.machine_output_dir, exist_ok=True)
    print("> machine_output_dir: ", os.path.abspath(config.machine_output_dir))

    # log
    config.machine_log_dir = os.path.join(config.machine_output_dir, config.log_dir)
    os.makedirs(config.machine_log_dir, exist_ok=True)
    # print("> machine_log_dir:\t", os.path.abspath(config.machine_log_dir))

    # check point
    config.machine_checkpoint_dir = os.path.join(config.machine_output_dir, config.checkpoint_dir)
    os.makedirs(config.machine_checkpoint_dir, exist_ok=True)
    # print("> machine_checkpoint_dir: ", os.path.abspath(config.machine_checkpoint_dir))  # prep all dir

    ''' GPU and device setting '''
    setup_local_device(config)


def test_load_config():
    print('test_load_config')
    config = load_config()
    pprint(config)


'''***********************************************************************'''
'''Mics tools'''
'''***********************************************************************'''
import shutil
def copy_folder(src, dst):
    try:
        shutil.copytree(src, dst)
        print(f"Folder '{src}' copied to '{dst}' successfully.")
    except Exception as e:
        print(f"Error: {e}")


def standardize_tensor(data, mean, std, eps=1e-6):
    if len(mean) != data.shape[1]:
        raise RuntimeError('!Data dimension dis-matched! len(mean) != data.shape[1]')

    if len(std) != data.shape[1]:
        raise RuntimeError('!Data dimension dis-matched! len(std) != data.shape[1]')

    for i_feature in range(data.shape[1]):
        data[:, i_feature] -= mean[i_feature]
        if abs(std[i_feature]) > eps:
            data[:, i_feature] /= std[i_feature]

    return data


def reverse_standardize_tensor(data, mean, std):
    if len(mean) != data.shape[1]:
        raise RuntimeError('!Data dimension dis-matched! len(mean) != data.shape[1]')

    if len(std) != data.shape[1]:
        raise RuntimeError('!Data dimension dis-matched! len(std) != data.shape[1]')

    for i_feature in range(data.shape[1]):
        data *= std[i_feature]
        data += mean[i_feature]

    return data


if __name__ == '__main__':
    # test config loading
    parser = argparse.ArgumentParser()
    config = get_config_from_cmd(parser)
    pprint(config)
