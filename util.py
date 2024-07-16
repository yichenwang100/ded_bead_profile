# src/utils/config.py
import argparse, os, shutil, time, random, copy
import yaml
from pprint import pprint
import numpy as np, torch


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


# load config by its file name
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        config = AttributeDict(config)
        config.device = torch.device("cuda" if config.enable_gpu and torch.cuda.is_available() else "cpu")
    return config


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_config(config, config_path='config.yaml'):
    with open(config_path, 'w') as file:
        config_save = copy.deepcopy(dict(config))
        config_save["device"] = None
        yaml.dump(config_save, file)


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
                arg_type = str2bool if isinstance(value, bool) else type(value) # process bool type
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


# prep all dir
def setup_dir(config):
    print("> dataset_dir: ", os.path.abspath(config.dataset_dir))
    print("> dataset_path: ", os.path.abspath(os.path.join(config.dataset_dir, config.dataset_name)) + '.pt')

    if config.enable_task_name:
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

        task_name += f" H={config.fc_hidden_size}"

        config.output_dir = config.output_dir + '/' + task_name + '/'

    # remove the project dir if it exists
    if config.enable_rewrite_output_dir:
        if os.path.exists(config.output_dir):
            shutil.rmtree(config.output_dir)
            time.sleep(0.2)
            print("! output_dir removed:\t", os.path.abspath(config.output_dir))

    os.makedirs(config.output_dir, exist_ok=True)
    print("> output_dir:\t", os.path.abspath(config.output_dir))

    config.log_dir = config.output_dir + config.log_dir
    os.makedirs(config.log_dir, exist_ok=True)
    print("> log_dir:\t", os.path.abspath(config.log_dir))

    config.checkpoint_dir = config.output_dir + config.checkpoint_dir
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print("> checkpoint_dir: ", os.path.abspath(config.checkpoint_dir))


def get_model_name(directory_name):
    return directory_name.split(' ')[1]


def get_sample_num(directory_name):
    return int(directory_name.split(' ')[0].split('_')[-1])


def get_interval(directory_name):
    return int(directory_name.split(' ')[0].split('_')[-2])


# backup the config to output folder
def backup_config(config):
    output_file_path = os.path.join(config.output_dir, 'config.yaml')

    # Save the dictionary to a YAML file
    save_config(config, output_file_path)
    print(f"> config backup: {output_file_path}")


def test_load_config():
    print('test_load_config')
    config = load_config()
    pprint(config)


# calcualte mean abosolute precentage error (MAPE) from results file
def compute_MAPE(y_pred, y_true, mean, std):
    y_pred = y_pred * std + mean
    y_true = y_true * std + mean
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == '__main__':
    # test config loading
    parser = argparse.ArgumentParser()
    config = get_config_from_cmd(parser)
    pprint(config)
