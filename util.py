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

    if config.enable_uuid_naming:
        # get a shortened UUID
        def get_uuid(length=-1):
            import uuid, base64
            # Generate a UUID
            uuid_obj = uuid.uuid4()

            # Convert UUID to bytes
            uuid_bytes = uuid_obj.bytes

            # Encode bytes to Base64
            short_uuid = base64.urlsafe_b64encode(uuid_bytes).rstrip(b'=').decode('utf-8')

            if length > 0:
                return short_uuid[0:length]

            else:
                return short_uuid

        task_name = f"{curr_time_str()}.{get_uuid(length=8)}"

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


def compute_ape(y_pred, y_true, eps=1e-6):
    ape = torch.abs((y_true - y_pred) / y_true) * 100
    ape[y_true.abs() < eps] = torch.nan
    ape[(y_pred.abs() < eps) & (y_true.abs() < eps)] = 1
    if torch.isnan(ape).all():
        return torch.nan
    return torch.nanmean(ape)


def compute_iou(y_pred, y_true, noise_cutoff=0):
    """
    Calculate an IoU-like metric for scalar values representing the intersection of the area
    under the predicted and true curves divided by the total area under the true curve.

    Parameters:
    y_true (torch.Tensor): Ground truth values, shape (batch_size, N).
    y_pred (torch.Tensor): Predicted values, shape (batch_size, N).

    Returns:
    float: IoU-like metric.
    """
    # Ensure the input tensors are of the correct shape
    assert y_pred.shape == y_true.shape

    # cutoff noise value
    y_pred_temp = y_pred.clone()
    y_true_temp = y_true.clone()
    y_pred_temp[y_pred_temp < noise_cutoff] = 0
    y_true_temp[y_true_temp < noise_cutoff] = 0

    # Calculate the total area under the y_true curve using the trapezoidal rule
    total_area_true = torch.trapz(y_true_temp.abs(), dim=1)

    # Calculate the intersection area
    min_values = torch.min(y_true_temp.abs(), y_pred_temp.abs())
    intersection_area = torch.trapz(min_values, dim=1)

    # Compute the IoU-like metric
    iou = intersection_area / total_area_true
    return iou

def time_to_HHMMSS(elapsed_time):
    hours, rem = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


from datetime import datetime
def curr_time_str():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as yymmdd-hhmmss
    formatted_date_time = now.strftime("%y%m%d-%H%M%S")

    return formatted_date_time


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
                print(f'{attr_name} has size {tensor_size/1e6} Mb')

            total_size += tensor_size

    if print_size:
        print(f"total size: {total_size/1e6} Mb")
    return total_size

def get_tensor_size(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.nelement()
    return 0

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
    "REAL_PROFILE1", "REAL_PROFILE2", "REAL_PROFILE3", "REAL_PROFILE4", "REAL_PROFILE5", "REAL_PROFILE6",
    "REAL_PROFILE7", "REAL_PROFILE8", "REAL_PROFILE9", "REAL_PROFILE10", "REAL_PROFILE11", "REAL_PROFILE12",
    "REAL_PROFILE13", "REAL_PROFILE14", "REAL_PROFILE15", "REAL_PROFILE16", "REAL_PROFILE17", "REAL_PROFILE18",
    "REAL_PROFILE19", "REAL_PROFILE20", "REAL_PROFILE21", "REAL_PROFILE22", "REAL_PROFILE23", "REAL_PROFILE24",
    "REAL_PROFILE25", "REAL_PROFILE26", "REAL_PROFILE27", "REAL_PROFILE28", "REAL_PROFILE29", "REAL_PROFILE30",
    "REAL_PROFILE31", "REAL_PROFILE32", "REAL_PROFILE33", "REAL_PROFILE34", "REAL_PROFILE35", "REAL_PROFILE36",
    "REAL_PROFILE37", "REAL_PROFILE38", "REAL_PROFILE39", "REAL_PROFILE40",
    "FIT_PROFILE1", "FIT_PROFILE2", "FIT_PROFILE3", "FIT_PROFILE4", "FIT_PROFILE5", "FIT_PROFILE6",
    "FIT_PROFILE7", "FIT_PROFILE8", "FIT_PROFILE9", "FIT_PROFILE10", "FIT_PROFILE11", "FIT_PROFILE12",
    "FIT_PROFILE13", "FIT_PROFILE14", "FIT_PROFILE15", "FIT_PROFILE16", "FIT_PROFILE17", "FIT_PROFILE18",
    "FIT_PROFILE19", "FIT_PROFILE20", "FIT_PROFILE21", "FIT_PROFILE22", "FIT_PROFILE23", "FIT_PROFILE24",
    "FIT_PROFILE25", "FIT_PROFILE26", "FIT_PROFILE27", "FIT_PROFILE28", "FIT_PROFILE29", "FIT_PROFILE30",
    "FIT_PROFILE31", "FIT_PROFILE32", "FIT_PROFILE33", "FIT_PROFILE34", "FIT_PROFILE35", "FIT_PROFILE36",
    "FIT_PROFILE37", "FIT_PROFILE38", "FIT_PROFILE39", "FIT_PROFILE40"
]

if __name__ == '__main__':
    # test config loading
    parser = argparse.ArgumentParser()
    config = get_config_from_cmd(parser)
    pprint(config)
