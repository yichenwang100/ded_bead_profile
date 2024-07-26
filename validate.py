import os, time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from util import *
from model import *
from data import *

''' Dataset decomposition
Data
 - Development  
    - Train
    - Val
    - Test
 - Depoloyment
    - Deploy
 '''

DEPLOY_ON = False


# test trained models in the output dir based on the original dataset as defined in config.yaml
def test_trained_model(file_dir):
    # Load configuration
    config = load_config(os.path.join(file_dir, 'config.yaml'))

    '''temp override'''
    # config.enable_save_activation = True
    # config.enable_save_attention = True

    # Load the test dataset
    dataset = MyCombinedDataset(config) if config.enable_iterate_dataset else MyDataset(config)
    _, _, data_loader = get_dataloaders(dataset, config)

    # Initialize the model
    model, criterion, _, _ = get_model(config)

    # Load the best model weights
    best_model_path = os.path.join(file_dir, 'best_model_wts.pth')
    if not os.path.exists(best_model_path):
        print(f"! path {best_model_path} does not exist!")
        return
    model.load_state_dict(torch.load(best_model_path))

    # Move the model to the appropriate device
    model = model.to(config.device)

    # Set the random seed
    if config.enable_seed:
        seed_everything(seed=config.seed)

    # Validate the model
    validate(config, model, criterion, dataset, data_loader)


def delopy_trained_model(dataset_dir, model_dir, output_dir):
    import shutil
    def copy_folder(src, dst):
        try:
            shutil.copytree(src, dst)
            print(f"Folder '{src}' copied to '{dst}' successfully.")
        except Exception as e:
            print(f"Error: {e}")

    copy_folder(src=model_dir, dst=output_dir)

    # Load configuration
    config = load_config(os.path.join(output_dir, 'config.yaml'))
    config.dataset_dir = dataset_dir
    config.dataset_name = 'clean_interval_24_1000'
    config.enable_iterate_dataset = False
    config.output_dir = output_dir
    config.train_val_test_ratio = [0.001, 0.001, 0.998]

    '''temp override'''
    config.enable_save_activation = True
    config.enable_save_attention = True

    # Set the random seed for reproducibility
    if config.enable_seed:
        seed_everything(seed=config.seed)

    # Load the test dataset
    dataset = MyDataset(config, data_mean=config.data_train_mean, data_std=config.data_train_std)
    _, _, data_loader = get_dataloaders(dataset, config, shuffle=False)

    # Initialize the model
    model, criterion, _, _ = get_model(config)

    # Load the best model weights
    best_model_path = os.path.join(model_dir, 'best_model_wts.pth')
    if not os.path.exists(best_model_path):
        print(f"! path {best_model_path} does not exist!")
        return
    model.load_state_dict(torch.load(best_model_path))

    # Move the model to the appropriate device
    model = model.to(config.device)

    # Validate the model
    validate(config, model, criterion, dataset, data_loader)


def validate(config, model, criterion, dataset, data_loader):
    # Set the model to evaluation mode
    model.eval()

    ''' Test Starts '''
    val_loss = 0.0
    val_raw_data_history = torch.zeros((len(data_loader), config.raw_data_size))    # add a mask column
    val_y_true_history = torch.zeros((len(data_loader), config.output_size))    # batch size = 1
    val_y_pred_history = torch.zeros_like(val_y_true_history)
    val_y_iou = torch.zeros((len(data_loader), 1))
    output_noise_cutoff = config.output_noise_cutoff

    # val_activation_history = torch.zeros((val_batch_size, len(data_loader),
    #                                       config.n_seq_total * config.embed_dim))
    # val_p_history = torch.zeros((val_batch_size, len(data_loader),
    #                              config.n_seq_total * config.param_size))

    t_start = time.time()
    with torch.no_grad():
        for i, (index, x_img, x_param, x_pos, y) in enumerate(tqdm(data_loader)):
            # x, y = x.to(config.device), y.to(config.device)
            y_pred = model(x_img, x_param, x_pos)

            loss = criterion(y_pred, y)
            val_loss += loss.cpu().item()

            val_y_true_history[i, :] = y.cpu().detach().squeeze(0)
            val_y_pred_history[i, :] = y_pred.cpu().detach().squeeze(0)

            raw_data = dataset.get_raw_data(index.cpu().item()).cpu()
            val_raw_data_history[i, :] = raw_data[:-1] # get rid of the right most column for mask
            val_y_iou[i, 0] = compute_iou(y_pred, y, output_noise_cutoff).mean().cpu().item()

            # if config.enable_save_activation:
            #     val_activation_history[ i, :] = model.final.activation.detach()

            # if config.enable_save_attention:
            #     val_attention_map_history[i, :, :] = model.st_layer.attn_map.detach()
            #
            #     if config.model == 'CSAXG':
            #         val_attention_gamma_hisotry[i] = model.st_layer.gamma.detach()

    # metrics and temp results
    val_loss_mean = val_loss / len(data_loader)
    val_iou_mean = val_y_iou.mean()


    ''' Save Results '''
    ''' save stats fro best model '''
    column_header = ['epoch', 'elapsed_t', 'lr',
                     'train_loss', 'val_loss', 'train_iou', 'val_iou',
                     'extra']
    best_model_stats = [-1, time.time() - t_start, -1,
                        -1, val_loss_mean, -1, val_iou_mean,
                        -1]

    stats_df = pd.DataFrame(columns=column_header)
    stats_df.loc[0] = best_model_stats
    stats_df.to_csv(os.path.join(config.output_dir, "best_model_stats_test.csv"), index=False)

    # save model activation
    # torch.save(val_activation_history, os.path.join(config.output_dir, "best_model_activation_test.pth"))

    # save attention map
    # if config.enable_save_attention:
    #     torch.save(val_attention_map_history, os.path.join(config.output_dir, "best_model_attn_map_test.pth"))
    #     if config.model == 'CSAXG':
    #         torch.save(val_attention_gamma_hisotry, os.path.join(config.output_dir, "best_model_attn_gamma_test.pth"))

    # save results
    val_results = torch.cat((val_raw_data_history,
                             val_y_pred_history, val_y_true_history,
                             val_y_iou), dim=1)
    column_header = excel_headers
    column_header += [f"y_pred_{i+1}" for i in range(config.output_size)]
    column_header += [f"y_true_{i+1}" for i in range(config.output_size)]
    column_header += [f"y_iou"]
    stats_df = pd.DataFrame(val_results.numpy(), columns=column_header)
    stats_df.to_csv(os.path.join(config.output_dir, "best_model_results_test.csv"), index=False)


if __name__ == '__main__':
    if DEPLOY_ON:  # deploy mode on
        for model in ['STEN-GP']:
            dataset_dir = r'C:\mydata\dataset\proj_melt_pool_pred\20240625_single_line'

            # model_dir = r'C:\mydata\output\proj_melt_pool_pred\test12_0'
            model_dir = r'C:\mydata\output\proj_melt_pool_pred\test13_1'
            model_name = f"clean_interval_24_300 {model} b=16 lr=5.0E-04adap drop=0.2 H=20"
            model_dir = os.path.join(model_dir, model_name)

            output_dir = r'C:\mydata\output\proj_melt_pool_pred\test13_9'
            output_dir = os.path.join(output_dir, model_name)
            delopy_trained_model(dataset_dir=dataset_dir, model_dir=model_dir, output_dir=output_dir)

    else:  # Test Mode On
        root_dir = r'C:\mydata\output\p2_ded_bead_profile\v0.0'
        print(f"root_dir: {root_dir}")
        file_list = [entry.name for entry in os.scandir(root_dir)
                     if entry.is_dir()
                     and entry.name.endswith('new_mask')
                     ]
        print(f"> number of sub dir: {len(file_list)}")
        for i_file, file_name in enumerate(file_list):
            print(f"> file [{i_file}]/{len(file_list)}: {file_name}")
            test_trained_model(os.path.join(root_dir, file_name))
