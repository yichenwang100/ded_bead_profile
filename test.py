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


# test trained models in the output dir based on the original dataset as defined in config.yaml
def test_trained_model(file_dir):
    # Load configuration
    config = load_config(os.path.join(file_dir, 'config.yaml'))

    '''temp override'''
    config.enable_save_attention = True

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

    # Test the model
    testify(config, model, criterion, dataset, data_loader)


def deploy_trained_model(model_dir, output_dir):
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
    config.output_dir = output_dir
    config.enable_deploy_dataset = True
    config.train_val_test_ratio = [0.001, 0.001, 0.998]

    '''temp override'''
    config.enable_save_attention = False

    # Set the random seed for reproducibility
    if config.enable_seed:
        seed_everything(seed=config.seed)

    # Load the test dataset
    dataset = MyCombinedDataset(config) if config.enable_iterate_dataset else MyDataset(config)
    _, _, test_loader = get_dataloaders(dataset, config, shuffle=False)

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

    # Test the model
    testify(config, model, criterion, dataset, test_loader)


def testify(config, model, criterion, dataset, data_loader):
    # Set the model to evaluation mode
    model.eval()

    ''' Test Starts '''
    len_loader = len(data_loader)

    val_loss_sum = 0.0
    val_y_loss = torch.zeros((len_loader, 1))

    val_iou_sum = 0.0
    val_y_iou = torch.zeros((len_loader, 1))

    val_raw_data_history = torch.zeros((len_loader, config.raw_data_size))  # add a mask column
    val_y_true_history = torch.zeros((len_loader, config.output_size))  # batch size = 1
    val_y_pred_history = torch.zeros_like(val_y_true_history)
    output_noise_cutoff = config.output_noise_cutoff

    if config.enable_save_attention:
        feature_attn_sum = torch.zeros((config.embed_dim, config.embed_dim))
        temporal_attn_sum = torch.zeros((config.n_seq_total, config.n_seq_total))

    t_start = time.time()
    with torch.no_grad():
        for i, (index, x_img, x_param, x_pos, y) in enumerate(tqdm(data_loader)):
            # x, y = x.to(config.device), y.to(config.device)
            y_pred = model(x_img, x_param, x_pos)

            loss = criterion(y_pred, y).cpu().item()
            val_loss_sum += loss
            val_y_loss[i, 0] = loss

            val_iou = compute_iou(y_pred, y, output_noise_cutoff).mean().cpu().item()
            val_iou_sum += val_iou
            val_y_iou[i, 0] = val_iou

            val_y_true_history[i, :] = y.cpu().detach().squeeze(0)
            val_y_pred_history[i, :] = y_pred.cpu().detach().squeeze(0)

            raw_data = dataset.get_raw_data(index.cpu().item()).cpu()
            val_raw_data_history[i, :] = raw_data[:-1]  # get rid of the right most column for mask

            if config.enable_save_attention:
                # feature_attn_sum += model.encoder.encoder[0].attn_map.cpu().detach().squeeze(0)
                temporal_attn_sum += model.encoder.encoder[1].attn_map.cpu().detach().squeeze(0)

     # metrics and temp results
    val_loss_mean = val_loss_sum / len_loader
    val_iou_mean = val_iou_sum / len_loader

    if config.enable_save_attention:
        feature_attn_mean = feature_attn_sum / len_loader
        temporal_attn_mean = temporal_attn_sum / len_loader


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

    ''' Save attention map '''
    if config.enable_save_attention:
        torch.save(feature_attn_mean, os.path.join(config.output_dir, "feature_attn_mean.pth"))
        torch.save(temporal_attn_mean, os.path.join(config.output_dir, "temporal_attn_mean.pth"))

    # save results
    val_results = torch.cat((val_raw_data_history,
                             val_y_pred_history, val_y_true_history,
                             val_y_loss, val_y_iou), dim=1)
    column_header = excel_headers.copy()
    column_header += [f"Y_PRED_{i + 1}" for i in range(config.output_size)]
    column_header += [f"Y_TRUE_{i + 1}" for i in range(config.output_size)]
    column_header += [f"MSE"]
    column_header += [f"IOU"]
    stats_df = pd.DataFrame(val_results.numpy(), columns=column_header)
    stats_df.to_csv(os.path.join(config.output_dir, "best_model_results_test.csv"), index=False)


if __name__ == '__main__':
    # TEST_MODE = 'Test-Set'
    TEST_MODE = 'Deploy'
    # TEST_MODE = 'Test-Saliency'

    if TEST_MODE == 'Test-Set':     # Test on the test dataset
        root_dir = r'C:\mydata\output\p2_ded_bead_profile\v3.0'
        print(f"root_dir: {root_dir}")
        file_list = [entry.name for entry in os.scandir(root_dir)
                     if entry.is_dir()
                     and entry.name.startswith('240805-235557.18450500')
                     ]
        print(f"> number of sub dir: {len(file_list)}")
        for i_file, file_name in enumerate(file_list):
            print(f"\n> file [{i_file}]/{len(file_list)}: {file_name}")
            test_trained_model(os.path.join(root_dir, file_name))

    elif TEST_MODE == 'Deploy':  # deploy mode on
        # model_dir = r'C:\mydata\output\proj_melt_pool_pred\test12_0'
        # model_dir = r'C:\mydata\output\p2_ded_bead_profile\v2.0'
        # model_name = f"240803-215833.28260170.ffd_ta.embed_default.no_gamma.ratio_1_no_noise_dataset.embed6.sampling_8.lr_1e-4adap0.96"
        model_dir = r'C:\mydata\output\p2_ded_bead_profile\v3.2'
        model_name = f"240806-171800.17812270.ffd_ta.embed6.sampling_2.lr_1e-4adap0.96.loss_iou_0.1.dropout_0.3"
        model_dir = os.path.join(model_dir, model_name)

        # output_dir = r'C:\mydata\output\p2_ded_bead_profile\v2.0.d'
        output_dir = r'C:\mydata\output\p2_ded_bead_profile\v3.2.d'
        output_dir = os.path.join(output_dir, model_name)
        deploy_trained_model(model_dir=model_dir, output_dir=output_dir)

    else:
        pass


