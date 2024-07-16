import os, time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from util import *
from model import get_model
from data import LightDataset, get_dataloaders

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
    config.enable_save_lstm_hidden = False

    # Set the random seed for reproducibility
    if config.enable_seed:
        seed_everything(seed=config.seed)

    # Load the test dataset
    dataset = LightDataset(config, data_mean=config.data_train_mean, data_std=config.data_train_std)
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

    # Validate the model
    validate(config, model, criterion, data_loader)


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
    config.enable_save_lstm_hidden = False

    # Set the random seed for reproducibility
    if config.enable_seed:
        seed_everything(seed=config.seed)

    # Load the test dataset
    dataset = LightDataset(config, data_mean=config.data_train_mean, data_std=config.data_train_std)
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
    validate(config, model, criterion, data_loader)


def validate(config, model, criterion, data_loader):
    # Set the model to evaluation mode
    model.eval()

    ''' Test Starts '''
    val_batch_size = 1
    val_loss = 0.0
    val_y_true_history = torch.zeros((val_batch_size, len(data_loader), config.output_size))
    val_y_original_history = torch.zeros_like(val_y_true_history)
    val_y_pred_history = torch.zeros_like(val_y_true_history)
    val_activation_history = torch.zeros((val_batch_size, len(data_loader),
                                          config.n_seq_total * config.fc_hidden_size))
    val_p_history = torch.zeros((val_batch_size, len(data_loader),
                                 config.n_seq_total * config.param_size))

    if config.model.startswith('CSAX') and config.enable_save_attention:
        val_attention_map_history = torch.zeros((val_batch_size, len(data_loader),
                                                 config.n_seq_total, config.n_seq_total))
        val_attention_gamma_hisotry = torch.zeros((val_batch_size, len(data_loader)))
        # val_attention_out_history = torch.zeros((val_batch_size, len(data_loader),
        #                                          config.n_seq_total, config.fc_hidden_size))
        # val_attention_final_out_history = torch.zeros_like(val_attention_out_history)

    if config.model.startswith('CBLX') and config.enable_save_lstm_hidden:
        val_lstm_hidden_history = torch.zeros((val_batch_size, len(data_loader),
                                               config.n_seq_total, config.fc_hidden_size))

    t_start = time.time()
    with torch.no_grad():
        for i, (x, y, p, y_original) in enumerate(tqdm(data_loader)):
            # x, y = x.to(config.device), y.to(config.device)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            val_loss += loss.cpu().item()

            val_y_true_history[:, i, :] = y.cpu().detach()
            val_y_pred_history[:, i, :] = y_pred.cpu().detach()
            val_p_history[:, i, :] = p.view(p.size()[0], -1).cpu().detach()
            val_y_original_history[:, i, :] = y_original.cpu().detach()

            if not config.model.startswith('CF1X') and config.enable_save_activation:
                val_activation_history[:, i, :] = model.final.activation.detach()

            if config.model.startswith('CSAX') and config.enable_save_attention:
                val_attention_map_history[:, i, :, :] = model.st_layer.attn_map.detach()
                # val_attention_out_history[:, i, :, :] = model.st_layer.attn_out.detach()
                # val_attention_final_out_history[:, i, :, :] = model.st_layer.final_out.detach()

                if config.model == 'CSAXG':
                    val_attention_gamma_hisotry[:, i] = model.st_layer.gamma.detach()

            if config.model.startswith('CBLX') and config.enable_save_lstm_hidden:
                val_lstm_hidden_history[:, i, :, :] = model.st_layer.hidden_data.detach()

    # metrics and temp results
    val_loss_mean = val_loss / len(data_loader)
    val_y_mse = [F.mse_loss(val_y_pred_history[:, :, i], val_y_true_history[:, :, i])
                 for i in range(config.output_size)]
    val_y_mse = np.array(val_y_mse)

    if config.enable_standardization:
        data_mean = config.data_train_mean
        data_std = config.data_train_std
    else:
        data_mean = [0, 0, 0]
        data_std = [1, 1, 1]

    [val_mape_h, val_mape_w, val_mape_a] = [compute_MAPE(val_y_pred_history[:, :, i], val_y_true_history[:, :, i],
                                                         config.data_train_mean[i], config.data_train_std[i])
                                            for i in range(config.output_size)]
    val_mape = (val_mape_h + val_mape_w + val_mape_a) / 3

    ''' stats on results '''
    best_model_stats = [-1, -1,
                        -1, -1, -1, -1,
                        val_loss_mean, val_y_mse[0], val_y_mse[1], val_y_mse[2],
                        val_mape, val_mape_h, val_mape_w, val_mape_a,
                        time.time() - t_start]

    ''' Save Results '''
    # save model activation
    torch.save(val_activation_history, os.path.join(config.output_dir, "best_model_activation_test.pth"))

    # save attention map
    if config.model.startswith('CSAX') and config.enable_save_attention:
        torch.save(val_attention_map_history, os.path.join(config.output_dir, "best_model_attn_map_test.pth"))
        if config.model == 'CSAXG':
            torch.save(val_attention_gamma_hisotry, os.path.join(config.output_dir, "best_model_attn_gamma_test.pth"))
        # torch.save(val_attention_out_history, os.path.join(config.output_dir, "best_model_attn_out_test.pth"))
        # torch.save(val_attention_final_out_history, os.path.join(config.output_dir, "best_model_attn_final_out_test.pth"))

    if config.model.startswith('CBLX') and config.enable_save_lstm_hidden:
        torch.save(val_lstm_hidden_history, os.path.join(config.output_dir, "best_model_lstm_hidden_test.pth"))

    # save stats
    column_header = ['epoch', 'lr',
                     'train_loss', 'train_mse_h', 'train_mse_w', 'train_mse_a',
                     'val_loss', 'val_mse_h', 'val_mse_w', 'val_mse_a',
                     'MAPE', 'MAPE_h', 'MAPE_w', 'MAPE_a',
                     'elapsed_t'
                     ]
    stats_df = pd.DataFrame(columns=column_header)
    stats_df.loc[0] = best_model_stats
    stats_df.to_csv(os.path.join(config.output_dir, "best_model_stats_test.csv"), index=False)

    # save results
    val_y_pred_history = val_y_pred_history.reshape(-1, config.output_size)
    val_y_true_history = val_y_true_history.reshape(-1, config.output_size)
    val_y_original_history = val_y_original_history.reshape(-1, config.output_size)

    val_y_pred_inverse_history = torch.zeros_like(val_y_pred_history)
    for i in range(config.output_size):
        val_y_pred_inverse_history[:, i] = val_y_pred_history[:, i] * config.data_train_std[i] + config.data_train_mean[i]

    val_y_true_inverse_history = torch.zeros_like(val_y_true_history)
    for i in range(config.output_size):
        val_y_true_inverse_history[:, i] = val_y_true_history[:, i] * config.data_train_std[i] + config.data_train_mean[i]

    val_y_ape_history = torch.abs((val_y_pred_inverse_history - val_y_true_inverse_history) / val_y_true_inverse_history * 100)
    val_y_ape_history_mean = torch.mean(val_y_ape_history, dim=1, keepdim=True)

    val_p_history = val_p_history.reshape(val_batch_size * len(data_loader), -1)

    val_results = torch.cat((val_y_pred_history, val_y_true_history,
                             val_y_pred_inverse_history, val_y_true_inverse_history,
                             val_y_original_history,
                             val_y_ape_history, val_y_ape_history_mean,
                             val_p_history,
                             ), dim=1)
    column_header = ['pred_h', 'pred_w', 'pred_a', 'true_h', 'true_w', 'true_a']
    column_header += ['pred_h_inverse', 'pred_w_inverse', 'pred_a_inverse',
                      'true_h_inverse', 'true_w_inverse', 'true_a_inverse']
    column_header += ['true_h_origin', 'true_w_origin', 'true_a_origin']
    column_header += ['ape_h', 'ape_w', 'ape_a', 'ape_all']
    column_header += [f"p{i//config.param_size}_{i%config.param_size}" for i in range(config.n_seq_total * config.param_size)]
    stats_df = pd.DataFrame(val_results.numpy(), columns=column_header)
    stats_df.to_csv(os.path.join(config.output_dir, "best_model_results_test.csv"), index=False)


if __name__ == '__main__':
    if DEPLOY_ON:  # deploy mode on
        # for model in ['CF1X', 'CF2X', 'CSAX', 'CBLX']:
        for model in ['CF1XG', 'CF2XG', 'CF1XGG', 'CF2XGG', 'CSAXG', 'CBLXG']:
            dataset_dir = r'C:\mydata\dataset\proj_melt_pool_pred\20240625_single_line'

            # model_dir = r'C:\mydata\output\proj_melt_pool_pred\test12_0'
            model_dir = r'C:\mydata\output\proj_melt_pool_pred\test13_1'
            model_name = f"clean_interval_24_300 {model} b=16 lr=5.0E-04adap drop=0.2 H=20"
            model_dir = os.path.join(model_dir, model_name)

            output_dir = r'C:\mydata\output\proj_melt_pool_pred\test13_9'
            output_dir = os.path.join(output_dir, model_name)
            delopy_trained_model(dataset_dir=dataset_dir, model_dir=model_dir, output_dir=output_dir)

    else:  # Test Mode On
        root_dir = r'C:\mydata\output\proj_melt_pool_pred\test14_null'
        # root_dir = r'C:\mydata\output\proj_melt_pool_pred\test14_0'
        root_dir = r'C:\mydata\output\proj_melt_pool_pred\test14_h50'
        # root_dir = r'C:\mydata\output\proj_melt_pool_pred\test14_h50_no_dropout'
        # root_dir = r'C:\mydata\output\proj_melt_pool_pred\test14_h50_no_norm'
        # root_dir = r'C:\mydata\output\proj_melt_pool_pred\test14_h50_no_gamma'

        # root_dir = r'C:\mydata\output\proj_melt_pool_pred\test14_comp_test'

        print(f"root_dir: {root_dir}")
        file_list = [entry.name for entry in os.scandir(root_dir)
                     if entry.is_dir()
                     and entry.name.startswith('clean')
                     # and get_interval(entry.name) == 24
                     # and get_sample_num(entry.name) == 300
                     # and get_model_name(entry.name).endswith('Null')
                     # and (entry.name.split(' ')[1] == 'CF1XGG' or entry.name.split(' ')[1] == 'CF2XGG')
                     ]
        print(f"> number of sub dir: {len(file_list)}")
        for i_file, file_name in enumerate(file_list):
            print(f"> file [{i_file}]/{len(file_list)}: {file_name}")
            test_trained_model(os.path.join(root_dir, file_name))
