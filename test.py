import os, time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from util import *
from model import *
from data import *


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
    model, adaptor, criterion, _, _ = get_model(config)

    # Load the best model weights
    best_model_path = os.path.join(file_dir, 'best_model_wts.pth')
    if not os.path.exists(best_model_path):
        print(f"! path {best_model_path} does not exist!")
        return
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # Move the model to the appropriate device
    model = model.to(config.device)

    # Set the random seed
    if config.enable_seed:
        seed_everything(seed=config.seed)

    # Test the model
    testify(config, model, criterion, dataset, data_loader, test_mode='test')


def deploy_trained_model(model_dir, output_dir, dataset_dir,
                         gpu_core_idx=0,
                         use_all_dataset=True,
                         dataset_ratio=[0, 0.5],
                         self_reg=False):
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
    config.dp_core_idx = gpu_core_idx
    config = config_setup_device(config)
    config.output_dir = output_dir
    config.dataset_dir = dataset_dir
    config.enable_deploy_dataset = True
    config.train_val_test_ratio = [0, 0, 1]

    '''temp override'''
    config.enable_save_attention = False

    # Set the random seed for reproducibility
    if config.enable_seed:
        seed_everything(seed=config.seed)

    if self_reg:
        config.sys_sampling_interval = 1
        config.batch_size = 1
        config.n_seq_dec_pool = 0

    # Load the test dataset
    if use_all_dataset:
        raw_file_list = [file for file in os.listdir(config.dataset_dir)
                        if file.endswith('.pt')]
        file_num = int(len(raw_file_list))
        file_list = raw_file_list[int(file_num * dataset_ratio[0]):int(file_num * dataset_ratio[1])]
    else:
        file_list = config.dataset_exclude

    for dataset_name in file_list:
        config.dataset_name = dataset_name
        dataset = MyDataset(config)
        _, _, test_loader = get_dataloaders(dataset, config, shuffle=False)

        # Initialize the model
        model, adaptor, criterion, _, _ = get_model(config)

        # Load the best model weights
        best_model_path = os.path.join(model_dir, 'best_model_wts.pth')
        if not os.path.exists(best_model_path):
            print(f"! path {best_model_path} does not exist!")
            return

        # process states dics for parallel/non-parallel models
        state_dict = torch.load(best_model_path, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                new_state_dict[f"module.{k}"] = v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(torch.load(best_model_path))

        # Move the model to the appropriate device
        model = model.to(config.device)

        # Test the model
        print(f"\n> Testing dataset: {dataset_name}")
        testify(config, model, adaptor, criterion, dataset, test_loader, test_mode='deploy', extra_name=dataset_name)


def testify(config, model, adaptor, criterion, dataset, data_loader, test_mode='test', extra_name=''):
    # Set the model to evaluation mode
    model.eval()

    ''' Test Starts '''
    batch_size = config.batch_size
    len_loader = len(data_loader)

    n_seq_enc_look_back = config.n_seq_enc_look_back
    # n_seq_enc_look_back = 200
    n_seq_enc_total = config.n_seq_enc_total

    len_data = len_loader * batch_size * 1  # n_seq_enc_look_back
    i_record = 0

    val_loss_sum = 0.0
    val_y_loss = torch.zeros((len_data, 1))

    val_iou_sum = 0.0
    val_y_iou = torch.zeros((len_data, 1))

    val_raw_data_history = torch.zeros((len_data, config.raw_data_size))  # add a mask column
    val_y_true_history = torch.zeros((len_data, config.label_size))  # batch size = 1
    val_y_pred_history = torch.zeros_like(val_y_true_history)
    y_noise_cutoff = config.label_noise_cutoff

    enable_auto_regression = config.decoder_option == 'transformer'

    # if config.enable_save_attention:
    #     feature_attn_sum = torch.zeros((config.embed_dim, config.embed_dim))
    #     temporal_attn_sum = torch.zeros((config.n_seq_enc_total, config.n_seq_enc_total))

    t_start = time.time()
    progress_bar = tqdm(range(len(data_loader)), ncols=100)
    with torch.no_grad():
        # auto-regression:
        if enable_auto_regression:
            y_pool = torch.zeros(config.batch_size, 1, config.output_size, device=config.device)

        for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(data_loader):
            # x, y = x.to(config.device), y.to(config.device)

            i_dec = 0
            # Forward
            y_pred = model(x_img[:, i_dec:i_dec + n_seq_enc_total, :],
                           x_param[:, i_dec:i_dec + n_seq_enc_total, :],
                           x_pos[:, i_dec:i_dec + n_seq_enc_total, :],
                           y_pool if enable_auto_regression else None,
                           # reset_dec_hx=(i_dec == 0),
                           reset_dec_hx=True,
                           )

            y_true = y[:, i_dec + n_seq_enc_look_back, :]

            # Shift elements left and insert the new prediction at the end
            if enable_auto_regression:
                if y_pool.size(1) <= n_seq_enc_look_back:
                    y_pool = torch.cat((y_pool, y_pred.unsqueeze(1)), dim=1).detach()
                else:
                    y_pool = torch.cat((y_pool[:, 1:, :], y_pred.unsqueeze(1)), dim=1).detach()
                # y_pool = y_pred.unsqueeze(1) # if keep lstm memory

            # Adaptor
            if 'enable_adaptor' in config and config.enable_adaptor:
                y_pred = adaptor.compute(y_pred)

            # Criterion
            loss = criterion(y_pred, y_true).cpu().item()
            val_loss_sum += loss
            val_y_loss[i_loader, 0] = loss

            iou = compute_iou(y_pred, y_true, y_noise_cutoff).mean().cpu().item()
            val_iou_sum += iou
            val_y_iou[i_loader, 0] = iou

            # Record
            val_y_true_history[i_record, :] = y_true.cpu().detach()
            val_y_pred_history[i_record, :] = y_pred.cpu().detach()

            raw_data = dataset.get_raw_data(index, index_shift=i_dec).squeeze()
            # get rid of the right most column for mask
            val_raw_data_history[i_record] = raw_data[:-1].cpu()

            i_record += 1

            if config.enable_save_attention:
                pass
                # feature_attn_sum += model.encoder.encoder[0].attn_map.cpu().detach().squeeze(0)
                # temporal_attn_sum += model.encoder.encoder[1].attn_map.cpu().detach().squeeze(0)

            progress_bar.set_description(f"Step [{i_loader}/{len(data_loader)}]"
                                         f", L:{val_loss_sum / (i_loader+1):.4f}"
                                         f", iou:{val_iou_sum / (i_loader+1):.3f}"
                                         f"\t|")
            progress_bar.update()

    # metrics and temp results
    val_loss_mean = val_loss_sum / len_loader
    val_iou_mean = val_iou_sum / len_loader

    # if config.enable_save_attention:
    #     feature_attn_mean = feature_attn_sum / len_loader
    #     temporal_attn_mean = temporal_attn_sum / len_loader

    ''' Save Results '''
    ''' save stats fro best model '''
    column_header = ['epoch', 'elapsed_t', 'lr',
                     'train_loss', 'val_loss', 'train_iou', 'val_iou',
                     'extra']
    test_stats = [-1, time.time() - t_start, -1,
                  -1, val_loss_mean, -1, val_iou_mean,
                  -1]

    stats_df = pd.DataFrame(columns=column_header)
    stats_df.loc[0] = test_stats
    stats_df.to_csv(os.path.join(config.output_dir, f"best_model_stats.{test_mode}.{extra_name}.csv"), index=False)

    ''' Save attention map '''
    # if config.enable_save_attention:
    #     torch.save(feature_attn_mean, os.path.join(config.output_dir, "feature_attn_mean.pth"))
    #     torch.save(temporal_attn_mean, os.path.join(config.output_dir, "temporal_attn_mean.pth"))

    # save results
    val_results = torch.cat((val_raw_data_history,
                             val_y_pred_history, val_y_true_history,
                             val_y_loss, val_y_iou), dim=1)
    column_header = excel_headers.copy()
    column_header += [f"Y_PRED_{i + 1}" for i in range(config.label_size)]
    column_header += [f"Y_TRUE_{i + 1}" for i in range(config.label_size)]
    column_header += [f"MSE"]
    column_header += [f"IOU"]
    stats_df = pd.DataFrame(val_results.numpy(), columns=column_header)
    stats_df.to_csv(os.path.join(config.output_dir, f"best_model_results.{test_mode}.{extra_name}.csv"), index=False)


if __name__ == '__main__':
    TEST_MODE = 'test'
    TEST_MODE = 'deploy'
    # TEST_MODE = 'test-Saliency'

    if TEST_MODE == 'test':  # Test on the test dataset
        root_dir = r'C:\mydata\output\p2_ded_bead_profile\v3.3'
        print(f"root_dir: {root_dir}")
        file_list = [entry.name for entry in os.scandir(root_dir)
                     if entry.is_dir()
                     and entry.name.startswith('240810-020242.95447320')
                     ]
        print(f"> number of sub dir: {len(file_list)}")
        for i_file, file_name in enumerate(file_list):
            print(f"\n> file [{i_file}]/{len(file_list)}: {file_name}")
            test_trained_model(os.path.join(root_dir, file_name))

    elif TEST_MODE == 'deploy':  # deploy mode on
        # model_dir = r'C:\mydata\output\proj_melt_pool_pred\test12_0'
        # model_dir = r'C:\mydata\output\p2_ded_bead_profile\v2.0'
        # model_name = f"240803-215833.28260170.ffd_ta.embed_default.no_gamma.ratio_1_no_noise_dataset.embed6.sampling_8.lr_1e-4adap0.96"
        model_dir = '/home/ubuntu/Desktop/mydata/output/p2_ded_bead_profile/v4.3'
        model_name = f"240820-000258.49774100.sample_200.enc_200.dec_100.pool_200.dec_lstm.schedule_ep_20.lr_1.2e-4_0.985.wd_1e-4.mix_loss"
        model_dir = os.path.join(model_dir, model_name)

        dataset_dir = '/home/ubuntu/Desktop/mydata/dataset/p2_ded_bead_profile/20240730'

        # output_dir = r'C:\mydata\output\p2_ded_bead_profile\v2.0.d'
        output_dir = '/home/ubuntu/Desktop/mydata/output/p2_ded_bead_profile/v4.3.d'
        output_dir = os.path.join(output_dir, model_name)

        deploy_trained_model(model_dir=model_dir,
                             output_dir=output_dir,
                             dataset_dir=dataset_dir,
                             gpu_core_idx=1,
                             use_all_dataset=True,
                             dataset_ratio=[0.5, 1],
                             self_reg=True)

    else:
        pass
