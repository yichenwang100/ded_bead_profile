import os, time

import yaml
from tqdm import tqdm
from util import *
from model import *
from data import *

# setup modes
ENABLE_SAVE_SALIENCY = True
ENABLE_SAVE_DETAILED_OUTPUT = True
TEST_BATCH_SIZE = 64


def testify(config, model, adaptor, criterion, metric, dataset, data_loader, test_mode='test', extra_name=''):
    # Set the model to evaluation mode
    if ENABLE_SAVE_SALIENCY:
        model.train()
    else:
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
    val_metric_sum = 0.0

    if ENABLE_SAVE_DETAILED_OUTPUT:
        val_y_loss = torch.zeros((len_data, 1))
        val_y_metric = torch.zeros((len_data, 1))

        val_raw_data_history = torch.zeros((len_data, dataset.data_tensor.shape[1]))  # add a mask column
        val_y_true_history = torch.zeros((len_data, config.label_size))  # batch size = 1
        val_y_pred_history = torch.zeros_like(val_y_true_history)

    enable_auto_regression = (config.decoder_option == 'transformer')

    if config.enable_save_attention:
        pass
        # feature_attn_sum = torch.zeros((config.embed_dim, config.embed_dim))
        # temporal_attn_sum = torch.zeros((config.n_seq_enc_total, config.n_seq_enc_total))

    t_start = time.time()
    progress_bar = tqdm(range(len(data_loader)), ncols=100)

    if ENABLE_SAVE_SALIENCY:
        torch.set_grad_enabled(True)  # same as with torch.no_grad():
    else:
        torch.set_grad_enabled(False)

    if ENABLE_SAVE_SALIENCY:
        saliency_map_img_hist = []
        saliency_map_param_hist = []
        saliency_map_pos_hist = []

    # auto-regression:
    if enable_auto_regression:
        y_pool = torch.zeros(config.batch_size, 1, config.output_size, device=config.device)

    for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(data_loader):
        # x, y = x.to(config.device), y.to(config.device)

        if ENABLE_SAVE_SALIENCY:
            x_img.requires_grad = True
            x_param.requires_grad = True
            x_pos.requires_grad = True

        i_dec = 0
        # Forward
        y_pred = model(x_img[:, i_dec:i_dec + n_seq_enc_total, :],
                       x_param[:, i_dec:i_dec + n_seq_enc_total, :],
                       x_pos[:, i_dec:i_dec + n_seq_enc_total, :],
                       y_pool if enable_auto_regression else None,
                       # reset_dec_hx=(i_dec == 0),
                       reset_dec_hx=True,
                       )

        if ENABLE_SAVE_SALIENCY:
            y_pred.sum().backward()

            saliency_map_img = x_img.grad.abs().squeeze(0).cpu().numpy()
            saliency_map_img_hist.append(saliency_map_img)

            saliency_map_param = x_param.grad.abs().squeeze(0).cpu().numpy()
            saliency_map_param_hist.append(saliency_map_param)

            saliency_map_pos = x_pos.grad.abs().squeeze(0).cpu().numpy()
            saliency_map_pos_hist.append(saliency_map_pos)

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

        metric_temp = metric(y_pred, y_true).cpu().item()
        val_metric_sum += metric_temp

        # Record
        if ENABLE_SAVE_DETAILED_OUTPUT:
            val_y_loss[i_loader, 0] = loss
            val_y_metric[i_loader, 0] = metric_temp

            val_y_true_history[i_record, :] = y_true.cpu().detach()
            val_y_pred_history[i_record, :] = y_pred.cpu().detach()

            val_raw_data_history[i_record] = dataset.get_raw_data(index, index_shift=i_dec).squeeze().cpu()

        i_record += 1

        if config.enable_save_attention:
            pass
            # feature_attn_sum += model.encoder.encoder[0].attn_map.cpu().detach().squeeze(0)
            # temporal_attn_sum += model.encoder.encoder[1].attn_map.cpu().detach().squeeze(0)

        progress_bar.set_description(f"Step [{i_loader}/{len(data_loader)}]"
                                     f", L:{val_loss_sum / (i_loader + 1):.4f}"
                                     f", metric:{val_metric_sum / (i_loader + 1):.3f}"
                                     f"\t>>>")
        progress_bar.update()

    # metrics and temp results
    val_loss_mean = val_loss_sum / len_loader
    val_metric_mean = val_metric_sum / len_loader

    if config.enable_save_attention:
        pass
        # feature_attn_mean = feature_attn_sum / len_loader
        # temporal_attn_mean = temporal_attn_sum / len_loader

    ''' Save Results '''
    ''' save stats fro best model '''
    column_header = ['elapsed_t', 'loss', 'metric', 'extra']
    test_stats = [time.time() - t_start, val_loss_mean, val_metric_mean, -1]

    stats_df = pd.DataFrame(columns=column_header)
    stats_df.loc[0] = test_stats
    stats_df.to_csv(os.path.join(config.machine_output_dir, f"best_model_stats.{test_mode}.{extra_name}.csv"),
                    index=False)

    ''' Save attention map '''
    if config.enable_save_attention:
        pass
        # torch.save(feature_attn_mean, os.path.join(config.output_dir, "feature_attn_mean.pth"))
        # torch.save(temporal_attn_mean, os.path.join(config.output_dir, "temporal_attn_mean.pth"))

    # save results
    if ENABLE_SAVE_DETAILED_OUTPUT:
        detailed_results = torch.cat((val_raw_data_history,
                                      val_y_pred_history, val_y_true_history,
                                      val_y_loss, val_y_metric), dim=1)
        torch.save(detailed_results,
                   os.path.join(config.machine_output_dir, f"best_model_results.{test_mode}.{extra_name}.pt"))

        column_header = []
        column_header += [f'RAW_{i + 1}' for i in range(dataset.data_tensor.shape[1])]
        column_header += [f"Y_PRED_{i + 1}" for i in range(config.label_size)]
        column_header += [f"Y_TRUE_{i + 1}" for i in range(config.label_size)]
        column_header += [f"LOSS"]
        column_header += [f"METRIC"]
        stats_df = pd.DataFrame(detailed_results.numpy(), columns=column_header)
        stats_df.to_csv(os.path.join(config.machine_output_dir, f"best_model_results.{test_mode}.{extra_name}.csv"),
                        index=False)

    if ENABLE_SAVE_SALIENCY:
        saliency_map_img_stack = np.stack(saliency_map_img_hist)
        saliency_map_param_stack = np.stack(saliency_map_param_hist)
        saliency_map_pos_stack = np.stack(saliency_map_pos_hist)

        saliency_map_img_mean = np.mean(saliency_map_img_stack, axis=0)
        saliency_map_param_mean = np.mean(saliency_map_param_stack, axis=0)
        saliency_map_pos_mean = np.mean(saliency_map_pos_stack, axis=0)

        os.makedirs(os.path.join(config.machine_output_dir, 'temp'), exist_ok=True)
        saliency_map_img_mean.save(
            os.path.join(config.machine_output_dir, 'temp', f'saliency_map_img.{test_mode}.{extra_name}.npy'))
        saliency_map_param_mean.save(
            os.path.join(config.machine_output_dir, 'temp', f'saliency_map_param.{test_mode}.{extra_name}.npy'))
        saliency_map_pos_mean.save(
            os.path.join(config.machine_output_dir, 'temp', f'saliency_map_pos.{test_mode}.{extra_name}.npy'))

        # extra info saving
        temp_header = ['dataset_len']
        temp_df = pd.DataFrame(columns=temp_header)
        temp_df.loc[0] = [len(saliency_map_img_hist)]

        temp_df.to_csv(
            os.path.join(config.machine_output_dir, 'temp', f'saliency_map_stats.{test_mode}.{extra_name}.csv'),
            index=False)


def deploy_trained_model(output_dir,
                         extra_name,
                         dataset_dir,
                         model_dir,
                         model_name,
                         use_all_dataset=True,
                         dataset_file_ratio=[0, 1],
                         self_reg=False):
    # setup local environment
    machine_config = load_attribute_dict('machine.yaml')
    machine_model_dir = os.path.join(machine_config.data_root_dir, model_dir, model_name)

    config = load_config(os.path.join(machine_model_dir, 'config.yaml'))

    config.output_dir = output_dir
    config.extra_name = extra_name
    config.dataset_dir = dataset_dir

    setup_local_config(config)

    save_config(config, os.path.join(config.machine_output_dir, 'config.yaml'))

    # copy model to target dir
    shutil.copy(src=os.path.join(machine_model_dir, 'best_model_stats.csv'), dst=config.machine_output_dir)
    shutil.copy(src=os.path.join(machine_model_dir, 'best_model_wts.pth'), dst=config.machine_output_dir)

    '''system override'''
    config.enable_deploy_dataset = True
    config.train_val_test_ratio = [0, 0, 1]
    config.batch_size = 1

    if self_reg:
        config.sys_sampling_interval = 1
        config.n_seq_dec_pool = 0

    '''temp override'''
    config.enable_save_attention = False

    ''' load model '''
    model, adaptor, criterion, metric, _, _ = get_model(config)

    # Load the best model weights
    best_model_path = os.path.join(config.machine_output_dir, 'best_model_wts.pth')
    # state_dict = torch.load(best_model_path, weights_only=True)
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     if not k.startswith('module.'):
    #         new_state_dict[f"module.{k}"] = v
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(torch.load(best_model_path))

    # Move the model to the appropriate device
    model = model.to(config.device)

    ''' Load the test dataset '''
    if use_all_dataset:
        file_list = [file for file in os.listdir(config.machine_dataset_dir) if file.endswith('.pt')]
    else:
        file_list = config.dataset_exclude_for_deploy

    file_num = int(len(file_list))
    file_list = file_list[int(file_num * dataset_file_ratio[0]):int(file_num * dataset_file_ratio[1])]

    for dataset_name in file_list:
        if config.enable_seed:
            seed_everything(seed=config.seed)

        config.dataset_name = dataset_name
        dataset = MyDataset(config)
        if config.enable_standardize_feature:
            dataset.apply_standardization(config)
        if config.enable_exclude_feature:
            dataset.apply_exclusion(config)
        _, _, test_loader = get_dataloaders(dataset, config, shuffle=False)

        # Test the model
        print(f"\n> Deploying on dataset: {dataset_name}")
        testify(config, model, adaptor, criterion, metric, dataset, test_loader,
                test_mode='deploy', extra_name=dataset_name)


if __name__ == '__main__':
    TEST_MODE = 'deploy'
    # TEST_MODE = 'test-Saliency'

    output_dir = './output/p2_ded_bead_profile/v13.1.d'
    extra_name = 'all_datasets'

    dataset_dir = './dataset/p2_ded_bead_profile/20240919'

    model_dir = './output/p2_ded_bead_profile/v13.1'
    model_name = f"241031-193730.9630.param_5.standardize.sample_1.enc_201_ah_100.label_40.b64.blstm_ffd.lr_0.4e-5_0.985.loss_008812"

    if TEST_MODE == 'deploy':  # deploy mode on
        deploy_trained_model(output_dir=output_dir,
                             extra_name=extra_name,
                             dataset_dir=dataset_dir,
                             model_dir=model_dir,
                             model_name=model_name,
                             use_all_dataset=True,
                             dataset_file_ratio=[0, 1],
                             self_reg=False)
    else:
        pass
