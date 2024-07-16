import torch, numpy as np, pandas as pd
import torch.nn.functional as F
import copy, time
from util import *
from data import *
from model import *
from tqdm import tqdm


def train(config):
    # set up dir
    setup_dir(config)

    # Set up random seed
    if config.enable_seed:
        seed_everything(seed=config.seed)

    # Set up the device
    config.device = torch.device("cuda" if (config.enable_gpu and torch.cuda.is_available()) else "cpu")

    # Prepare data
    dataset = LightDataset(config)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, config)

    # Set up mean and std
    if config.enable_standardization:
        config.data_train_mean = train_loader.dataset.dataset.data_mean.cpu().tolist()
        config.data_train_std = train_loader.dataset.dataset.data_std.cpu().tolist()

    # Set up model, loss function, optimizer, and scheduler for adaptive lr
    model, criterion, optimizer, scheduler = get_model(config)

    # Backup config to output dir
    backup_config(config)

    # Create logger using tensorboard
    if config.enable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(config.log_dir)

    # Create logger using pandas.dataframe and csv
    if config.enable_save_history_stats_to_csv:
        column_header = ['epoch', 'lr', 'train_loss', 'val_loss', 'elapsed_t', 'gamma']
        history_stats_df = pd.DataFrame(columns=column_header)

    # Training loop starts
    print('\n> Training Loop Starts...')
    num_epochs = config.num_epochs
    best_model_loss = float('inf')
    best_model_stats = None
    best_model_wts = None
    progress_bar = tqdm(range(num_epochs), ncols=100)
    t_start = time.time()
    for epoch in progress_bar:
        model.train()
        train_loss = 0.0
        for i, (x, y, _, _) in enumerate(train_loader):
            # x, y = x.to(config.device), y.to(config.device) # no need as this is done in the dataset init.

            # Forward
            y_pred = model(x)
            loss = criterion(y_pred, y)
            train_loss += loss.cpu().item()

            # Backward and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # metrics and temp results
        lr = optimizer.param_groups[0]['lr']
        train_loss_mean = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (x, y, _, _) in enumerate(val_loader):
                # x, y = x.to(config.device), y.to(config.device) # no need as this is done in the dataset init.
                y_pred = model(x)

                loss = criterion(y_pred, y)
                val_loss += loss.cpu().item()

        # metrics and temp results
        val_loss_mean = val_loss / len(val_loader)

        # Log
        if config.enable_tensorboard:
            logger.add_scalars(main_tag="loss",
                               tag_scalar_dict={'train': train_loss_mean,
                                                'val': val_loss_mean, },
                               global_step=epoch, )

            logger.add_scalars(main_tag="param",
                               tag_scalar_dict={'lr': lr},
                               global_step=epoch, )

        if config.enable_save_history_stats_to_csv:
            # ['epoch', 'lr', 'train_loss', 'val_loss', 'elapsed_t']
            epoch_stats = [epoch, lr,
                           train_loss_mean, val_loss_mean,
                           time.time() - t_start,
                           model.gamma.cpu().item() if
                           config.enable_residual_gamma and not config.model.endswith("Null") else 0]
            history_stats_df.loc[len(history_stats_df)] = epoch_stats

        if config.enable_save_best_model:
            if val_loss_mean < best_model_loss:
                best_model_loss = val_loss_mean
                best_model_stats = epoch_stats
                best_model_wts = copy.deepcopy(model.state_dict())

        # Print
        progress_bar.set_description(f"Ep [{epoch + 1}/{num_epochs}]"
                                     f"| L_train: {train_loss_mean:.4f}"
                                     f", L_val: {val_loss_mean:.4f}"
                                     f"\t|")

        # Adaptive learning rate
        if config.enable_adaptive_lr:
            scheduler.step()

    ''' Save Results '''
    # Close the SummaryWriter
    if config.enable_tensorboard:
        logger.close()

    # Write csv to file
    if config.enable_save_history_stats_to_csv:
        history_stats_df.to_csv(f"{config.log_dir}/train_val_stats.csv", index=False)

    # Save best model result and params
    if config.enable_save_best_model:
        epoch = best_model_stats[0]

        # save model weights and activation of last run
        torch.save(best_model_wts, f"{config.checkpoint_dir}/best_model_wts.pth")

        # save stats
        stats_df = pd.DataFrame(columns=column_header)
        stats_df.loc[0] = best_model_stats
        stats_df.to_csv(f"{config.checkpoint_dir}/best_model_stats.csv", index=False)


if __name__ == '__main__':
    config_raw = get_config_from_cmd(argparse.ArgumentParser())

    if config_raw.enable_iterate_dataset and config_raw.enable_iterate_model:
        raise RuntimeError("Err: enable_iterate_dataset and enable_iterate_model could not both be True")

    # iterate all dataset in the folder
    t_start = time.time()
    if config_raw.enable_light_dataset and config_raw.enable_iterate_dataset:
        if config_raw.enable_clean_data_only:
            file_list = [file for file in os.listdir(config_raw.dataset_dir)
                         if file.endswith('.pt')
                         # and get_interval(file) == 8
                         and file.startswith('clean')]
        else:
            file_list = [file for file in os.listdir(config_raw.dataset_dir)
                         if file.endswith('.pt')]

        file_list = sorted(file_list, key=lambda d: get_sample_num(d.split('.')[0]))

        for i_file, file_name in enumerate(file_list):
            config_train = copy.deepcopy(config_raw)
            file_name_base = file_name.split('.')[0]
            config_train.dataset_name = file_name_base
            config_train_dataset_interval = get_interval(file_name_base)
            config_train.dataset_sample_number = get_sample_num(file_name_base)
            if config_train.dataset_sample_number > 300:
                print("! skip training file: ", file_name)
                continue  # skip 400, 600


            def time_to_HHMMSS(elapsed_time):
                hours, rem = divmod(int(elapsed_time), 3600)
                minutes, seconds = divmod(rem, 60)
                return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


            time_elapsed = time.time() - t_start
            time_elapsed_mean = (time_elapsed / i_file) if i_file > 0 else 0
            time_remain = (time_elapsed / i_file * (len(file_list) - i_file)) if i_file > 0 else 0
            print(f"\n\n", '>' * 100)
            print(f"> Dataset: [{i_file}/{len(file_list)}]"
                  f", Elapsed: {time_to_HHMMSS(time_elapsed)}"
                  f", Mean: {time_to_HHMMSS(time_elapsed_mean)}"
                  f", Remain: {time_to_HHMMSS(time_remain)}")
            print(f'> dataset_name: ', config_train.dataset_name)
            train(config_train)

    # iterate all models
    elif config_raw.enable_iterate_model:
        model_list = ['CF1Null', 'CF2Null', 'CF1X', 'CF2X', 'CBLX', 'CSAX']
        for model in model_list:
            config_train = copy.deepcopy(config_raw)
            config_train.model = model
            print('\n> target model: ', config_train.model)
            train(config_train)

    elif config_raw.enable_computational_test:
        model_list = ['CF1X', 'CF2X', 'CSAX', 'CBLX']
        k_list = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        for model in model_list:
            for k in k_list:
                print("\n")
                print('>' * 50)
                config_train = copy.deepcopy(config_raw)
                config_train.fc_hidden_size = k
                config_train.model = model
                config_train.enable_tensorboard = False
                config_train.enable_save_best_model = False
                config_train.enable_save_activation = False
                config_train.enable_save_attention = False
                config_train.enable_save_lstm_hidden = False
                print('> fc_hidden_size: ', config_train.fc_hidden_size)
                print('> target model: ', config_train.model)
                train(config_train)

    else:
        print('> Training one dataset with user modified config')
        train(config_raw)
