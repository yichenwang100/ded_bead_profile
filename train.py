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

    # Set up the device
    config.device = torch.device("cuda" if (config.enable_gpu and torch.cuda.is_available()) else "cpu")

    # Prepare data
    dataset = MyCombinedDataset(config) if config.enable_iterate_dataset else MyDataset(config)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, config)
    train_loader_len, val_loader_len = len(train_loader), len(val_loader)

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
        column_header = ['epoch', 'elapsed_t', 'lr', 'train_loss', 'val_loss', 'train_iou', 'val_iou', 'extra']
        history_stats_df = pd.DataFrame(columns=column_header)

    # Set up random seed
    if config.enable_seed:
        seed_everything(seed=config.seed)

    # Training loop starts
    print('\n> Training Loop Starts...')
    num_epochs = config.num_epochs
    n_seq_enc_look_back = config.n_seq_enc_look_back
    n_seq_enc_total = config.n_seq_enc_total

    best_model_metrics = 0
    best_model_stats = None
    best_model_wts = None
    y_noise_cutoff = config.output_noise_cutoff

    progress_bar = tqdm(range(num_epochs), ncols=110)
    train_print_step = 0
    val_print_step = 0
    t_start = time.time()
    for epoch in progress_bar:
        model.train()
        train_loss_sum = 0.0
        train_iou_sum = 0.0
        t_train_start = time.time()
        for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(train_loader):
            # x, y = x.to(config.device), y.to(config.device) # no need as this is done in the dataset init.

            # auto-regression:
            y_pool = torch.zeros(config.batch_size, 1, config.output_size).to(config.device)

            loss_temp_sum = 0
            iou_temp_sum = 0
            for i_dec in range(config.n_seq_dec_pool):
                # Forward
                y_pred = model(x_img[:, i_dec:i_dec+n_seq_enc_total, :],
                               x_param[:, i_dec:i_dec+n_seq_enc_total, :],
                               x_pos[:, i_dec:i_dec+n_seq_enc_total, :],
                               y_pool)

                y_true = y[:, i_dec+n_seq_enc_look_back, :]

                # scheduled sampling: use mixed labels of true and pred
                def is_using_true_label(i_epoch):
                    return (torch.rand(1) < (1 - i_epoch / config.scheduled_sampling_max_epoch)).item()

                if 'enable_scheduled_sampling' in config and config.enable_scheduled_sampling:
                    y_new = y_true if is_using_true_label(epoch) else y_pred
                else:
                    y_new = y_pred

                # Shift elements left and insert the new prediction at the end
                if y_pool.size(1) <= n_seq_enc_look_back:
                    y_pool = torch.cat((y_pool, y_new.unsqueeze(1)), dim=1).detach()
                else:
                    y_pool = torch.cat((y_pool[:, 1:, :], y_new.unsqueeze(1)), dim=1).detach()


                # Criterion
                loss = criterion(y_pred, y_true)
                loss_temp_sum += loss.cpu().item()
                iou = compute_iou(y_pred, y_true, y_noise_cutoff).mean()
                iou_temp_sum += iou.cpu().item()

                # Backward and Optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss_sum += loss_temp_sum / config.n_seq_dec_pool
            train_iou_sum += iou_temp_sum / config.n_seq_dec_pool

            # Print
            if i_loader % (train_loader_len // 4) == 0 or i_loader == train_loader_len-1:
                train_print_step += 1

                temp_train_loss = train_loss_sum / (i_loader + 1)
                temp_train_iou = train_iou_sum / (i_loader + 1)
                t_elapsed = (time.time() - t_train_start) / 60  # min
                progress_bar.set_description(f"Ep [{epoch + 1}/{num_epochs}]"
                                             f"| train {i_loader}/{train_loader_len}"
                                             f", L:{temp_train_loss:.4f}"
                                             f", iou:{temp_train_iou:.3f}"
                                             f", t:{t_elapsed:.1f}/{t_elapsed / (i_loader + 1) * train_loader_len:.1f}m"
                                             f"\t|")

                if config.enable_tensorboard:
                    logger.add_scalars(main_tag="step_loss",
                                       tag_scalar_dict={'train': temp_train_loss},
                                       global_step=train_print_step)
                    logger.add_scalars(main_tag="step_iou",
                                       tag_scalar_dict={'train': temp_train_iou},
                                       global_step=train_print_step)

        # metrics and temp results
        lr = optimizer.param_groups[0]['lr']
        train_loss_mean = train_loss_sum / train_loader_len
        train_iou_mean = train_iou_sum / train_loader_len

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        t_val_start = time.time()
        with torch.no_grad():

            for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(val_loader):

                # auto-regression:
                y_pool = torch.zeros(config.batch_size, 1 + n_seq_enc_look_back, config.output_size).to(config.device)
                loss_temp_sum = 0
                iou_temp_sum = 0
                for i_dec in range(config.n_seq_dec_pool):
                    # Forward
                    y_pred = model(x_img[:, i_dec:i_dec + n_seq_enc_total, :],
                                   x_param[:, i_dec:i_dec + n_seq_enc_total, :],
                                   x_pos[:, i_dec:i_dec + n_seq_enc_total, :],
                                   y_pool[:, :i_dec + 1, :])

                    # Shift elements left and insert the new prediction at the end
                    if y_pool.size(1) <= n_seq_enc_look_back:
                        y_pool[:, y_pool.size(1), :] = y_pred
                    else:
                        y_pool[:, :-1, :] = y_pool[:, 1:, :].clone()
                        y_pool[:, -1, :] = y_pred

                    # Criterion
                    y_true = y[:, i_dec+n_seq_enc_look_back, :]
                    loss = criterion(y_pred, y_true)
                    loss_temp_sum += loss.cpu().item()
                    iou = compute_iou(y_pred, y_true, y_noise_cutoff).mean()
                    iou_temp_sum += iou.cpu().item()

                val_loss_sum += loss_temp_sum / config.n_seq_dec_pool
                val_iou_sum += iou_temp_sum / config.n_seq_dec_pool

                # Print
                if i_loader % (val_loader_len // 4) == 0 or i_loader == val_loader_len-1:
                    val_print_step += 1

                    temp_val_loss = val_loss_sum / (i_loader + 1)
                    temp_val_iou = val_iou_sum / (i_loader + 1)
                    t_elapsed = (time.time() - t_val_start) / 60  # min
                    progress_bar.set_description(f"Ep [{epoch + 1}/{num_epochs}]"
                                                 f"| val {i_loader}/{val_loader_len}"
                                                 f", L:{temp_val_loss:.4f}"
                                                 f", iou:{temp_val_iou:.3f}"
                                                 f", t:{t_elapsed:.1f}/{t_elapsed / (i_loader + 1) * val_loader_len:.1f}m"
                                                 f"\t|")

                    if config.enable_tensorboard:
                        logger.add_scalars(main_tag="step_loss",
                                           tag_scalar_dict={'val': temp_val_loss},
                                           global_step=val_print_step)
                        logger.add_scalars(main_tag="step_iou",
                                           tag_scalar_dict={'val': temp_val_iou},
                                           global_step=val_print_step)

        # metrics and temp results
        val_loss_mean = val_loss_sum / val_loader_len
        val_iou_mean = val_iou_sum / val_loader_len

        # ['epoch', 'elapsed_t', 'lr', 'train_loss', 'val_loss', 'train_iou', 'val_iou', 'extra']
        extra_text = f""
        epoch_stats = [epoch, time.time() - t_start, lr,
                       train_loss_mean, val_loss_mean, train_iou_mean, val_iou_mean,
                       extra_text]

        if config.enable_tensorboard:
            logger.add_scalars(main_tag="epoch_loss",
                               tag_scalar_dict={'train': train_loss_mean,
                                                'val': val_loss_mean},
                               global_step=epoch)
            logger.add_scalars(main_tag="epoch_iou",
                               tag_scalar_dict={'train': train_iou_mean,
                                                'val': val_iou_mean},
                               global_step=epoch)

        # Log
        if config.enable_save_history_stats_to_csv:
            history_stats_df.loc[len(history_stats_df)] = epoch_stats

            if epoch % config.checkpoint_epoch_interval == 0 or epoch == num_epochs - 1:
                history_stats_df.to_csv(f"{config.log_dir}/train_val_stats.csv", index=False)

        if config.enable_save_best_model:
            if val_iou_mean > best_model_metrics:
                best_model_metrics = val_iou_mean
                best_model_wts = copy.deepcopy(model.state_dict())
                best_model_stats = epoch_stats

            if epoch % config.checkpoint_epoch_interval == 0 or epoch == num_epochs - 1:
                # save model weights
                torch.save(best_model_wts, f"{config.checkpoint_dir}/best_model_wts.pth")

                # save stats
                stats_df = pd.DataFrame(columns=column_header)
                stats_df.loc[0] = best_model_stats
                stats_df.to_csv(f"{config.checkpoint_dir}/best_model_stats.csv", index=False)

        # Adaptive learning rate
        if config.enable_adaptive_lr and epoch < config.lr_adaptive_max_epoch:
            scheduler.step()

    # Close the SummaryWriter
    if config.enable_tensorboard:
        logger.close()


if __name__ == '__main__':
    config_raw = get_config_from_cmd(argparse.ArgumentParser())

    if config_raw.enable_computational_test:
        model_list = ['CF1X', 'CF2X', 'CSAX', 'CBLX']
        k_list = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        for model in model_list:
            for k in k_list:
                print("\n")
                print('>' * 50)
                config_train = copy.deepcopy(config_raw)
                config_train.embed_dim = k
                config_train.model = model
                config_train.enable_tensorboard = False
                config_train.enable_save_best_model = False
                config_train.enable_save_attention = False
                print('> embed_dim: ', config_train.embed_dim)
                print('> target model: ', config_train.model)
                train(config_train)

    else:
        print('> Training one dataset with user modified config')
        train(config_raw)
