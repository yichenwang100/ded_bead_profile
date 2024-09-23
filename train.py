import torch, numpy as np, pandas as pd
import torch.nn.functional as F
import copy, time
from util import *
from data import *
from model import *
from tqdm import tqdm


def train(config):
    # set up dir
    setup_local_config(config)

    # Set up random seed
    if config.enable_seed:
        seed_everything(seed=config.seed)

    dataset = MyCombinedDataset(config) if config.enable_iterate_dataset else MyDataset(config)
    if config.enable_standardize_feature:
        calculate_standardization(dataset, config)
        dataset.apply_standardization(config)
    if config.enable_exclude_feature:
        dataset.apply_exclusion(config)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, config)
    train_loader_len, val_loader_len, test_loader_len = len(train_loader), len(val_loader), len(test_loader)

    # Backup config to output dir
    save_config(config, os.path.join(config.machine_output_dir, 'config.yaml'))

    # Set up model, loss function, etc.
    model, adaptor, criterion, metric, optimizer, scheduler = get_model(config)
    enable_auto_regression = (config.decoder_option == 'transformer')

    # Create logger using tensorboard
    if config.enable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(config.machine_log_dir)

    # Create logger using pandas.dataframe and csv
    history_stats_df = None

    # Training loop starts
    print('\n> Training Loop Starts...')
    num_epochs = config.num_epochs
    n_seq_enc_look_back = config.n_seq_enc_look_back
    n_seq_enc_total = config.n_seq_enc_total

    best_model_tested = False
    best_model_saved = False
    test_loss_mean = 0
    test_metric_mean = 0
    best_model_metrics = 0

    progress_bar = tqdm(range(num_epochs), ncols=110)
    train_print_step = 0
    val_print_step = 0
    t_start = time.time()
    for epoch in progress_bar:

        '''-------------------------------------------------------------------------------------------------'''
        ''' Stage I. Train '''
        '''-------------------------------------------------------------------------------------------------'''
        train_loss_sum = 0.0
        train_metric_sum = 0.0
        t_train_start = time.time()

        model.train()
        for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(train_loader):
            # x, y = x.to(config.device), y.to(config.device) # no need as this is done in the dataset init.

            # auto-regression:
            if enable_auto_regression:
                y_pool = torch.zeros(config.batch_size, 1, config.output_size, device=y.device)

            loss_temp_sum = 0
            metric_temp_sum = 0
            for i_dec in range(config.n_seq_dec_pool):
                # Forward
                y_pred = model(x_img[:, i_dec:i_dec + n_seq_enc_total, :],
                               x_param[:, i_dec:i_dec + n_seq_enc_total, :],
                               x_pos[:, i_dec:i_dec + n_seq_enc_total, :],
                               y_pool if enable_auto_regression else None,
                               reset_dec_hx=True)

                y_true = y[:, i_dec + n_seq_enc_look_back, :]

                # scheduled sampling: use mixed labels of true and pred
                if enable_auto_regression:
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

                # Adaptor
                if 'enable_adaptor' in config and config.enable_adaptor:
                    y_pred = adaptor.compute(y_pred)

                # Criterion & Metrics
                loss_temp = criterion(y_pred, y_true)
                loss_temp_sum += loss_temp.cpu().item()
                metric_temp = metric(y_pred, y_true)
                metric_temp_sum += metric_temp.cpu().item()

                # Backward and Optimization
                optimizer.zero_grad()
                loss_temp.backward()
                optimizer.step()

            train_loss_sum += loss_temp_sum / config.n_seq_dec_pool
            train_metric_sum += metric_temp_sum / config.n_seq_dec_pool

            # Print
            if i_loader % (train_loader_len // 4) == 0 or i_loader == train_loader_len - 1:
                train_print_step += 1

                temp_train_loss = train_loss_sum / (i_loader + 1)
                temp_train_metric = train_metric_sum / (i_loader + 1)
                t_elapsed = (time.time() - t_train_start) / 60  # min
                progress_bar.set_description(f"> train {i_loader}/{train_loader_len}"
                                             f", L:{temp_train_loss:.4f}"
                                             f", {config.metric_option}:{temp_train_metric:.3f}"
                                             f", t:{t_elapsed:.1f}/{t_elapsed / (i_loader + 1) * train_loader_len:.1f}m"
                                             f"\t>>>")

                if config.enable_tensorboard:
                    logger.add_scalars("step_loss", {'train': temp_train_loss},
                                       global_step=train_print_step)
                    logger.add_scalars("step_metric", {'train': temp_train_metric},
                                       global_step=train_print_step)

        # metrics and temp results
        lr = optimizer.param_groups[0]['lr']
        train_loss_mean = train_loss_sum / train_loader_len
        train_metric_mean = train_metric_sum / train_loader_len
        if config.enable_tensorboard:
            logger.add_scalars("epoch_loss", {'train': train_loss_mean}, global_step=epoch)
            logger.add_scalars("epoch_metric", {'train': train_metric_mean}, global_step=epoch)

        # Adaptive learning rate
        if config.enable_adaptive_lr and epoch < config.lr_adaptive_max_epoch:
            scheduler.step()

        '''-------------------------------------------------------------------------------------------------'''
        ''' Stage II. Val (Validation) '''
        '''-------------------------------------------------------------------------------------------------'''
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        t_val_start = time.time()

        model.eval()
        with torch.no_grad():
            for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(val_loader):

                # auto-regression:
                if enable_auto_regression:
                    y_pool = torch.zeros(config.batch_size, 1, config.output_size, device=y.device)

                loss_temp_sum = 0
                metric_temp_sum = 0
                for i_dec in range(config.n_seq_dec_pool):
                    # Forward
                    y_pred = model(x_img[:, i_dec:i_dec + n_seq_enc_total, :],
                                   x_param[:, i_dec:i_dec + n_seq_enc_total, :],
                                   x_pos[:, i_dec:i_dec + n_seq_enc_total, :],
                                   y_pool if enable_auto_regression else None,
                                   reset_dec_hx=True,
                                   )

                    if enable_auto_regression:
                        # Shift elements left and insert the new prediction at the end
                        if y_pool.size(1) <= n_seq_enc_look_back:
                            y_pool = torch.cat((y_pool, y_pred.unsqueeze(1)), dim=1).detach()
                        else:
                            y_pool = torch.cat((y_pool[:, 1:, :], y_pred.unsqueeze(1)), dim=1).detach()

                    # Adaptor
                    if 'enable_adaptor' in config and config.enable_adaptor:
                        y_pred = adaptor.compute(y_pred)

                    # Criterion and metrics
                    y_true = y[:, i_dec + n_seq_enc_look_back, :]
                    loss_temp = criterion(y_pred, y_true)
                    loss_temp_sum += loss_temp.cpu().item()
                    metric_temp = metric(y_pred, y_true)
                    metric_temp_sum += metric_temp.cpu().item()

                val_loss_sum += loss_temp_sum / config.n_seq_dec_pool
                val_metric_sum += metric_temp_sum / config.n_seq_dec_pool

                # Print
                if i_loader % (val_loader_len // 4) == 0 or i_loader == val_loader_len - 1:
                    val_print_step += 1

                    temp_val_loss = val_loss_sum / (i_loader + 1)
                    temp_val_metric = val_metric_sum / (i_loader + 1)
                    t_elapsed = (time.time() - t_val_start) / 60  # min
                    progress_bar.set_description(f"> val {i_loader}/{val_loader_len}"
                                                 f", L:{temp_val_loss:.4f}"
                                                 f", {config.metric_option}:{temp_val_metric:.3f}"
                                                 f", t:{t_elapsed:.1f}/{t_elapsed / (i_loader + 1) * val_loader_len:.1f}m"
                                                 f"\t>>>")

                    if config.enable_tensorboard:
                        logger.add_scalars("step_loss", {'val': temp_val_loss},
                                           global_step=val_print_step)
                        logger.add_scalars("step_metric", {'val': temp_val_metric},
                                           global_step=val_print_step)

        # metrics and temp results
        val_loss_mean = val_loss_sum / val_loader_len
        val_metric_mean = val_metric_sum / val_loader_len
        if config.enable_tensorboard:
            logger.add_scalars("epoch_loss", {'val': val_loss_mean}, global_step=epoch)
            logger.add_scalars("epoch_metric", {'val': val_metric_mean}, global_step=epoch)

        '''-------------------------------------------------------------------------------------------------'''
        ''' Decide Best Model / Early stop '''
        '''-------------------------------------------------------------------------------------------------'''
        # Best model
        if val_metric_mean > best_model_metrics:
            best_model_tested = False
            best_model_saved = False
            best_model_metrics = val_metric_mean

        # early stop
        # not implemented

        '''-------------------------------------------------------------------------------------------------'''
        ''' Stage III. Test '''
        '''-------------------------------------------------------------------------------------------------'''
        if best_model_tested:
            pass
        else:
            best_model_tested = True
            test_loss_sum = 0.0
            test_metric_sum = 0.0

            model.eval()
            with torch.no_grad():
                for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(test_loader):

                    # auto-regression:
                    if enable_auto_regression:
                        y_pool = torch.zeros(config.batch_size, 1, config.output_size, device=y.device)

                    loss_temp_sum = 0
                    metric_temp_sum = 0
                    for i_dec in range(config.n_seq_dec_pool):
                        # Forward
                        y_pred = model(x_img[:, i_dec:i_dec + n_seq_enc_total, :],
                                       x_param[:, i_dec:i_dec + n_seq_enc_total, :],
                                       x_pos[:, i_dec:i_dec + n_seq_enc_total, :],
                                       y_pool if enable_auto_regression else None,
                                       reset_dec_hx=True,
                                       )

                        if enable_auto_regression:
                            # Shift elements left and insert the new prediction at the end
                            if y_pool.size(1) <= n_seq_enc_look_back:
                                y_pool = torch.cat((y_pool, y_pred.unsqueeze(1)), dim=1).detach()
                            else:
                                y_pool = torch.cat((y_pool[:, 1:, :], y_pred.unsqueeze(1)), dim=1).detach()

                        # Adaptor
                        if 'enable_adaptor' in config and config.enable_adaptor:
                            y_pred = adaptor.compute(y_pred)

                        # Criterion
                        y_true = y[:, i_dec + n_seq_enc_look_back, :]
                        loss_temp = criterion(y_pred, y_true)
                        loss_temp_sum += loss_temp.cpu().item()
                        metric_temp = metric(y_pred, y_true)
                        metric_temp_sum += metric_temp.cpu().item()

                    test_loss_sum += loss_temp_sum / config.n_seq_dec_pool
                    test_metric_sum += metric_temp_sum / config.n_seq_dec_pool

            # metrics and temp results
            test_loss_mean = test_loss_sum / test_loader_len
            test_metric_mean = test_metric_sum / test_loader_len

        # log test results (every epoch)
        if config.enable_tensorboard:
            logger.add_scalars("epoch_loss", {'test': test_loss_mean}, global_step=epoch)
            logger.add_scalars("epoch_metric", {'test': test_metric_mean}, global_step=epoch)

        '''-------------------------------------------------------------------------------------------------'''
        ''' Log & Checkpoint '''
        '''-------------------------------------------------------------------------------------------------'''

        # stats
        # ['epoch', 'elapsed_t', 'lr',
        # 'train_loss', 'val_loss', 'test_loss', 'train_metric', 'val_metric', 'test_metric',
        # 'extra']
        extra_text = f""
        epoch_stats = [epoch, time.time() - t_start, lr,
                       train_loss_mean, val_loss_mean, test_loss_mean,
                       train_metric_mean, val_metric_mean, test_metric_mean,
                       extra_text]

        # Log saving
        stats_header = ['epoch', 'elapsed_t', 'lr',
                        'train_loss', 'val_loss', 'test_loss', 'train_metric', 'val_metric', 'test_metric',
                        'extra']
        if config.enable_save_history_stats_to_csv:
            if history_stats_df is None:
                history_stats_df = pd.DataFrame(columns=stats_header)
            history_stats_df.loc[len(history_stats_df)] = epoch_stats

            if epoch % config.checkpoint_epoch_interval == 0 or epoch == num_epochs - 1:
                history_stats_df.to_csv(f"{config.machine_log_dir}/train_val_stats.csv", index=False)

        # Checkpoint saving
        if config.enable_save_best_model and not best_model_saved:
            best_model_saved = True

            if epoch % config.checkpoint_epoch_interval == 0 or epoch == num_epochs - 1:
                # save model weights
                torch.save(model.state_dict(), f"{config.machine_checkpoint_dir}/best_model_wts.pth")

                # save stats
                stats_df = pd.DataFrame(columns=stats_header)
                stats_df.loc[0] = epoch_stats
                stats_df.to_csv(f"{config.machine_checkpoint_dir}/best_model_stats.csv", index=False)


if __name__ == '__main__':
    config_raw = get_config_from_cmd(argparse.ArgumentParser())

    # if config_raw.enable_computational_test:
    #     model_list = ['CF1X', 'CF2X', 'CSAX', 'CBLX']
    #     k_list = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #     for model in model_list:
    #         for k in k_list:
    #             print("\n")
    #             print('>' * 50)
    #             config_train = copy.deepcopy(config_raw)
    #             config_train.embed_dim = k
    #             config_train.model = model
    #             config_train.enable_tensorboard = False
    #             config_train.enable_save_best_model = False
    #             config_train.enable_save_attention = False
    #             print('> embed_dim: ', config_train.embed_dim)
    #             print('> target model: ', config_train.model)
    #             train(config_train)
    #
    # else:
    train(config_raw)
