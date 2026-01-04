import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# NOTE: This file is NEW.
# It adds an incremental/continual learning training entrypoint WITHOUT modifying train.py.
# It reuses the same model/data/util modules and copies the per-epoch training/eval logic
# from the existing train.py, but wraps it to run domain-by-domain and record NxN matrices.

from util import *
from data import *
from model import *

PRINT_INTERVAL = 4


# =============================================================================
# NEW (Incremental Learning): utility helpers
# =============================================================================
def _save_matrix_csv(path: str, matrix: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    """Save a matrix with row/col labels to CSV."""
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    df.to_csv(path, index=True)


def _ensure_unique_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    return list(dict.fromkeys(items))

def _reset_learning_rate(optimizer, base_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr

# =============================================================================
# NEW (Incremental Learning): wrappers around existing epoch logic
#   - The bodies are adapted from train.py Stage I/II/III with minimal changes:
#     - loader variables are passed in
#     - lengths are derived from the passed loader
#     - any file-wide variables (e.g., PRINT_INTERVAL) remain unchanged
# =============================================================================
def _train_one_epoch_on_loader(
    *,
    config,
    model,
    adaptor,
    criterion,
    metric,
    optimizer,
    scheduler,
    loader,
    epoch: int,
    enable_auto_regression: bool,
    n_seq_enc_look_back: int,
    n_seq_enc_total: int,
    progress_bar=None,
    train_print_step: int = 0,
):
    train_loss_sum = 0.0
    train_metric_sum = 0.0
    t_train_start = time.time()

    model.train()
    for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(loader):
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
            if 'label_additional_noise' in config and config.label_additional_noise != 0:
                noise_level = config.label_additional_noise
                noise = torch.randn_like(y_true) * (noise_level * y_true.abs())  # scale by each element's magnitude
                y_true = y_true + noise

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

        # print progress
        if i_loader % PRINT_INTERVAL == 0:
            train_print_step += 1
            if progress_bar is not None:
                progress_bar.set_description(
                    f"Ep {epoch:04d} | step {train_print_step:04d} | "
                    f"train_L={loss_temp.item():.4f} train_M={metric_temp.item():.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} |  "
                )

    train_loader_len = len(loader) if len(loader) > 0 else 1
    train_loss_mean = train_loss_sum / train_loader_len
    train_metric_mean = train_metric_sum / train_loader_len
    t_train_end = time.time()

    return train_loss_mean, train_metric_mean, train_print_step, (t_train_end - t_train_start)


@torch.no_grad()
def _eval_one_epoch_on_loader(
    *,
    config,
    model,
    adaptor,
    criterion,
    metric,
    loader,
    epoch: int,
    enable_auto_regression: bool,
    n_seq_enc_look_back: int,
    n_seq_enc_total: int,
):
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    t_start = time.time()

    model.eval()
    for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(loader):
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

    loader_len = len(loader) if len(loader) > 0 else 1
    loss_mean = val_loss_sum / loader_len
    metric_mean = val_metric_sum / loader_len
    t_end = time.time()

    return loss_mean, metric_mean, (t_end - t_start)


# =============================================================================
# NEW (Incremental Learning): main incremental training function
# =============================================================================
def train_incremental(config):
    """
    Incremental/continual learning training:
      - domains are semantic types: const/tooth/sin/square/noise
      - train sequentially domain-by-domain
      - 100 epochs per domain (config.num_epochs)
      - after each domain training, evaluate on ALL domains and record NxN matrices (loss + metric)
      - global mean/std computed once on all datasets (dataset_name_list_by_type)
      - optimizer/scheduler continuity preserved (no reset between domains)
    """

    # --- same setup behavior as train.py
    setup_local_config(config)

    if config.enable_seed:
        seed_everything(seed=config.seed)

    # -------------------------------------------------------------------------
    # NEW: build semantic domains from the manual list in data.py
    # -------------------------------------------------------------------------
        # In case of wildcard import behavior differences, fallback to direct name
        pass

    domains, domain_file = build_domains_from_dataset_names(dataset_name_list_by_type)
    print('> domains:\n', domains)

    # order (can be overridden from config)
    domain_order = getattr(config, "domain_order", ["const", "tooth", "sin", "square", "noise"])
    domain_order = [d for d in domain_order if d in domains]
    if len(domain_order) == 0:
        raise ValueError("No valid domains found. Check dataset_name_list_by_type and domain_order.")

    # -------------------------------------------------------------------------
    # NEW: global standardization once (then reuse config.param_mean/std, pos_mean/std)
    # -------------------------------------------------------------------------
    global_dataset = MyCombinedDataset(config, dataset_names=domain_file)
    if config.enable_standardize_feature:
        calculate_standardization(global_dataset, config)

    # per-domain loaders
    domain_loaders = {}
    for d in domain_order:
        domain_dataset = MyCombinedDataset(config, dataset_names=domains[d])

        # disabled for this is already standardized for all dataset:
        # if config.enable_standardize_feature:
        #     domain_dataset.apply_standardization(config)
        if config.enable_exclude_feature:
            domain_dataset.apply_exclusion(config)

        train_loader_d, val_loader_d, test_loader_d = get_dataloaders(domain_dataset, config)
        domain_loaders[d] = {"train": train_loader_d, "val": val_loader_d, "test": test_loader_d}

    # backup config
    save_config(config, os.path.join(config.machine_output_dir, 'config.yaml'))

    # model stack
    model, adaptor, criterion, metric, optimizer, scheduler = get_model(config)
    model_param_num = get_model_parameter_num(model)
    enable_auto_regression = (config.decoder_option == 'transformer')

    # sequence params
    n_seq_enc_look_back = config.n_seq_enc_look_back
    n_seq_enc_total = config.n_seq_enc_total

    # NEW controls
    num_phases = getattr(config, "num_phases", 1)
    epochs_per_domain = getattr(config, "num_epochs", 100)
    eval_splits = ["val", "test"]

    # optional tensorboard
    logger = None
    if config.enable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(config.machine_log_dir)

    # output dirs
    out_dir = config.machine_log_dir  # consistent with your existing logging location

    print("\n> Incremental Training Loop Starts...")
    t_start = time.time()

    N = len(domain_order)
    global_epoch = 0
    long_log = []  # phase, trained_domain, eval_domain, split, loss, metric, global_epoch

    for phase in range(num_phases):
        loss_mat = np.full((N, N), np.nan, dtype=np.float64)
        metric_mat = np.full((N, N), np.nan, dtype=np.float64)

        for i_trained, d_train in enumerate(domain_order):
            print(f"\n> Training starts for domain[{i_trained+1}/{len(domain_loaders)}]: ['{d_train}'], "
                  f"batch number={len(domain_loaders[d_train]['train'])}")
            # reset lr for the scheduler
            _reset_learning_rate(optimizer, config['lr'])

            # -------------------------
            # Train this domain for epochs_per_domain epochs
            # -------------------------
            progress_bar = tqdm(range(epochs_per_domain), ncols=110)
            train_print_step = 0
            for _ in progress_bar:
                train_loss_mean, train_metric_mean, train_print_step, train_dt = _train_one_epoch_on_loader(
                    config=config,
                    model=model,
                    adaptor=adaptor,
                    criterion=criterion,
                    metric=metric,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loader=domain_loaders[d_train]["train"],
                    epoch=global_epoch,
                    enable_auto_regression=enable_auto_regression,
                    n_seq_enc_look_back=n_seq_enc_look_back,
                    n_seq_enc_total=n_seq_enc_total,
                    progress_bar=progress_bar,
                    train_print_step=train_print_step,
                )

                # scheduler continuity (same rule as train.py)
                if config.enable_adaptive_lr and global_epoch < config.lr_adaptive_max_epoch:
                    scheduler.step()

                if logger is not None:
                    logger.add_scalars("train/loss_by_domain", {d_train: train_loss_mean}, global_step=global_epoch)
                    logger.add_scalars("train/metric_by_domain", {d_train: train_metric_mean}, global_step=global_epoch)

                global_epoch += 1

            # -------------------------
            # Evaluate on all domains and save NxN matrices
            # -------------------------
            print(f"\n> Evaluation starts for domain[{i_trained+1}]: '{d_train}', "
                  f"file len={len(domain_loaders[d_train]['train'])}")
            for split in eval_splits:
                for j_eval, d_eval in enumerate(domain_order):
                    loss_mean, metric_mean, eval_dt = _eval_one_epoch_on_loader(
                        config=config,
                        model=model,
                        adaptor=adaptor,
                        criterion=criterion,
                        metric=metric,
                        loader=domain_loaders[d_eval][split],
                        epoch=global_epoch,
                        enable_auto_regression=enable_auto_regression,
                        n_seq_enc_look_back=n_seq_enc_look_back,
                        n_seq_enc_total=n_seq_enc_total,
                    )

                    loss_mat[i_trained, j_eval] = loss_mean
                    metric_mat[i_trained, j_eval] = metric_mean
                    long_log.append([phase, d_train, d_eval, split, loss_mean, metric_mean, global_epoch])

                print(f">loss_matrix for {split}\n", loss_mat)
                print(f">metric_matrix for {split}\n", metric_mat)

                # save matrices after training domain d_train
                _save_matrix_csv(
                    os.path.join(out_dir, f"phase_{phase:02d}_after_{i_trained:02d}_{d_train}_{split}_loss_matrix.csv"),
                    loss_mat, domain_order, domain_order
                )
                _save_matrix_csv(
                    os.path.join(out_dir, f"phase_{phase:02d}_after_{i_trained:02d}_{d_train}_{split}_metric_matrix.csv"),
                    metric_mat, domain_order, domain_order
                )

    # save long-format log
    df_long = pd.DataFrame(long_log, columns=["phase", "trained_domain", "eval_domain", "split", "loss", "metric", "global_epoch"])
    df_long.to_csv(os.path.join(out_dir, "incremental_long_log.csv"), index=False)

    if logger is not None:
        logger.close()

    # cleanup similar to train.py
    model.to('cpu')
    for gpu_object in [model, adaptor, criterion, metric, optimizer, scheduler]:
        if gpu_object is not None:
            del gpu_object
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    epoch_stats = [global_epoch - 1, time.time() - t_start, None, "", "", "", "", "", "", ""]
    return model_param_num, epoch_stats


# =============================================================================
# NEW: __main__ for quick testing
#   Example:
#     python train_incremental.py --config config.yaml
#   or use your existing CLI args that get_config_from_cmd supports.
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = get_config_from_cmd(parser)
    train_incremental(config)