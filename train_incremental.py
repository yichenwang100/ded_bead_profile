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

PRINT_INTERVAL = 5


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


def _train_one_epoch_on_loader(
        *,
        config,
        model,
        criterion,
        metric,
        optimizer,
        loader,
        epoch: int,
        progress_bar=None,
):
    num_epochs = config.num_epochs
    n_seq_enc_look_back = config.n_seq_enc_look_back
    train_loss_sum = 0.0
    train_metric_sum = 0.0
    t_train_start = time.time()

    model.train()
    for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(loader):
        # x, y = x.to(config.device), y.to(config.device) # no need as this is done in the dataset init.

        # Forward
        y_pred = model(x_img, x_param, x_pos,
                       reset_dec_hx=True)

        # Criterion & Metrics
        y_true = y[:, n_seq_enc_look_back, :]

        loss_temp = criterion(y_pred, y_true)
        metric_temp = metric(y_pred, y_true)

        train_loss_sum += loss_temp.cpu().item()
        train_metric_sum += metric_temp.cpu().item()

        # Backward and Optimization
        optimizer.zero_grad()
        loss_temp.backward()
        optimizer.step()

        # print progress
        if i_loader % PRINT_INTERVAL == 0:
            if progress_bar is not None:
                progress_bar.set_description(
                    f"Ep {epoch + 1:03d}/{num_epochs}| "
                    f"Train: L={loss_temp.item():.4f} M={metric_temp.item():.4f} "
                    f"M_mean={train_metric_sum / (i_loader + 1):.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} |  "
                )

    train_loader_len = len(loader) if len(loader) > 0 else 1
    train_loss_mean = train_loss_sum / train_loader_len
    train_metric_mean = train_metric_sum / train_loader_len
    t_train_end = time.time()

    return train_loss_mean, train_metric_mean, (t_train_end - t_train_start)


@torch.no_grad()
def _eval_one_epoch_on_loader(
        *,
        config,
        model,
        criterion,
        metric,
        loader,
        n_seq_enc_look_back: int,
):
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    t_start = time.time()

    model.eval()
    for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(loader):
        # Forward
        y_pred = model(x_img, x_param, x_pos,
                       reset_dec_hx=True,
                       )

        # Criterion and metrics
        y_true = y[:, n_seq_enc_look_back, :]

        loss_temp = criterion(y_pred, y_true)
        metric_temp = metric(y_pred, y_true)

        val_loss_sum += loss_temp.cpu().item()
        val_metric_sum += metric_temp.cpu().item()

    loader_len = len(loader) if len(loader) > 0 else 1
    loss_mean = val_loss_sum / loader_len
    metric_mean = val_metric_sum / loader_len
    t_end = time.time()

    return loss_mean, metric_mean, (t_end - t_start)


# =============================================================================
# NEW (Incremental Learning): main incremental training function
# =============================================================================
def train_incremental_baseline(config):
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

    domain_order = config.domain_order
    domain_dict, domain_files = build_domain_dict(domain_list=domain_order,
                                                  file_dir=config.machine_dataset_dir)
    pprint(domain_dict)

    # global standardization once
    if config.enable_standardize_feature:
        global_dataset = MyCombinedDataset(config, file_list_override=domain_files)
        calculate_standardization(global_dataset, config)

    # per-domain loaders
    domain_loaders = {}
    for d in domain_order:
        domain_dataset = MyCombinedDataset(config, file_list_override=domain_dict[d],)

        if config.enable_exclude_feature:
            domain_dataset.apply_exclusion(config)

        train_loader_d, val_loader_d, test_loader_d = get_dataloaders(domain_dataset, config)
        domain_loaders[d] = {"train": train_loader_d, "val": val_loader_d, "test": test_loader_d}

    # backup config
    save_config(config, os.path.join(config.machine_output_dir, 'config.yaml'))

    # model stack
    model, adaptor, criterion, metric, optimizer, scheduler = get_model(config)

    # load hyper params
    n_seq_enc_look_back = config.n_seq_enc_look_back
    num_phases = config.num_phases
    epochs_per_domain = config.num_epochs

    # optional tensorboard
    logger = None
    if config.enable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(config.machine_log_dir)

    # output dirs
    out_dir = config.machine_log_dir  # consistent with your existing logging location

    print("\n> Incremental Training Loop Starts...")
    t_start = time.time()

    num_domain = len(domain_order)
    global_epoch = 0
    long_log = []  # phase, trained_domain, eval_domain, split, loss, metric, global_epoch

    for phase in range(num_phases):
        loss_mat = np.full((num_domain, num_domain), np.nan, dtype=np.float64)
        metric_mat = np.full((num_domain, num_domain), np.nan, dtype=np.float64)

        for i_trained, d_train in enumerate(domain_order):
            print(f"\n> Phase {phase + 1}/{num_phases} | "
                  f"Training starts for domain[{i_trained + 1}/{len(domain_loaders)}]: ['{d_train}'], "
                  f"batch number for train={len(domain_loaders[d_train]['train'])}")

            # reset lr for the scheduler
            _reset_learning_rate(optimizer, config['lr'])

            progress_bar = tqdm(range(epochs_per_domain), ncols=140)
            for _ in progress_bar:
                train_loss_mean, train_metric_mean, train_dt = _train_one_epoch_on_loader(
                    config=config,
                    model=model,
                    criterion=criterion,
                    metric=metric,
                    optimizer=optimizer,
                    loader=domain_loaders[d_train]["train"],
                    epoch=global_epoch,
                    progress_bar=progress_bar,
                )

                val_loss_mean, val_metric_mean, val_dt = _eval_one_epoch_on_loader(
                    config=config,
                    model=model,
                    criterion=criterion,
                    metric=metric,
                    loader=domain_loaders[d_train]['val'],
                    n_seq_enc_look_back=n_seq_enc_look_back,
                )

                # scheduler continuity (same rule as train.py)
                if config.enable_adaptive_lr:
                    scheduler.step()

                if logger is not None:
                    logger.add_scalars("train/loss_by_domain", {d_train: train_loss_mean}, global_step=global_epoch)
                    logger.add_scalars("train/metric_by_domain", {d_train: train_metric_mean}, global_step=global_epoch)
                    logger.add_scalars("val/loss_by_domain", {d_train: val_loss_mean}, global_step=global_epoch)
                    logger.add_scalars("val/metric_by_domain", {d_train: val_metric_mean}, global_step=global_epoch)

                global_epoch += 1

            # -------------------------
            # Evaluate on all domains and save NxN matrices
            # -------------------------
            print(f"\n> Evaluation starts for domain[{i_trained + 1}]: '{d_train}', "
                  f"file len={len(domain_loaders[d_train]['train'])}")

            for j_eval, d_eval in enumerate(domain_order):
                loss_mean, metric_mean, eval_dt = _eval_one_epoch_on_loader(
                    config=config,
                    model=model,
                    criterion=criterion,
                    metric=metric,
                    loader=domain_loaders[d_eval]['test'],
                    n_seq_enc_look_back=n_seq_enc_look_back,
                )

                loss_mat[i_trained, j_eval] = loss_mean
                metric_mat[i_trained, j_eval] = metric_mean
                long_log.append([phase, d_train, d_eval, loss_mean, metric_mean, global_epoch])

            print(f">metric_matrix for test set")
            pprint(metric_mat)

        # save matrices after training domain d_train
        _save_matrix_csv(
            path=os.path.join(out_dir, f"phase_{phase + 1:02d}_loss_matrix.csv"),
            matrix=loss_mat,
            row_labels=domain_order, col_labels=domain_order
        )
        _save_matrix_csv(
            path=os.path.join(out_dir,
                              f"phase_{phase + 1:02d}_metric_matrix.csv"),
            matrix=metric_mat,
            row_labels=domain_order, col_labels=domain_order
        )

    # save long-format log
    df_long = pd.DataFrame(long_log, columns=["phase", "trained_domain", "eval_domain", "loss", "metric",
                                              "global_epoch"])
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


# =============================================================================
# Replay-based incremental learning
# =============================================================================

from torch.utils.data import ConcatDataset, Subset


def _as_dataloader(dataset, config, *, shuffle: bool, drop_last: bool = True):
    """Create a DataLoader that matches get_dataloaders() defaults."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )


def _build_replay_mixed_train_dataset(
        *,
        new_train_dataset,
        replay_train_dataset,
        replay_ratio: float,
        seed: int,
):
    """
    Build a ConcatDataset where old data accounts for ~replay_ratio of total.
    If replay_train_dataset is None or empty, return new_train_dataset directly.

    replay_ratio = old / (old + new)
    => old_size = replay_ratio/(1-replay_ratio) * new_size
    """
    if replay_train_dataset is None:
        return new_train_dataset
    if len(replay_train_dataset) == 0:
        return new_train_dataset

    new_size = len(new_train_dataset)
    if new_size <= 0:
        return new_train_dataset

    replay_ratio = float(np.clip(replay_ratio, 0.0, 0.95))
    if replay_ratio <= 0:
        return new_train_dataset

    old_size = int(round((replay_ratio / max(1e-8, (1.0 - replay_ratio))) * new_size))
    old_size = min(old_size, len(replay_train_dataset))
    if old_size <= 0:
        return new_train_dataset

    g = torch.Generator()
    g.manual_seed(int(seed))

    # sample without replacement from replay_train_dataset
    perm = torch.randperm(len(replay_train_dataset), generator=g).tolist()
    sel = perm[:old_size]
    replay_subset = Subset(replay_train_dataset, sel)

    return ConcatDataset([new_train_dataset, replay_subset])


def train_incremental_replay(config):
    """
    Replay-based strategy:
      - at each domain step, train on (new_domain_train) + ~20% replay from OLD domains
      - replay is sampled from the TRAIN split of previously seen domains
      - after each domain training, evaluate on all domains (NxN matrices)
    """

    # --- same setup behavior as train.py
    setup_local_config(config)

    if config.enable_seed:
        seed_everything(seed=config.seed)

    domain_order = config.domain_order
    domain_dict, domain_files = build_domain_dict(domain_list=domain_order,
                                                  file_dir=config.machine_dataset_dir)
    pprint(domain_dict)

    # global standardization once
    if config.enable_standardize_feature:
        global_dataset = MyCombinedDataset(config, file_list_override=domain_files)
        calculate_standardization(global_dataset, config)

    # Build per-domain datasets (NOT loaders yet; we need train subsets for replay mixing)
    per_domain_datasets = {}
    for d in domain_order:
        domain_dataset = MyCombinedDataset(config, file_list_override=domain_dict[d],)
        if config.enable_exclude_feature:
            domain_dataset.apply_exclusion(config)

        train_dataset, val_dataset, test_dataset, _, _, _ = split_dataset(domain_dataset, config, shuffle=True)

        per_domain_datasets[d] = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    # backup config
    save_config(config, os.path.join(config.machine_output_dir, 'config.yaml'))

    model, adaptor, criterion, metric, optimizer, scheduler = get_model(config)
    n_seq_enc_look_back = config.n_seq_enc_look_back

    num_phases = config.num_phases
    epochs_per_domain = config.num_epochs

    replay_ratio = config.replay_ratio
    replay_seed_base = config.seed

    # tensorboard (optional)
    logger = None
    if config.enable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(config.machine_log_dir)

    out_dir = config.machine_log_dir
    num_domain = len(domain_order)
    global_epoch = 0
    long_log = []

    # replay memory: list of domains already learned
    seen_domains: list[str] = []

    for phase in range(num_phases):
        loss_mat = np.full((num_domain, num_domain), np.nan, dtype=np.float64)
        metric_mat = np.full((num_domain, num_domain), np.nan, dtype=np.float64)

        for i_trained, d_train in enumerate(domain_order):
            print(f"\n> [Replay] Phase {phase + 1}/{num_phases} | "
                  f"Training starts for domain[{i_trained + 1}/{num_domain}]: '{d_train}'")

            _reset_learning_rate(optimizer, config['lr'])

            # new domain splits
            new_train_ds = per_domain_datasets[d_train]["train"]
            val_loader = _as_dataloader(per_domain_datasets[d_train]["val"], config, shuffle=False)

            # build replay pool from previously seen domains (train split only)
            replay_train_ds = None
            if len(seen_domains) > 0:
                replay_sources = [per_domain_datasets[d_old]["train"] for d_old in seen_domains]
                replay_train_ds = ConcatDataset(replay_sources)

            mixed_train_ds = _build_replay_mixed_train_dataset(
                new_train_dataset=new_train_ds,
                replay_train_dataset=replay_train_ds,
                replay_ratio=replay_ratio,
                seed=replay_seed_base + phase * 1000 + i_trained,
            )

            train_loader = _as_dataloader(mixed_train_ds, config, shuffle=True)

            print(f"> [Replay] train sizes: new={len(new_train_ds)}, "
                  f"replay_pool={(len(replay_train_ds) if replay_train_ds is not None else 0)}, "
                  f"mixed={len(mixed_train_ds)} | replay_ratio≈{replay_ratio:.2f}")

            # train epochs
            progress_bar = tqdm(range(epochs_per_domain), ncols=140)
            for _ in progress_bar:
                train_loss_mean, train_metric_mean, _ = _train_one_epoch_on_loader(
                    config=config,
                    model=model,
                    criterion=criterion,
                    metric=metric,
                    optimizer=optimizer,
                    loader=train_loader,
                    epoch=global_epoch,
                    progress_bar=progress_bar,
                )

                val_loss_mean, val_metric_mean, _ = _eval_one_epoch_on_loader(
                    config=config,
                    model=model,
                    criterion=criterion,
                    metric=metric,
                    loader=val_loader,
                    n_seq_enc_look_back=n_seq_enc_look_back,
                )

                if config.enable_adaptive_lr:
                    scheduler.step()

                if logger is not None:
                    logger.add_scalars("train/loss_by_domain", {d_train: train_loss_mean},
                                       global_step=global_epoch)
                    logger.add_scalars("train/metric_by_domain", {d_train: train_metric_mean},
                                       global_step=global_epoch)
                    logger.add_scalars("val/loss_by_domain", {d_train: val_loss_mean},
                                       global_step=global_epoch)
                    logger.add_scalars("val/metric_by_domain", {d_train: val_metric_mean},
                                       global_step=global_epoch)

                global_epoch += 1

            # after training this domain, add it to replay memory
            if d_train not in seen_domains:
                seen_domains.append(d_train)

            # evaluate on all domains
            for j_eval, d_eval in enumerate(domain_order):
                test_loader = _as_dataloader(per_domain_datasets[d_eval]["test"], config, shuffle=False)
                loss_mean, metric_mean, _ = _eval_one_epoch_on_loader(
                    config=config,
                    model=model,
                    criterion=criterion,
                    metric=metric,
                    loader=test_loader,
                    n_seq_enc_look_back=n_seq_enc_look_back,
                )
                loss_mat[i_trained, j_eval] = loss_mean
                metric_mat[i_trained, j_eval] = metric_mean
                long_log.append([phase, d_train, d_eval, loss_mean, metric_mean, global_epoch])

            print(f"> [Replay] metric_matrix (test)")
            pprint(metric_mat)

        _save_matrix_csv(
            path=os.path.join(out_dir, f"phase_{phase + 1:02d}_loss_matrix.csv"),
            matrix=loss_mat,
            row_labels=domain_order, col_labels=domain_order
        )
        _save_matrix_csv(
            path=os.path.join(out_dir, f"phase_{phase + 1:02d}_metric_matrix.csv"),
            matrix=metric_mat,
            row_labels=domain_order, col_labels=domain_order
        )

    pd.DataFrame(long_log, columns=["phase", "trained_domain", "eval_domain", "loss", "metric", "global_epoch"]).to_csv(
        os.path.join(out_dir, "incremental_long_log.csv"), index=False
    )

    if logger is not None:
        logger.close()

    # cleanup
    model.to('cpu')
    for gpu_object in [model, adaptor, criterion, metric, optimizer, scheduler]:
        if gpu_object is not None:
            del gpu_object
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Regularization-based incremental learning (EWC)
# =============================================================================


def _ewc_snapshot_params(model) -> dict[str, torch.Tensor]:
    """Store a detached copy of current trainable parameters."""
    snap = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            snap[name] = p.detach().clone()
    return snap


@torch.no_grad()
def _ewc_zero_fisher_like(model) -> dict[str, torch.Tensor]:
    fisher = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            fisher[name] = torch.zeros_like(p, device=p.device)
    return fisher


def _ewc_compute_fisher(
        *,
        config,
        model,
        criterion,
        loader,
        n_seq_enc_look_back: int,
        max_batches: int | None = None,
):
    """
    Estimate diagonal Fisher via average squared gradients of loss.
    For regression (your setting), this works as a practical approximation.
    """
    model.eval()  # deterministic grads (dropout off)
    fisher = _ewc_zero_fisher_like(model)

    n_batches = 0
    for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(loader):
        if (max_batches is not None) and (i_loader >= max_batches):
            break

        y_pred = model(x_img, x_param, x_pos, reset_dec_hx=True)
        y_true = y[:, n_seq_enc_look_back, :]
        loss = criterion(y_pred, y_true)

        model.zero_grad(set_to_none=True)
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad and (p.grad is not None):
                fisher[name] += (p.grad.detach() ** 2)

        n_batches += 1

    if n_batches == 0:
        return fisher

    for k in fisher:
        fisher[k] /= float(n_batches)

    return fisher


def _ewc_penalty(model, fisher: dict[str, torch.Tensor], theta_star: dict[str, torch.Tensor]) -> torch.Tensor:
    """EWC quadratic penalty: sum_i F_i (theta_i - theta*_i)^2."""
    loss = torch.zeros((), device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if (name in fisher) and (name in theta_star) and p.requires_grad:
            loss = loss + (fisher[name] * (p - theta_star[name]) ** 2).sum()
    return loss


def train_incremental_regularization(config):
    """
    EWC-based strategy:
      - after finishing each domain, estimate Fisher on that domain's TRAIN split
      - during training on next domains, add EWC penalty against previous theta*
      - after each domain training, evaluate on all domains (NxN matrices)
    """

    # --- same setup behavior as train.py
    setup_local_config(config)

    if config.enable_seed:
        seed_everything(seed=config.seed)

    domain_order = config.domain_order
    domain_dict, domain_files = build_domain_dict(domain_list=domain_order,
                                                  file_dir=config.machine_dataset_dir)
    pprint(domain_dict)

    # global standardization once
    if config.enable_standardize_feature:
        global_dataset = MyCombinedDataset(config, file_list_override=domain_files)
        calculate_standardization(global_dataset, config)

    # per-domain loaders
    domain_loaders = {}
    for d in domain_order:
        domain_dataset = MyCombinedDataset(config, file_list_override=domain_dict[d],)

        if config.enable_exclude_feature:
            domain_dataset.apply_exclusion(config)

        train_loader_d, val_loader_d, test_loader_d = get_dataloaders(domain_dataset, config)
        domain_loaders[d] = {"train": train_loader_d, "val": val_loader_d, "test": test_loader_d}

    # backup config
    save_config(config, os.path.join(config.machine_output_dir, 'config.yaml'))

    model, adaptor, criterion, metric, optimizer, scheduler = get_model(config)
    n_seq_enc_look_back = config.n_seq_enc_look_back

    num_phases = config.num_phases
    epochs_per_domain = config.num_epochs

    ewc_lambda = config.ewc_lambda
    fisher_max_batches = config.ewc_fisher_max_batches

    logger = None
    if config.enable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(config.machine_log_dir)

    out_dir = config.machine_log_dir
    num_domain = len(domain_order)
    global_epoch = 0
    long_log = []

    # EWC state (consolidated across domains)
    theta_star = None
    fisher_star = None

    def _train_one_epoch_ewc(loader, epoch, progress_bar=None):
        model.train()
        train_loss_sum = 0.0
        train_metric_sum = 0.0

        for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(loader):
            y_pred = model(x_img, x_param, x_pos, reset_dec_hx=True)
            y_true = y[:, n_seq_enc_look_back, :]

            base_loss = criterion(y_pred, y_true)
            ewc_loss = torch.zeros_like(base_loss)

            if (theta_star is not None) and (fisher_star is not None) and (ewc_lambda > 0):
                ewc_loss = _ewc_penalty(model, fisher_star, theta_star) * ewc_lambda

            loss = base_loss + ewc_loss
            metric_temp = metric(y_pred, y_true)

            train_loss_sum += loss.detach().cpu().item()
            train_metric_sum += metric_temp.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_loader % PRINT_INTERVAL == 0 and progress_bar is not None:
                progress_bar.set_description(
                    f"Ep {epoch + 1:03d}/{epochs_per_domain}| "
                    f"Train: L={loss.item():.4f} (base={base_loss.item():.4f} ewc={ewc_loss.item():.4f}) "
                    f"M={metric_temp.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        loader_len = len(loader) if len(loader) > 0 else 1
        return train_loss_sum / loader_len, train_metric_sum / loader_len

    for phase in range(num_phases):
        loss_mat = np.full((num_domain, num_domain), np.nan, dtype=np.float64)
        metric_mat = np.full((num_domain, num_domain), np.nan, dtype=np.float64)

        for i_trained, d_train in enumerate(domain_order):
            print(f"\n> Phase {phase + 1}/{num_phases} | "
                  f"Training starts for domain[{i_trained + 1}/{len(domain_loaders)}]: ['{d_train}'], "
                  f"batch number for train={len(domain_loaders[d_train]['train'])}")

            # reset lr for the scheduler
            _reset_learning_rate(optimizer, config['lr'])

            progress_bar = tqdm(range(epochs_per_domain), ncols=140)
            for _ in progress_bar:
                train_loss_mean, train_metric_mean = _train_one_epoch_ewc(
                    loader=domain_loaders[d_train]["train"],
                    epoch=global_epoch,
                    progress_bar=progress_bar
                )

                val_loss_mean, val_metric_mean, _ = _eval_one_epoch_on_loader(
                    config=config,
                    model=model,
                    criterion=criterion,
                    metric=metric,
                    loader=domain_loaders[d_train]['val'],
                    n_seq_enc_look_back=n_seq_enc_look_back,
                )

                if config.enable_adaptive_lr:
                    scheduler.step()

                if logger is not None:
                    logger.add_scalars("train/loss_by_domain", {d_train: train_loss_mean}, global_step=global_epoch)
                    logger.add_scalars("train/metric_by_domain", {d_train: train_metric_mean},
                                       global_step=global_epoch)
                    logger.add_scalars("val/loss_by_domain", {d_train: val_loss_mean}, global_step=global_epoch)
                    logger.add_scalars("val/metric_by_domain", {d_train: val_metric_mean}, global_step=global_epoch)

                global_epoch += 1

            # After finishing this domain: update theta* and fisher*
            theta_d = _ewc_snapshot_params(model)
            fisher_d = _ewc_compute_fisher(
                config=config,
                model=model,
                criterion=criterion,
                loader=domain_loaders[d_train]["train"],
                n_seq_enc_look_back=n_seq_enc_look_back,
                max_batches=fisher_max_batches,
            )

            if theta_star is None:
                theta_star = theta_d
                fisher_star = fisher_d
            else:
                # consolidate: sum fishers (simple, common practice)
                for k in fisher_star:
                    fisher_star[k] = fisher_star[k] + fisher_d.get(k, 0.0)
                # keep latest theta* (compact variant)
                theta_star = theta_d

            # Evaluate on all domains
            for j_eval, d_eval in enumerate(domain_order):
                loss_mean, metric_mean, _ = _eval_one_epoch_on_loader(
                    config=config,
                    model=model,
                    criterion=criterion,
                    metric=metric,
                    loader=domain_loaders[d_eval]['test'],
                    n_seq_enc_look_back=n_seq_enc_look_back,
                )
                loss_mat[i_trained, j_eval] = loss_mean
                metric_mat[i_trained, j_eval] = metric_mean
                long_log.append([phase, d_train, d_eval, loss_mean, metric_mean, global_epoch])

            print(f">metric_matrix for test set")
            pprint(metric_mat)

        _save_matrix_csv(
            path=os.path.join(out_dir, f"phase_{phase + 1:02d}_loss_matrix.csv"),
            matrix=loss_mat,
            row_labels=domain_order, col_labels=domain_order
        )
        _save_matrix_csv(
            path=os.path.join(out_dir, f"phase_{phase + 1:02d}_metric_matrix.csv"),
            matrix=metric_mat,
            row_labels=domain_order, col_labels=domain_order
        )

    pd.DataFrame(long_log, columns=["phase", "trained_domain", "eval_domain", "loss", "metric", "global_epoch"]).to_csv(
        os.path.join(out_dir, "incremental_long_log.csv"), index=False
    )

    if logger is not None:
        logger.close()

    # cleanup
    model.to('cpu')
    for gpu_object in [model, adaptor, criterion, metric, optimizer, scheduler]:
        if gpu_object is not None:
            del gpu_object
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Distillation-based incremental learning
# =============================================================================

def _build_frozen_teacher(model) -> torch.nn.Module:
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _distill_loss_regression(student_pred: torch.Tensor, teacher_pred: torch.Tensor) -> torch.Tensor:
    """MSE distillation for regression outputs."""
    return F.mse_loss(student_pred, teacher_pred)


def train_incremental_distillation(config):
    """
    Distillation-based strategy (teacher-student):
      - before training each new domain, snapshot the current model as a frozen teacher
      - train student (same model) on new domain with:
          loss = base_criterion(student, y_true) + kd_lambda * MSE(student, teacher)
      - after each domain training, evaluate on all domains (NxN matrices)
    """

    # --- same setup behavior as train.py
    setup_local_config(config)

    if config.enable_seed:
        seed_everything(seed=config.seed)

    domain_order = config.domain_order
    domain_dict, domain_files = build_domain_dict(domain_list=domain_order,
                                                  file_dir=config.machine_dataset_dir)
    pprint(domain_dict)

    # global standardization once
    if config.enable_standardize_feature:
        global_dataset = MyCombinedDataset(config, file_list_override=domain_files)
        calculate_standardization(global_dataset, config)

    # per-domain loaders
    domain_loaders = {}
    for d in domain_order:
        domain_dataset = MyCombinedDataset(config, file_list_override=domain_dict[d],)

        if config.enable_exclude_feature:
            domain_dataset.apply_exclusion(config)

        train_loader_d, val_loader_d, test_loader_d = get_dataloaders(domain_dataset, config)
        domain_loaders[d] = {"train": train_loader_d, "val": val_loader_d, "test": test_loader_d}

    # backup config
    save_config(config, os.path.join(config.machine_output_dir, 'config.yaml'))

    model, adaptor, criterion, metric, optimizer, scheduler = get_model(config)
    n_seq_enc_look_back = config.n_seq_enc_look_back

    num_phases = config.num_phases
    epochs_per_domain = config.num_epochs

    kd_lambda = config.kd_lambda

    logger = None
    if config.enable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(config.machine_log_dir)

    out_dir = config.machine_log_dir
    N = len(domain_order)
    global_epoch = 0
    long_log = []

    teacher = None

    def _train_one_epoch_kd(loader, epoch, progress_bar=None):
        model.train()
        train_loss_sum = 0.0
        train_metric_sum = 0.0

        for i_loader, (index, x_img, x_param, x_pos, y) in enumerate(loader):
            y_pred = model(x_img, x_param, x_pos, reset_dec_hx=True)
            y_true = y[:, n_seq_enc_look_back, :]

            base_loss = criterion(y_pred, y_true)

            kd_loss = torch.zeros_like(base_loss)
            if (teacher is not None) and (kd_lambda > 0):
                with torch.no_grad():
                    y_t = teacher(x_img, x_param, x_pos, reset_dec_hx=True)
                kd_loss = _distill_loss_regression(y_pred, y_t) * kd_lambda

            loss = base_loss + kd_loss
            metric_temp = metric(y_pred, y_true)

            train_loss_sum += loss.detach().cpu().item()
            train_metric_sum += metric_temp.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_loader % PRINT_INTERVAL == 0 and progress_bar is not None:
                progress_bar.set_description(
                    f"Ep {epoch + 1:03d}/{epochs_per_domain}| "
                    f"Train: L={loss.item():.4f} (base={base_loss.item():.4f} kd={kd_loss.item():.4f}) "
                    f"M={metric_temp.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        loader_len = len(loader) if len(loader) > 0 else 1
        return train_loss_sum / loader_len, train_metric_sum / loader_len

    for phase in range(num_phases):
        loss_mat = np.full((N, N), np.nan, dtype=np.float64)
        metric_mat = np.full((N, N), np.nan, dtype=np.float64)

        for i_trained, d_train in enumerate(domain_order):
            print(f"\n> [KD] Phase {phase + 1}/{num_phases} | Training domain[{i_trained + 1}/{N}]: '{d_train}'")
            _reset_learning_rate(optimizer, config['lr'])

            # snapshot teacher BEFORE training this new domain (except first domain)
            if i_trained == 0 and phase == 0:
                teacher = None
            else:
                teacher = _build_frozen_teacher(model).to(config.device)

            progress_bar = tqdm(range(epochs_per_domain), ncols=140)
            for _ in progress_bar:
                train_loss_mean, train_metric_mean = _train_one_epoch_kd(
                    loader=domain_loaders[d_train]["train"],
                    epoch=global_epoch,
                    progress_bar=progress_bar,
                )

                val_loss_mean, val_metric_mean, _ = _eval_one_epoch_on_loader(
                    config=config,
                    model=model,
                    criterion=criterion,
                    metric=metric,
                    loader=domain_loaders[d_train]['val'],
                    n_seq_enc_look_back=n_seq_enc_look_back,
                )

                if config.enable_adaptive_lr:
                    scheduler.step()

                if logger is not None:
                    logger.add_scalars("train/loss_by_domain", {d_train: train_loss_mean}, global_step=global_epoch)
                    logger.add_scalars("train/metric_by_domain", {d_train: train_metric_mean},
                                       global_step=global_epoch)
                    logger.add_scalars("val/loss_by_domain", {d_train: val_loss_mean}, global_step=global_epoch)
                    logger.add_scalars("val/metric_by_domain", {d_train: val_metric_mean}, global_step=global_epoch)

                global_epoch += 1

            # Evaluate on all domains
            for j_eval, d_eval in enumerate(domain_order):
                loss_mean, metric_mean, _ = _eval_one_epoch_on_loader(
                    config=config,
                    model=model,
                    criterion=criterion,
                    metric=metric,
                    loader=domain_loaders[d_eval]['test'],
                    n_seq_enc_look_back=n_seq_enc_look_back,
                )
                loss_mat[i_trained, j_eval] = loss_mean
                metric_mat[i_trained, j_eval] = metric_mean
                long_log.append([phase, d_train, d_eval, loss_mean, metric_mean, global_epoch])

            print(f"> [KD] metric_matrix (test)")
            pprint(metric_mat)

        _save_matrix_csv(
            path=os.path.join(out_dir, f"phase_{phase + 1:02d}_loss_matrix.csv"),
            matrix=loss_mat,
            row_labels=domain_order, col_labels=domain_order
        )
        _save_matrix_csv(
            path=os.path.join(out_dir, f"phase_{phase + 1:02d}_metric_matrix.csv"),
            matrix=metric_mat,
            row_labels=domain_order, col_labels=domain_order
        )

    pd.DataFrame(long_log, columns=["phase", "trained_domain", "eval_domain", "loss", "metric", "global_epoch"]).to_csv(
        os.path.join(out_dir, "incremental_long_log.csv"), index=False
    )

    if logger is not None:
        logger.close()

    # cleanup
    model.to('cpu')
    for gpu_object in [model, adaptor, criterion, metric, optimizer, scheduler]:
        if gpu_object is not None:
            del gpu_object
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = get_config_from_cmd(parser)
    train_incremental_baseline(config)
    # train_incremental_replay(config)
    # train_incremental_regularization(config)
    # train_incremental_distillation(config)
