"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import numpy as np
from pathlib import Path
import hydra
import pandas as pd
import matplotlib.pyplot as plt

from predictor.utils.seed import set_seed
from pytorch_lightning.utilities import rank_zero_only


def multi_run_lr_find(tuner, model, datamodule, cfg, run_hash):
    #execute multiple runs of LR Finder
    lr_finders = []
    for i in range(cfg.tuning.find_lr.num_runs):
            try:
                set_seed(i+cfg.seed)
                lr_finders.append(tuner.lr_find(
                model = model,
                datamodule = datamodule,
                min_lr = cfg.tuning.find_lr.lr_range[0],
                max_lr = cfg.tuning.find_lr.lr_range[1],
                num_training = cfg.tuning.find_lr.num_batches,
                early_stop_threshold=cfg.tuning.find_lr.early_stop_multiplier,
                ))
            except:
                continue
    
    # aggregate results
    all_losses = []
    all_lrs = []
    all_min_lr_suggested = []
    all_max_lr_suggested = []
    all_max_lr = []
    for lr_find in lr_finders:
        all_losses.append(lr_find.results["loss"])
        all_lrs.append(lr_find.results["lr"])
        min_lr_suggested, max_lr_suggested, max_lr = select_lr(lr_find)
        all_min_lr_suggested.append(min_lr_suggested)
        all_max_lr_suggested.append(max_lr_suggested)
        all_max_lr.append(max_lr)

    # pad if necessary, stack into arrays
    n = len(all_lrs[0])
    padded_losses = []  # Create a new list to store padded losses
    for l in all_losses:
        pad_width = (0, n - len(l))
        padded_lo = np.pad(l, pad_width, mode='constant', constant_values=(0, l[-1]))
        padded_losses.append(padded_lo) 

    all_losses[:] = padded_losses
    all_losses= np.vstack(all_losses)
    all_min_lr_suggested = np.vstack(all_min_lr_suggested)
    all_max_lr_suggested = np.vstack(all_max_lr_suggested )
    all_max_lr = np.vstack(all_max_lr)

    if cfg.tuning.find_lr.plot:
        plot_lr_loss(all_lrs[0], 
                    all_losses, 
                    all_min_lr_suggested,
                    all_max_lr_suggested, 
                    all_max_lr, 
                    optimizer="SGD",
                    bs=cfg.training.batch_size,
                    data_folder=cfg.paths.logs_path,
                    prefix=f"{cfg.model.model_name}_{run_hash}_num_{cfg.tuning.find_lr.num_runs}"
                )
    print("Suggested Learning Rates:", (np.median(all_max_lr_suggested), np.median(all_max_lr)))
    save_lr_find_results(cfg, 
                        np.median(all_min_lr_suggested),
                        np.median(all_max_lr_suggested),
                        np.median(all_max_lr),
                        run_hash
                        )
    #plot_lr_loss(all_lrs[0], all_losses, all_min_lr_suggested, all_max_lr_suggested, optimizer="SGD",bs=cfg.training.batch_size,data_folder=cfg.paths.logs_path,epoch_ratio=20)
    return (np.median(all_max_lr_suggested), np.median(all_max_lr))

def select_lr(lr_find):
    losses = np.array(lr_find.results["loss"])
    losses = losses[np.isfinite(losses)]
    lrs = np.array(lr_find.results["lr"])
    min_loss_idx = np.argmin(losses)
    min_loss_lr_div_10 = lrs[min_loss_idx]/10
    max_lr = lrs[min_loss_idx]
    max_lr_suggested_idx = (np.abs(lrs - min_loss_lr_div_10)).argmin()
    max_lr_suggested = lrs[max_lr_suggested_idx]
    # get the learning rate at the steepest descent (minimum gradient)
    # between the point where the loss starts to decrease and the point
    # of minimum loss
    loss_grad = np.gradient(np.array(losses))
    reverse_losses = np.flip(loss_grad[:min_loss_idx])
    start_descent_idx = min_loss_idx - np.argmax(reverse_losses >= 0)
    min_lr_suggested_idx = start_descent_idx + (loss_grad[start_descent_idx:min_loss_idx]).argmin()
    min_lr_suggested = lrs[min_lr_suggested_idx]
    return min_lr_suggested, max_lr_suggested, max_lr

@rank_zero_only
def save_lr_find_results(cfg, min_lr_suggested, max_lr_suggested, max_lr, run_hash):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    csv_file = Path(cfg.paths.logs_path, "found_learning_rates.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    new_row = {"ID": f"{cfg.model.model_name}_{run_hash}", 
                "min_lr_suggested": min_lr_suggested, 
                "max_lr_suggested": max_lr_suggested,
                "max_lr": max_lr,
                "batch_size": cfg.training.batch_size,
                "weight_decay": cfg.training.weight_decay,
                "datasplits": cfg.data.train.splits,
                "uncertainty": cfg.model.uncertainty,
                "config": hydra_cfg["runtime"]["output_dir"]}
    if csv_file.exists(): 
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(csv_file, index=False)

def plot_lr_loss(lrs_res, all_losses_res, all_min_lr_res, suggested_lr_res, all_max_lr_res, 
                 optimizer="SGD",
                 bs=512, prefix='nuScenes',
                 data_folder=None,
                 ylim_top_ratio=0.95):

    losses_m = np.median(all_losses_res, axis=0)
    losses_lo = np.quantile(all_losses_res, q=0.1, axis=0)
    losses_hi = np.quantile(all_losses_res, q=0.9, axis=0)

    min_lr_m = np.median(all_min_lr_res, axis=0)[0]
    min_lr_lo = np.quantile(all_min_lr_res, q=0.1, axis=0)[0]
    min_lr_hi = np.quantile(all_min_lr_res, q=0.9, axis=0)[0]

    sug_lr_m = np.median(suggested_lr_res, axis=0)[0]
    sug_lr_lo = np.quantile(suggested_lr_res, q=0.1, axis=0)[0]
    sug_lr_hi = np.quantile(suggested_lr_res, q=0.9, axis=0)[0]

    max_lr_m = np.median(all_max_lr_res, axis=0)[0]
    max_lr_lo = np.quantile(all_max_lr_res, q=0.1, axis=0)[0]
    max_lr_hi = np.quantile(all_max_lr_res, q=0.9, axis=0)[0]

    plt.figure(figsize=(20, 10))
    plt.plot(lrs_res, losses_m)
    plt.grid()
    plt.fill_between(lrs_res, losses_lo, losses_hi, alpha=0.1, color="b")

    min_idx = (np.abs(lrs_res - min_lr_m)).argmin()
    sug_idx = (np.abs(lrs_res - sug_lr_m)).argmin()
    max_idx = (np.abs(lrs_res - max_lr_m)).argmin()
    min_lr_pt = (min_lr_m, losses_m[min_idx])
    sug_lr_pt = (sug_lr_m, losses_m[sug_idx])
    max_lr_pt = (max_lr_m, losses_m[max_idx])

    plt.plot(min_lr_m, losses_m[min_idx], markersize=10, marker='*', color='red')
    plt.plot(sug_lr_m, losses_m[sug_idx], markersize=10, marker='*', color='red')
    plt.plot(max_lr_m, losses_m[max_idx], markersize=10, marker='*', color='red')

    plt.annotate(xy=min_lr_pt, text=f'{min_lr_m:.2E}')
    plt.annotate(xy=sug_lr_pt, text=f'{sug_lr_m:.2E}')
    plt.annotate(xy=max_lr_pt, text=f'{max_lr_m:.2E}')

    plt.plot([min_lr_lo, min_lr_hi], [losses_m[min_idx], losses_m[min_idx]], color='red')
    plt.plot([sug_lr_lo, sug_lr_hi], [losses_m[sug_idx], losses_m[sug_idx]], color='red')
    plt.plot([max_lr_lo, max_lr_hi], [losses_m[max_idx], losses_m[max_idx]], color='red')

    plt.xlabel("Learning Rate", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.xscale('log')
    gap = losses_m[0] - np.min(losses_m)
    plt.ylim(bottom=np.min(losses_m) - 0.5 * gap, top=losses_m[0] + ylim_top_ratio * gap)

    plt.xlim(left=1e-7, right=100)

    plt.title(f"{prefix.title()} LR Range Test with {optimizer} and batch size {bs}", fontsize=20)

    if data_folder:
        filename = f"{prefix}_lrrt_{optimizer}_{bs}.png"
        plt.savefig(Path(data_folder, filename))