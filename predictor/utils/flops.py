"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import numpy as np
import torch
from pathlib import Path
import hydra
import pandas as pd
from pytorch_lightning.utilities import measure_flops
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities import rank_zero_only


def measure_model_complexity(model, datamodule, config, run_hash):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    with torch.device("cuda"):
        model.to("cuda")
        datamodule.setup("predict")
        batch = next(iter(datamodule.predict_dataloader()))
    batch["input_dict"] = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in batch["input_dict"].items()}
    model_fwd = lambda: model(batch)
    fwd_flops_per_batch = measure_flops(model, model_fwd) * 1e-6
    fwd_flops_per_sample = fwd_flops_per_batch/batch["batch_size"]
    print(ModelSummary(model, max_depth=1))
    print("Forward MFLOPS per batch:", fwd_flops_per_batch ) # in MFLOP
    print("Forward MFLOPS per sample:", fwd_flops_per_sample ) # in MFLOP
    # save results
    csv_file = Path(config.paths.logs_path, "model_flops.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e-6
    new_row = {"ID": f"{config.model.model_name}_{run_hash}", 
        "forward_MFLOPS_batch": fwd_flops_per_batch , 
        "forward_MFLOPs_sample": fwd_flops_per_sample,
        "trainable_parameters": trainable_params,
        "model": config.model,
        "config": hydra_cfg["runtime"]["output_dir"]}
    save_to_csv(new_row, csv_file)

@rank_zero_only
def save_to_csv(new_row, csv_file):
    if csv_file.exists(): 
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(csv_file, index=False)
