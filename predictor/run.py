"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import sys 
import os

basepath = os.path.dirname(os.path.dirname(__file__))
if not basepath in sys.path:
    sys.path.insert(0, basepath)
    print(f"sys PATH now includes: '{basepath}'")

import pytorch_lightning as pl
import torch
from pathlib import Path

torch.set_float32_matmul_precision('medium')
from aim.pytorch_lightning import AimLogger
from aim.sdk.objects.artifact import Artifact
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profilers import PyTorchProfiler
from models import build_model
from datasets.datamodule import DataModule
from utils.seed import set_seed, find_latest_checkpoint
from utils.callbacks import get_callbacks, get_ep_callbacks
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="training")
def run(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    # OmegaConf.set_struct(cfg, False)  # Open the struct
    # cfg = OmegaConf.merge(cfg, cfg.paths)
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    predictor_cfg = cfg.predictor
    model = build_model(predictor_cfg)
    datamodule = DataModule(predictor_cfg)

    profiler = PyTorchProfiler(dirpath=cfg.paths.logs_path, filename="pytorch_profiling")

    # setup aim logger
    aim_logger = AimLogger(
        repo = f"{cfg.paths.logs_path}",
        experiment = predictor_cfg.trainer.experiment_name,
        run_name = predictor_cfg.trainer.logger.run_name,
        run_hash=predictor_cfg.trainer.logger.resume_run_hash,
        train_metric_prefix='train_',
        val_metric_prefix='val_',
        test_metric_prefix='test_',
        log_system_params=True if predictor_cfg.trainer.logger.monitor_device else False,
        capture_terminal_logs=predictor_cfg.trainer.logger.terminal_logs,
    )
    aim_logger.experiment.hash

    # track hyperparameters from config
    aim_logger.log_hyperparams(params=cfg)
    # setup artifact storage and save config as yaml artifact

    artifact_path = f"{hydra_cfg['runtime']['output_dir']}"
    artifact_uri = f"file://{artifact_path}"
    aim_logger.experiment.set_artifacts_uri(artifact_uri)
    aim_logger.experiment.log_artifact(f"{hydra_cfg['runtime']['output_dir']}/.hydra/config.yaml", name="run-config")

    # csv fallback logger
    csv_logger = CSVLogger(
        name=None,
        version=aim_logger.version,
        save_dir=f"{hydra_cfg['runtime']['output_dir']}",
    )

    trainer = pl.Trainer(
        default_root_dir=cfg.paths.logs_path,
        accumulate_grad_batches=predictor_cfg.trainer.accumulate_grad_batches,
        max_epochs=predictor_cfg.trainer.max_epochs,
        logger=[csv_logger, aim_logger,],#None if cfg.debug else [aim_logger, csv_logger], 
        log_every_n_steps=predictor_cfg.trainer.logger.log_every_n_steps,
        devices=1 if cfg.debug else predictor_cfg.trainer.devices,
        num_nodes=int(os.getenv("SLURM_NNODES",1)),
        gradient_clip_val=predictor_cfg.trainer.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler=profiler if predictor_cfg.trainer.logger.profiling=="pytorch" else predictor_cfg.trainer.logger.profiling,
        precision=predictor_cfg.trainer.precision,
        strategy="auto" if cfg.debug else "ddp", #ddp stands for "distributed data-parallel" training
        callbacks=get_callbacks(predictor_cfg, aim_logger._run_hash, artifact_path),
        check_val_every_n_epoch = predictor_cfg.trainer.val_check_interval,
    )

    if cfg.train:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=predictor_cfg.trainer.resume_ckpt_path)
        # track artifacts in aim repo
        ckpt_path = Path(f"{trainer.loggers[0].save_dir}/{str(trainer.loggers[0].name)}/{trainer.loggers[0].version}/checkpoints")
        for file_path in ckpt_path.rglob('*.ckpt'):
            # write small function that checks if checkpoints are lying on artifacts_uri path, if yes, then just track, if no, then upload
            artifact = Artifact(file_path, uri=f"{artifact_path}/{aim_logger._run_hash}/checkpoints")
            # artifact.upload(block=block)
            aim_logger.experiment.meta_run_tree.subtree('artifacts')[artifact.name] = artifact
            # aim_logger.experiment.log_artifact(file_path)
    ckpt_path = predictor_cfg.trainer.resume_ckpt_path if predictor_cfg.trainer.resume_ckpt_path else "best"

    if cfg.eval:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    if cfg.predict:
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path, return_predictions=False)
        # df = predictions_to_dataframe(predictions)
        # print(df)
        # breakpoint()
        # df_path = Path(f"{trainer.loggers[0].save_dir}/{str(trainer.loggers[0].name)}/{trainer.loggers[0].version}/df_predictions").with_suffix(cfg.data.compression_cache)
        # df.to_pickle(df_path)
        # artifact = Artifact(df_path, uri=f"{aim_logger.experiment.artifacts_uri}")
        # aim_logger.experiment.meta_run_tree.subtree('artifacts')[artifact.name] = artifact


if __name__ == '__main__':
    run()


