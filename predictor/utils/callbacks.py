"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

from pathlib import Path
from omegaconf import DictConfig
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, ModelSummary, DeviceStatsMonitor, BasePredictionWriter  # Import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from hydra.utils import instantiate
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib
# matplotlib.use("Agg")
from utils.visualization import visualize_prediction


def get_callbacks(cfg: DictConfig, run_hash: str=None, output_path: Path=None) -> list:
    # mandatory callbacks
    checkpoint_last_callback = instantiate(cfg.callbacks.checkpoint_last) # ModelCheckpoint( # last epoch for resuming training
    #     filename='last_{epoch}-{val_brier_minFDE6:.2f}-{val_minNLL:.2f}',
    #     #save_last=True,
    # )
    checkpoint_best_callback = instantiate(cfg.callbacks.checkpoint_best) #ModelCheckpoint( # best epoch for evaluation
    #     monitor=f'val_{cfg.logging.checkpoint_metric}',  # Replace with your validation metric
    #     filename='best_{epoch}-{val_brier_minFDE6:.2f}-{val_minNLL:.2f}',
    #     save_top_k=1,
    #     mode='min',  # 'min' for loss/error, 'max' for accuracy
    # )
    learning_rate_monitor = LearningRateMonitor(
        # logging_interval="step",
        log_momentum=True,
        log_weight_decay=True,
    )
    reset_gp_callback = ResetPrecisionMatrixCallback()

    callbacks = [
        checkpoint_last_callback,
        checkpoint_best_callback, 
        learning_rate_monitor,
        reset_gp_callback,
        ModelSummary(max_depth=1),
    ]

    # optional callbacks
    if run_hash:
        prediction_writer = instantiate(cfg.callbacks.prediction_writer)
        # prediction_writer = PredictionWriter(
        #     output_path = Path(output_path) / Path(f"{run_hash}/predictions"),
        #     write_interval = 'batch',
        #     scenes_to_plot = cfg.trainer.logger.scenes_to_plot
        # )
        prediction_plotter = instantiate(cfg.callbacks.prediction_plotter)
        callbacks.append(prediction_writer)
        callbacks.append(prediction_plotter)
    if cfg.trainer.early_stopping:
        early_stopping_callback = instantiate(cfg.callbacks.early_stopping)
        callbacks.append(early_stopping_callback)
    if cfg.trainer.logger.monitor_device: callbacks.append(DeviceStatsMonitor(cpu_stats=True)) 
    return callbacks


def get_ep_callbacks():
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE_1', save_top_k=5, save_last = True, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    return [model_checkpoint, lr_monitor]



class MetricsPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_path: str,
        file_name: str,
        keys_from_input=[],
        keys_from_prediction=[],
    ):
        # write_interval="batch" ensures write_on_batch_end is called
        super().__init__(write_interval="batch")
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.file_name = file_name
        self.output_file = self.output_path / file_name

        self.keys_from_input = keys_from_input
        self.keys_from_prediction = keys_from_prediction

        self._rows: List[Dict[str, Any]] = []

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None):
        # Reset for each predict run
        if stage == "predict" or stage is None:
            self._rows = []


    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,            # this is what predict_step returns (prediction)
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """
        Called at the end of each predict_step.
        `outputs` == return value of pl_module.predict_step(batch, batch_idx)
        """
        prediction = outputs  # your predict_step returns just `prediction`
        B = batch["batch_size"]

        metrics_per_sample: Dict[str, torch.Tensor] = pl_module.compute_metrics_per_sample(
            batch=batch,
            prediction=prediction,
            metrics=pl_module.predict_metrics,
        )

        # Helper to safely convert tensor to Python object
        def _to_python(x: Any):
            if torch.is_tensor(x):
                x = x.detach().cpu()
                if x.ndim == 0:
                    return x.item()
                return x.tolist()
            return x

        # Build per-sample rows
        for i in range(B):
            row: Dict[str, Any] = {}

            # Selected input fields
            inputs = batch["input_dict"]
            for key in self.keys_from_input:
                if key not in inputs:
                    continue
                val = inputs[key]
                # Assume batch dimension is 0
                row[key] = _to_python(val[i])

            # Selected input fields
            for key in self.keys_from_prediction:
                if key not in prediction:
                    continue
                val = prediction[key]
                # Assume batch dimension is 0
                row[key] = _to_python(val[i])

            # Metrics
            for m_name, m_values in metrics_per_sample.items():
                row[m_name] = _to_python(m_values[i])

            self._rows.append(row)

    @rank_zero_only  # only global rank 0 writes the file in distributed setups
    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self._rows:
            return
        
        output_file = self.output_path / self.file_name
        df = pd.DataFrame(self._rows)
        df.to_parquet(output_file, index=False)
        logger.info(f"[PredictionWriter] Saved {len(df)} rows to {output_file}")


class StreamingMetricsPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_path: str,
        file_name: str,
        keys_from_input=[],
        keys_from_prediction=[],
    ):
        # write_interval="batch" ensures write_on_batch_end is called
        super().__init__(write_interval="batch")
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.file_name = file_name
        self.output_file = self.output_path / file_name

        self.keys_from_input = keys_from_input
        self.keys_from_prediction = keys_from_prediction

        # Streaming-related state
        self._parquet_writer: Optional[pq.ParquetWriter] = None
        self._schema: Optional[pa.Schema] = None

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,
    ):
        # Reset for each predict run
        if stage == "predict" or stage is None:
            # Close any previous writer just in case
            if self._parquet_writer is not None:
                self._parquet_writer.close()
            self._parquet_writer = None
            self._schema = None

            # Start fresh file for this predict run
            if self.output_file.exists():
                self.output_file.unlink()

    def _to_python(self, x: Any) -> Any:
        """Convert tensors to CPU Python types / lists; leave others as-is."""
        if torch.is_tensor(x):
            x = x.detach().cpu()
            if x.ndim == 0:
                return x.item()
            return x.tolist()
        return x

    def _write_batch_rows(self, rows: List[Dict[str, Any]]):
        """Convert a list of row dicts to a Table and stream-append to Parquet."""
        if not rows:
            return

        # DataFrame for this batch only
        df_batch = pd.DataFrame(rows)

        # Convert to Arrow table
        table = pa.Table.from_pandas(df_batch, preserve_index=False)

        # First batch: create writer with inferred schema
        if self._parquet_writer is None:
            self._schema = table.schema
            self._parquet_writer = pq.ParquetWriter(
                where=str(self.output_file),
                schema=self._schema,
            )
        else:
            # Ensure schema compatibility (allow upcasting / widening)
            if table.schema != self._schema:
                table = table.cast(self._schema, safe=False)

        # Append this batch as a new row group
        self._parquet_writer.write_table(table)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """
        Called at the end of each predict_step.
        `outputs` == return value of pl_module.predict_step(batch, batch_idx)
        """
        prediction = outputs  # your predict_step returns just `prediction`

        # How you get batch size – you had batch["batch_size"] before.
        # If that’s just an int, keep it. If not, adapt as needed.
        B = batch["batch_size"]

        metrics_per_sample: Dict[str, torch.Tensor] = pl_module.compute_metrics_per_sample(
            batch=batch,
            prediction=prediction,
            metrics=pl_module.predict_metrics,
        )

        batch_rows: List[Dict[str, Any]] = []

        for i in range(B):
            row: Dict[str, Any] = {}

            # Selected input fields from batch
            input = batch["input_dict"]
            for key in self.keys_from_input:
                if key not in input:
                    continue
                val = input[key]
                row[key] = self._to_python(val[i])

            # Selected fields from prediction
            for key in self.keys_from_prediction:
                if key not in prediction:
                    continue
                val = prediction[key]
                row[key] = self._to_python(val[i])

            # Metrics per sample
            for m_name, m_values in metrics_per_sample.items():
                row[m_name] = self._to_python(m_values[i])

            batch_rows.append(row)

        # Stream this batch straight to parquet
        self._write_batch_rows(batch_rows)

    @rank_zero_only  # only global rank 0 writes/closes the file in distributed setups
    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Close the writer to finalize the file
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
            logger.info(f"[MetricsPredictionWriter] Finished writing to {self.output_file}")


class PredictionPlotter(BasePredictionWriter):
    def __init__(
        self,
        output_path: str,
        file_name: str,
        n_samples: int,
        scenarios_to_plot: Optional[List[str]] = None,
    ):
        # write_interval="batch" ensures write_on_batch_end is called
        super().__init__(write_interval="batch")
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.file_name = file_name
        self.output_file = self.output_path / file_name

        if scenarios_to_plot:
            self.scenarios_to_plot = scenarios_to_plot
            self.n_samples = len(scenarios_to_plot)
        else:
            self.scenarios_to_plot = None
            self.n_samples = int(n_samples)

        self._n_saved = 0

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,
    ):
        # Reset for each predict run
        if stage == "predict" or stage is None:
            self._n_saved = 0

    @rank_zero_only  # only global rank 0 writes images in distributed setups
    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # Stop if we already have all requested plots
        if self._n_saved >= self.n_samples:
            return

        prediction = outputs  # your predict_step returns prediction

        # You used batch["batch_size"] before, keep that convention.
        # If you change your batch structure, adapt this line.
        B = batch["batch_size"]

        # For each sample in this batch, up to num_samples
        for i in range(B):
            if self._n_saved >= self.n_samples:
                break
            
            scenario_id = batch["input_dict"]["scenario_id"][i]
            dataset = batch["input_dict"]["dataset_name"][i]
            if self.scenarios_to_plot:
                if scenario_id in self.scenarios_to_plot:
                    self.plot_sample(batch, prediction, i, f"{dataset}_{scenario_id}")
            else:
                self.plot_sample(batch, prediction, i, f"{dataset}_{scenario_id}")

    def plot_sample(self, batch, prediction, idx, name):
        # Your provided visualization hook
        fig = visualize_prediction(batch, prediction, draw_index=idx)
        out_file = self.output_path / f"prediction_{name}.svg"
        
        # Save as SVG; assumes matplotlib Figure
        fig.savefig(out_file, format="svg", bbox_inches="tight")

        # Free the figure; use plt.close(fig) if you're using pyplot
        try:
            fig.clf()
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            # If it's not a matplotlib Figure or cleanup isn't needed, ignore
            pass

        self._n_saved += 1

    @rank_zero_only
    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info(
            f"[PredictionPlotter] Saved {self._n_saved} SVGs "
            f"to {self.output_path} (requested {self.n_samples})."
        )

class ResetPrecisionMatrixCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Call the reset function at the start of each epoch
        pl_module.reset_gp_precision_matrix()