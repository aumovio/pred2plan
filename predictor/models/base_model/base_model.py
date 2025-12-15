"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import json
import torch
import torchmetrics as tm
import pytorch_lightning as pl

from predictor.metrics.utils import build_metric_collection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_metrics: tm.MetricCollection = build_metric_collection(config.metrics.get("train", {}), "train")
        self.val_metrics:   tm.MetricCollection = build_metric_collection(config.metrics.get("val",   {}), "val")
        self.test_metrics:  tm.MetricCollection = build_metric_collection(config.metrics.get("test",  {}), "test")
        self.predict_metrics:  tm.MetricCollection = build_metric_collection(config.metrics.get("predict",  {}), "predict")

        self.uncertainty = self.config.model['uncertainty']
        self.flops_per_batch = None
        self.pred_dicts = []

    def reset_gp_precision_matrix(self):
        if self.uncertainty:
            self.prob_predictor.reset_precision_matrix()

    def _metric_kwargs(self, batch, out):
        # Map your batch/out to the metric input names you implemented
        batch = batch["input_dict"]
        log_std_range=(-1.609, 5.0)
        out["predicted_trajectory"][...,2] = torch.exp(torch.clip(out["predicted_trajectory"][..., 2] , min=log_std_range[0], max=log_std_range[1])) 
        out["predicted_trajectory"][...,3] = torch.exp(torch.clip(out["predicted_trajectory"][..., 3] , min=log_std_range[0], max=log_std_range[1])) 
        out["predicted_trajectory"][...,4] = torch.clip(out["predicted_trajectory"][..., 4], -0.5, 0.5)
        return dict(
            gt_traj=batch["center_gt_trajs"],       # [B,T,D]
            gt_mask=batch["center_gt_trajs_mask"],       # [B,T]
            pred_traj=out["predicted_trajectory"],     # [B,M,T,D]
            pred_prob=out["predicted_probability"],     # [B,M]
            trajs=out["predicted_trajectory"], # [B,M,T,D]
            probs=out["predicted_probability"],
            mask=torch.ones(batch["center_gt_trajs"].shape[0]).bool(),
            curr_vel = batch["center_gt_trajs_src"][:,self.config.data.past_len-1,7:8].norm(dim=-1),
            heading_idx = 5,
            # optionally:
            # gt_position_idx=slice(0,2),
            # pred_param_idx=slice(0,5),
            # curr_vel=batch.get("curr_vel"),
        )

    def forward(self, batch):
        """
        Forward pass for the model
        :param batch: input batch
        :return: prediction: {
                'predicted_probability': (batch_size,modes)),
                'predicted_trajectory': (batch_size,modes, modes, future_len, 5-7) # since we have GMMs
                }
                loss (with gradient)
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.log("train" + "_" + "loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=False)
        self.log("train" + "_" + "lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=False)
        self.train_metrics.update(**self._metric_kwargs(batch, prediction))
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.val_metrics.update(**self._metric_kwargs(batch, prediction))
        return loss
    
    def test_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.test_metrics.update(**self._metric_kwargs(batch, prediction))
        return loss
    
    def predict_step(self, batch, batch_idx):
        prediction, _ = self.forward(batch)
        return prediction
    
    def predict_and_evaluate(self, batch, batch_idx):
        prediction = self.predict_step(batch, batch_idx)
        metrics = self.compute_metrics_per_sample(batch, prediction, self.predict_metrics)
        return (prediction, metrics)

    @torch.no_grad()
    def compute_metrics_per_sample(self, batch, prediction, metrics):
        metrics_computed = {}
        kwargs = self._metric_kwargs(batch, prediction)
        B = kwargs["gt_traj"].shape[0]
        for metric_name in self.config.metrics.get("predict", {}):
            metric = metrics[metric_name]
            if hasattr(metric, "compute_per_sample_values"):
                metrics_computed[metric_name] = metric.compute_per_sample_values(**kwargs).cpu().numpy()
            else:
                logger.info(f"{metric_name} has no compute_per_sample_values() method for prediction. Filling with NaN.")
                metrics_computed[metric_name] = torch.full((B,), float('nan'))
        return metrics_computed

    # def predict_step(self, batch, batch_idx):
    #     prediction, loss = self.forward(batch)
    #     inputs = batch['input_dict']
    #     metrics = self.compute_custom_metrics(inputs, prediction)
    #     prediction.update(metrics)
    #     prediction["loss"] = loss
    #     return prediction

    def on_train_epoch_end(self):
        scores = self.train_metrics.compute()
        self.log_dict(scores, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        scores = self.val_metrics.compute()
        self.log_dict(scores, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        scores = self.test_metrics.compute()
        self.log_dict(scores, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        raise NotImplementedError


    

