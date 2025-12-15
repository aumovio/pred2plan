"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import torch
import torchmetrics as tm
from abc import ABC, abstractmethod
from typing import Optional, Union, Callable
### computing final valid idx ###

from predictor.metrics.utils import _extract_xy_coordinates, _normalize_weights, _get_denominator, _select_topk, _get_final_valid_idx, _resolve_timestep_as_mask, _resolve_final_timestep_as_batch

class Displacement(tm.Metric, ABC):
    """
    Abstract base for displacement metrics that output scalar per sample.
    Shared config:
      - timestep: max timestep to consider
      - k: top-K modes to consider (best among K)
      - order: || ||_order of the norm
    update() expects named args (so MetricCollection can forward kwargs):
      gt_traj:      [B, T, D]
      gt_mask:      [B, T]         (0/1)
      pred_traj:    [B, M, T, D]
      pred_prob:    [B, M]
    and optionally:
      position_idx: slice/list of idx for position dimensions in D (default slice(0,2))
      curr_vel:     [B] (default none -> no normalization)
      order: norm order for displacement (default 2)
      reduce_fn: Callable (default min = best of K)
    """

    def __init__(
        self,
        k: int = -1,
        t_max: int = -1,
        order: float = 2.0,
        eps: float = 1e-8,
        normalize: bool = False,
        reduce_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor] =
            lambda vals, _: vals.min(dim=1).values,  # default best-of-K
    ):
        super().__init__()
        self.k = int(k)
        self.t_max = int(t_max)
        self.order = float(order)
        self.reduce_fn = reduce_fn
        self.eps = float(eps)
        self.normalize = bool(normalize)

        self.add_state("sum_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0),   dist_reduce_fx="sum")


    # ---------- abstract hook and internal methods ----------

    @abstractmethod
    def _per_mode_value(
        self,
        diff: torch.Tensor,          # [B, K, T]
        gt_mask: torch.Tensor,       # [B, T]
        **kwargs,
    ) -> torch.Tensor:
        """Return per-mode scalar values [B, K] (e.g., ADE per mode, FDE per mode)."""
        raise NotImplementedError()

    def _compute_per_mode_values(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        pred_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ):
        B, M, T, D = pred_traj.shape

        # choose modes
        topk = _select_topk(self.k, pred_prob, M)                            # [B, K]
        K = topk.size(1)
        topk_w = _normalize_weights(pred_prob.gather(1, topk), self.eps)

        # gather predictions & probs
        ptraj = pred_traj.gather(1, topk[..., None, None].expand(B, K, T, D))  # [B, K, T, D]
        p = ptraj[..., pred_position_idx]                                      # [B, K, T, |position_idx|]
        g = gt_traj[..., gt_position_idx].unsqueeze(1)                        # [B, 1, T, |position_idx|]
        diff = torch.linalg.vector_norm(p - g, ord=self.order, dim=-1) # [B, K, T]

        per_mode = self._per_mode_value(
            diff=diff, gt_mask=gt_mask, **kwargs
        )  # [B, K]

        denom = _get_denominator(B, per_mode.device, per_mode.dtype, curr_vel, self.normalize).reshape(-1, 1)  # [B, 1]
        per_mode = per_mode / denom
        
        return per_mode, topk, topk_w

    # ---------- public API ----------
    @torch.no_grad()
    def compute_per_sample_values(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        pred_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ) -> torch.Tensor:
        per_mode, topk, topk_w = self._compute_per_mode_values(
            gt_traj,
            gt_mask,
            pred_traj,
            pred_prob,
            gt_position_idx,
            pred_position_idx,
            curr_vel,
            **kwargs,
        )

        per_sample= self.reduce_fn(per_mode, topk_w) # per default best among K -> [B]

        return per_sample

    @torch.no_grad()
    def update(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        pred_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ):
        per_sample = self.compute_per_sample_values(
            gt_traj,
            gt_mask,
            pred_traj,
            pred_prob,
            gt_position_idx,
            pred_position_idx,
            curr_vel,
            **kwargs,
        )

        self.sum_value += per_sample.sum()
        self.n_samples += per_sample.numel()

    def compute(self):
        return self.sum_value / self.n_samples.clamp_min(1).float()


class ADE(Displacement):
    """
    Average Displacement Error over valid steps (ADE). Best among top-K modes.
    """
    def _per_mode_value(self, diff, gt_mask, **kwargs) -> torch.Tensor:
        mask = _resolve_timestep_as_mask(self.t_max, gt_mask)  # [B, T]
        mask = mask.unsqueeze(1)                       # [B, 1, T]
        valid = mask.float().sum(dim=-1).clamp_min(1.0)   # [B, 1]
        return (diff * mask).sum(dim=-1) / valid          # [B, K]

class FDE(Displacement):
    """
    Final Displacement Error at chosen timestep. Best among top-K modes.
    timestep by default last valid observation
    """
    def _per_mode_value(self, diff, gt_mask, **kwargs) -> torch.Tensor: 
        B, K, T = diff.shape
        idx = _resolve_final_timestep_as_batch(self.t_max, gt_mask)
        return diff.gather(-1, idx.view(B, 1, 1).expand(B, K, 1)).squeeze(-1)  # [B, K]
    
class BrierFDE(FDE):
    """
    Brier-minFDE@K: combines minimum Final Displacement Error (FDE)
    over top-K modes with a probability calibration penalty.

        brier_fde = minFDE@K + (1 - p_best)^2

    where:
        - p_best = predicted probability of the mode that achieved minFDE
        - timestep: timestep to use for final displacement
                 (-1 → last valid timestep from gt_final_valid_idx)
    """

    @torch.no_grad()
    def compute_per_sample_values(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        pred_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ) -> torch.Tensor:
        B, M, T, D = pred_traj.shape

        per_mode, topk, topk_w = self._compute_per_mode_values(
            gt_traj,
            gt_mask,
            pred_traj,
            pred_prob,
            gt_position_idx,
            pred_position_idx,
            curr_vel,
            **kwargs,
        )

        best_idx = per_mode.argmin(dim=1)
        best = per_mode[torch.arange(B), best_idx]
        best_p = topk_w[torch.arange(B), best_idx]

        per_sample = best + (1.0 - best_p).pow(2)

        return per_sample 

class MissRate(FDE):
    """Binary miss over top-K: 1 if best final displacement > miss_threshold (meters), else 0."""
    def __init__(self, miss_threshold: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.miss_threshold = float(miss_threshold)

    @torch.no_grad()
    def compute_per_sample_values(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        pred_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ) -> torch.Tensor:
        per_mode, topk, topk_w = self._compute_per_mode_values(
            gt_traj,
            gt_mask,
            pred_traj,
            pred_prob,
            gt_position_idx,
            pred_position_idx,
            curr_vel,
            **kwargs,
        )

        best = per_mode.min(dim=1).values # [B]
        per_sample = (best > self.miss_threshold).float()  # [B]

        return per_sample
    


