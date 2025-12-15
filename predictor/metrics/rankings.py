"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import torch
import torchmetrics as tm
from typing import Optional, Union
from abc import abstractmethod

from predictor.metrics.displacement import ADE, FDE

def _ranks_from_order(order_idx: torch.Tensor) -> torch.Tensor:
    """order_idx: [B,K] of indices sorted by preference → ranks[b, j] = position of mode j."""
    B, K = order_idx.shape
    ranks = torch.empty_like(order_idx)
    positions = torch.arange(K, device=order_idx.device).expand(B, -1)  # 0..K-1
    ranks.scatter_(1, order_idx, positions)
    return ranks  # [B,K]


class Rank(tm.Metric):

    def __init__(
        self,
        displacement: FDE = FDE(),
        ordering: str = "descending",   
    ):
        super().__init__()
        self.displacement = displacement
        self.ordering = ordering

        self.add_state("sum_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0),   dist_reduce_fx="sum")

    def compare_ranks(self, dist_scores, prob_scores, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def compute_per_sample_values(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        pred_param_idx: Union[int, slice, list[int]] = slice(0,2), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ) -> torch.Tensor:
        per_mode, topk, topk_w = self.displacement._compute_per_mode_values(
            gt_traj,
            gt_mask,
            pred_traj,
            pred_prob,
            gt_position_idx,
            pred_param_idx,
            curr_vel,
            **kwargs,
        )
        
        order_d = torch.argsort(per_mode, dim=1, descending=False)
        order_p = topk_w
        per_sample = self.compare_ranks(order_d, order_p, **kwargs)

        return per_sample

    @torch.no_grad()
    def update(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2), #
        pred_param_idx: Union[int, slice, list[int]] = slice(0,2), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ) -> None:
        per_sample = self.compute_per_sample_values(
            gt_traj, gt_mask,
            pred_traj, pred_prob,
            gt_position_idx, pred_param_idx,
            curr_vel, **kwargs
        )

        self.sum_value += per_sample.sum()
        self.n_samples += per_sample.numel()

    def compute(self):
        return self.sum_value / self.n_samples.clamp_min(1).float()

class Rank1(Rank):
    
    def compare_ranks(self, dist_scores, prob_scores, **kwargs):
        dist_order = torch.argsort(dist_scores, dim=1, descending=False)
        dist_ranks = _ranks_from_order(dist_order)                      # [B,K]
        best_prob = prob_scores.argmax(dim=1, keepdim=True)                  # [B,1]
        r = dist_ranks.gather(1, best_prob).squeeze(1).float()          # [B], 0..K-1
        return r
    
class RankBest(Rank):

    def compare_ranks(self, dist_scores, prob_scores, **kwargs):
        prob_order = torch.argsort(prob_scores, dim=1, descending=True)      # [B,K]
        prob_ranks = _ranks_from_order(prob_order)                      # [B,K]
        best_dist = dist_scores.argmin(dim=1, keepdim=True)                # [B,1]
        r = prob_ranks.gather(1, best_dist).squeeze(1).float()          # [B], 0..K-1
        return r
    
