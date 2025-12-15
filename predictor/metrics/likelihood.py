"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import torch
import torchmetrics as tm
#from torch.amp import autocast
from typing import Optional, Union, Dict, Literal

from predictor.metrics.utils import _resolve_final_timestep_as_batch, _resolve_timestep_as_mask, _extract_gaussian_params, _normalize_weights, _get_denominator, _select_topk

class mixtureNLL(tm.Metric):
    
    def __init__(
            self, 
            k: int = -1, 
            t_max: int = -1,
            eps: float = 1e-8,
        ):
        super().__init__()
        self.k = int(k)
        self.t_max = int(t_max)
        self.eps = float(eps)

        self.add_state("sum_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0),   dist_reduce_fx="sum")

    def _compute_gaussian_pdf(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2),
        pred_param_idx: Union[int, slice, list[int]] = slice(0,5), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ):
        B, M, T, D = pred_traj.shape
        # extract gaussian parameters for choosen modes
        topk = _select_topk(self.k, pred_prob, M)                            # [B, K]
        K = topk.size(1)
        topk_prob = _normalize_weights(pred_prob.gather(1, topk)) # [B, K]

        # extract predicted gaussian parameters
        mu, std_x, std_y, rho, _ = _extract_gaussian_params(pred_traj, traj_idx = topk, param_idx = pred_param_idx)
        mu_x, mu_y = mu[..., 0], mu[..., 1]
        log_prob = torch.log(topk_prob.clamp_min(self.eps))
        log_std_x = torch.log(std_x)
        log_std_y = torch.log(std_y)
        # extract gt positions
        gt_xy = gt_traj[..., gt_position_idx]     # [B, T, 2]
        x, y  = gt_xy[..., 0].unsqueeze(1), gt_xy[..., 1].unsqueeze(1)  # [B, 1, T], [B, 1, T]

        # Standardized deltas
        dx = (x - mu_x) / std_x                                            # [B, K, T]
        dy = (y - mu_y) / std_y                                            # [B, K, T]

        # Bivariate normal log-pdf per mode
        # log N = -log(2π) - log σx - log σy - 0.5*log(1-ρ²) - (dx² - 2ρ dx dy + dy²)/(2(1-ρ²))
        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, device=mu_x.device, dtype=mu_x.dtype))
        log_one_minus_r2 = torch.log1p(-rho * rho)                      # stable log(1 - ρ²)
        quad = dx * dx - 2.0 * rho * dx * dy + dy * dy
        den  = 2.0 * (1.0 - rho * rho)
        log_pdf = -(log_two_pi + log_std_x + log_std_y) - 0.5 * log_one_minus_r2 - quad / den  # [B, K, T]

        # Mixture: log ∑_k π_k N_k
        log_mixture = torch.logsumexp(log_pdf + log_prob.unsqueeze(-1), dim=1)        # [B, T]

        return log_pdf, log_mixture
    
    @torch.no_grad()
    def compute_per_sample_values(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2),
        pred_param_idx: Union[int, slice, list[int]] = slice(0,5), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ) -> torch.Tensor:
        
        _, log_mixture = self._compute_gaussian_pdf(
            gt_traj,
            gt_mask,
            pred_traj,
            pred_prob,
            gt_position_idx,
            pred_param_idx,
            curr_vel,
            **kwargs,
        )

        # resolve timestep with clamping so out-of-range indices don't crash
        t_mask = _resolve_timestep_as_mask(self.t_max, gt_mask) # [B, T]
        log_mixture *= t_mask # [B, T]
        counts = t_mask.sum(dim=-1).clamp_min(1.0) # [B]

        # Compute negative log-likelihood for each time step: -log(p)
        nll = - log_mixture.sum(dim=-1) / counts  # [B]

        return nll


    @torch.no_grad()
    def update(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2),
        pred_param_idx: Union[int, slice, list[int]] = slice(0,5), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ):
        
        nll = self.compute_per_sample_values(
            gt_traj, gt_mask,
            pred_traj, pred_prob,
            gt_position_idx, pred_param_idx,
            curr_vel, **kwargs
        )

        self.sum_value += nll.sum()
        self.n_samples += nll.numel()

    def compute(self):
        return self.sum_value / self.n_samples.clamp_min(1).float()
    
class mixtureFNLL(mixtureNLL):

    @torch.no_grad()
    def compute_per_sample_values(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2),
        pred_param_idx: Union[int, slice, list[int]] = slice(0,5), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ) -> torch.Tensor:

        _, log_mixture = self._compute_gaussian_pdf(
            gt_traj,
            gt_mask,
            pred_traj,
            pred_prob,
            gt_position_idx,
            pred_param_idx,
            curr_vel,
            **kwargs,
        )

        # resolve timestep with clamping so out-of-range indices don't crash
        t_batch = _resolve_final_timestep_as_batch(self.t_max, gt_mask) # [B]
        log_mixture = log_mixture.gather(1, t_batch.view(-1, 1)).squeeze(1) # [B, T] -> [B]

        # Compute negative log-likelihood for each time step: -log(p)
        nll = -log_mixture  # [B]

        return nll

class minFNLL(mixtureFNLL):

    @torch.no_grad()
    def compute_per_sample_values(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2),
        pred_param_idx: Union[int, slice, list[int]] = slice(0,5), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ) -> torch.Tensor:

        log_pdf, _ = self._compute_gaussian_pdf(
            gt_traj,
            gt_mask,
            pred_traj,
            pred_prob,
            gt_position_idx,
            pred_param_idx,
            curr_vel,
            **kwargs,
        )
        # resolve timestep with clamping so out-of-range indices don't crash
        B, K, T = log_pdf.shape
        t_batch = _resolve_final_timestep_as_batch(self.t_max, gt_mask) # [B]
        log_pdf = log_pdf.gather(2, t_batch.view(B, 1, 1).expand(-1, K, 1)).squeeze(2) # [B, T] -> [B]
        # Compute negative log-likelihood for each time step: -log(p)
        nll = -log_pdf.max(dim=1).values # [B,K] -> [B]

        return nll

class minNLL(mixtureNLL):

    @torch.no_grad()
    def compute_per_sample_values(
        self,
        gt_traj: torch.Tensor,         # [B, T, D]
        gt_mask: torch.Tensor,         # [B, T]
        pred_traj: torch.Tensor,       # [B, M, T, D]
        pred_prob: torch.Tensor,       # [B, M]
        gt_position_idx: Union[int, slice, list[int]] = slice(0,2),
        pred_param_idx: Union[int, slice, list[int]] = slice(0,5), #
        curr_vel: Optional[torch.Tensor] = None,        # [B]
        **kwargs,
    ) -> torch.Tensor:

        log_pdf, _ = self._compute_gaussian_pdf(
            gt_traj,
            gt_mask,
            pred_traj,
            pred_prob,
            gt_position_idx,
            pred_param_idx,
            curr_vel,
            **kwargs,
        )

        # resolve timestep with clamping so out-of-range indices don't crash
        t_mask = _resolve_timestep_as_mask(self.t_max, gt_mask) # [B, T]
        log_pdf = log_pdf.max(dim=1).values  # [B, T]
        log_pdf *= t_mask # [B, T]
        counts = t_mask.sum(dim=-1).clamp_min(1.0) # [B]

        # Compute negative log-likelihood for each time step: -log(p)
        nll = - log_pdf.sum(dim=-1) / counts  # [B]

        return nll
