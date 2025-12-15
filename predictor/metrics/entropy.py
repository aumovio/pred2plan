"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import torch
import torchmetrics as tm
#from torch.amp import autocast
from typing import Optional, Union, Dict, Literal

from predictor.metrics.utils import _resolve_final_timestep_as_int, _extract_gaussian_params, _extract_xy_coordinates, _normalize_weights, _get_denominator, _select_topk
from predictor.metrics.displacement import ADE, FDE

class SoftmaxEntropy(tm.Metric):

    def __init__(self, normalize: bool=True, k: int = -1, eps: float=1e-8):
        super().__init__()
        self.k = int(k)
        self.normalize = normalize
        self.eps = float(eps)

        self.add_state("sum_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0),   dist_reduce_fx="sum")

    @torch.no_grad()
    def compute_per_sample_values(self, pred_prob: torch.Tensor, **kwargs) -> torch.Tensor: # pred_prob in shape [B, N]
        B, N = pred_prob.shape
        # choose modes
        topk = _select_topk(self.k, pred_prob, N) # [B, K]
        K = topk.size(1)
        topk_prob = _normalize_weights(pred_prob.gather(1, topk)) # [B, K]

        topk_prob = topk_prob.clamp_min(self.eps)
        H = -torch.sum(topk_prob * torch.log(topk_prob), dim=1) # [B] in nats
        if self.normalize:
            H = H / torch.log(torch.tensor(N, device=H.device, dtype=H.dtype))

        return H

    @torch.no_grad()
    def update(self, pred_prob: torch.Tensor, **kwargs): # pred_prob in shape [B, N]
        H = self.compute_per_sample_values(pred_prob, **kwargs)
        self.sum_value += H.sum()
        self.n_samples += H.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_value / self.n_samples.clamp_min(1).float()

class GaussianMixtureEntropy(tm.Metric):

    def __init__(self, 
                 t_max: int=-1, 
                 k: int = -1,
                 samples: int = 10000, 
                 eps: float = 1e-8):
        super().__init__()
        self.t_max = int(t_max)
        self.k = int(k)
        self.samples = int(samples)
        self.eps = float(eps)

        self.add_state("sum_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0),   dist_reduce_fx="sum")

    @torch.no_grad()
    def compute_per_sample_values(self, 
               pred_traj: torch.Tensor, # [B, N, T, D]
               pred_prob: torch.Tensor, #[B, N]
               pred_param_idx: slice | list[int] = slice(0,5),
               **kwargs
            ) -> torch.Tensor: 
        A, N, T, D = pred_traj.shape
        # resolve timestep safely
        t = _resolve_final_timestep_as_int(T, self.t_max)
        # choose modes and extract gaussian parameters
        topk = _select_topk(self.k, pred_prob, N)                            # [B, K]
        K = topk.size(1)
        topk_prob = _normalize_weights(pred_prob.gather(1, topk)) # [B, K]
        mu, std_x, std_y, rho, _ = _extract_gaussian_params(pred_traj, traj_idx = topk, param_idx = pred_param_idx, t = t)
        mu_x, mu_y = mu[..., 0], mu[..., 1]
        # log probabilities
        log_prob = torch.log(topk_prob.clamp_min(self.eps))
        # Sample component indices according to π
        mix_idx = torch.multinomial(topk_prob, num_samples=self.samples, replacement=True)  # (B, num_samples)

        # Gather parameters for each sample
        mu_x_s = torch.gather(mu_x, 1, mix_idx)  # (B, S)
        mu_y_s = torch.gather(mu_y, 1, mix_idx)
        sx_s   = torch.gather(std_x, 1, mix_idx)
        sy_s   = torch.gather(std_y, 1, mix_idx)
        rho_s  = torch.gather(rho, 1, mix_idx)

        # Draw base normals and construct correlated samples
        u = torch.randn_like(mu_x_s)
        v = torch.randn_like(mu_x_s)
        x_s = mu_x_s + sx_s * u
        y_s = mu_y_s + sy_s * (rho_s * u + torch.sqrt(1 - rho_s**2) * v)

        # Prepare for density eval: expand back to (B, S, m)
        x = x_s.unsqueeze(-1)             # (B, S, 1)
        y = y_s.unsqueeze(-1)             # (B, S, 1)
        mu_x_e  = mu_x.unsqueeze(1)       # (B, 1, m)
        mu_y_e  = mu_y.unsqueeze(1)
        sx_e    = std_x.unsqueeze(1)
        sy_e    = std_y.unsqueeze(1)
        rho_e   = rho.unsqueeze(1)
        log_pi_e= log_prob.unsqueeze(1)

        # Compute the exponent term of each component
        zx = (x - mu_x_e) / sx_e
        zy = (y - mu_y_e) / sy_e
        denom = 1 - rho_e**2
        exp_term = -0.5 * (zx**2 - 2*rho_e*zx*zy + zy**2) / denom

        # Normalizing constant
        norm_const = 2 * torch.pi * sx_e * sy_e * torch.sqrt(denom)
        log_comp = exp_term - torch.log(norm_const)  # (B, S, m)

        # Log-density of mixture via log-sum-exp
        log_mix = torch.logsumexp(log_pi_e + log_comp, dim=-1)  # (B, S)

        # MC estimate of entropy: H ≈ -E[log p(x)]
        H = -log_mix.mean(dim=1)  # (B,)

        return H

    @torch.no_grad()
    def update(self, 
               pred_traj: torch.Tensor, # [B, N, T, D]
               pred_prob: torch.Tensor, #[B, N]
               pred_param_idx: slice | list[int] = slice(0,5),
               **kwargs
            ): 
        
        H = self.compute_per_sample_values(pred_traj, pred_prob, pred_param_idx, **kwargs)
        
        self.sum_value += H.sum()
        self.n_samples += H.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_value / self.n_samples.clamp_min(1).float()


class BestGaussianEntropy(tm.Metric):
    def __init__(self, displacement: FDE = FDE(), t_max: int=-1, eps: float=1e-8):
        super().__init__()
        self.displacement = displacement
        self.t_max = int(t_max)
        self.eps = float(eps)

        self.add_state("sum_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0),   dist_reduce_fx="sum")

    @torch.no_grad()
    def compute_per_sample_values(self, 
                gt_traj: torch.Tensor,         # [B, T, D]
                gt_mask: torch.Tensor,         # [B, T]
                pred_traj: torch.Tensor,       # [B, M, T, D]
                pred_prob: torch.Tensor,       # [B, M]
                gt_position_idx: slice | list[int] = slice(0,2),
                pred_param_idx: slice | list[int] = slice(0,5),
                curr_vel: Optional[torch.Tensor] = None,        # [B]
                **kwargs,
            ) -> torch.Tensor: 

        A, N, T, D = pred_traj.shape
        # extract best fitting index from geometric point of view
        if isinstance(pred_param_idx, slice):
            pred_position_idx = slice(pred_param_idx.start, pred_param_idx.start+2)#
        else:
            pred_position_idx = pred_param_idx[:2]
            
        per_mode, topk, topk_w = self.displacement._compute_per_mode_values(
            gt_traj,
            gt_mask,
            pred_traj,
            pred_prob,
            gt_position_idx,
            pred_position_idx,
            curr_vel,
            **kwargs,
        )
        best_idx = per_mode.argmin(dim=1).unsqueeze(1) # [A, K]

        # resolve timestep at which to measure entropy
        t = _resolve_final_timestep_as_int(T, self.t_max,)
        # extract gaussian parameters
        mu, std_x, std_y, rho, _ = _extract_gaussian_params(pred_traj, traj_idx=best_idx, param_idx = pred_param_idx, t = t)
        # entropy
        det_cov = std_x**2 * std_y**2 * (1 - rho**2)
        H = torch.log(2 * torch.pi * torch.e * torch.sqrt(det_cov))

        return H.squeeze(-1) # [B]

    @torch.no_grad()
    def update(self, 
                gt_traj: torch.Tensor,         # [B, T, D]
                gt_mask: torch.Tensor,         # [B, T]
                pred_traj: torch.Tensor,       # [B, M, T, D]
                pred_prob: torch.Tensor,       # [B, M]
                gt_position_idx: slice | list[int] = slice(0,2),
                pred_param_idx: slice | list[int] = slice(0,5),
                curr_vel: Optional[torch.Tensor] = None,        # [B]
                **kwargs,
            ): 
        H = self.compute_per_sample_values(
            gt_traj, gt_mask,
            pred_traj, pred_prob,
            gt_position_idx, pred_param_idx,
            curr_vel, **kwargs
        )

        self.sum_value += H.sum()
        self.n_samples += H.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_value / self.n_samples.clamp_min(1).float()
