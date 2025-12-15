"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import torch
import torchmetrics as tm
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict
import ot

from predictor.metrics.utils import _resolve_final_timestep_as_int, _extract_gaussian_params, _select_topk, _extract_xy_coordinates, _normalize_weights, _get_denominator

class GaussianMixtureConsistency(tm.Metric):
    """
    Consistency between gaussian mixtures using POT.

    Expects:
      - traj_prev: [A, N, T, D] with 5-tuple (mu_x, mu_y, std_x, std_y, rho)
      - traj_curr: [A, N, T, D] with same layout
      - prob_prev: [A, N]
      - prob_curr: [A, N]
      - param_idx: indices/slice selecting the 5 parameters
      - curr_vel (Optional): [A] (m/s) for velocity normalization

    Options:
      - t_max: which t to read (negative = from end)
      - k: number of top weighted components to consider (k>=1)
      - eps: numerical floor / clamps
    """
    def __init__(self, t_max: int = -1, k: int = -1, eps: float = 1e-8, normalize: bool=False):
        super().__init__()
        self.t_max = int(t_max)
        self.eps = float(eps)
        self.k = int(k)
        self.normalize = normalize

        self.add_state("sum_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    # ---- public API ----

    @torch.no_grad()
    def update(
        self,
        traj_prev: torch.Tensor,  # [A, N, T, D]
        traj_curr: torch.Tensor,  # [A, N, T, D]
        prob_prev: torch.Tensor,  # [A, N]
        prob_curr: torch.Tensor,  # [A, N]
        param_idx: Union[int, slice, list[int]] = slice(0, 5),
        curr_vel: Optional[torch.Tensor] = None,  # [A], optional m/s
    ) -> torch.Tensor:
        A, N, T, D = traj_prev.shape

        denom = _get_denominator(A, curr_vel, self.normalize).reshape(-1, 1, 1)  # [A,1,1]

        t_curr = _resolve_final_timestep_as_int(T, self.t_max)
        t_prev = _resolve_final_timestep_as_int(T, t_curr - 1)

        idx_prev = _select_topk(self.k, prob_prev, N)
        idx_curr = _select_topk(self.k, prob_curr, N)

        mu_prev, _, _ , _, C_prev = _extract_gaussian_params(traj_prev, traj_idx = idx_prev, param_idx = param_idx, t = t_prev)
        mu_curr, _, _ , _, C_curr = _extract_gaussian_params(traj_curr, traj_idx = idx_curr, param_idx = param_idx, t = t_curr)

        w_prev = _normalize_weights(prob_prev.gather(1, idx_prev), self.eps)
        w_curr = _normalize_weights(prob_curr.gather(1, idx_curr), self.eps)
        # delegate to concrete metric (returns [A,1,1] in distance units)
        per_sample = self._consistency(mu_prev, mu_curr, C_prev, C_curr, w_prev, w_curr, denom)  # [A]

        # accumulate mean
        self.sum_value += per_sample.sum()
        self.n_samples += per_sample.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_value / self.n_samples.clamp_min(1).float()

    def _consistency(self, mu_prev, mu_curr, C_prev, C_curr, w_prev, w_curr, denom) -> torch.Tensor:
        """
        Compute per-batch W2 between (top-Kp) prev and (top-Kc) curr Gaussian mixtures.

        Returns:
        out: [A,1,1] in meters (or seconds if denom provided). If self.squared=True,
            returns squared units (meters^2 or seconds^2).
        """
        A = mu_prev.shape[0]
        device, dtype = mu_prev.device, mu_prev.dtype
        Kp = mu_prev.shape[1]
        Kc = mu_curr.shape[1]

        vals = []
        for a in range(A):
            if Kp == 1 and Kc == 1:
                # Fast path: Bures–Wasserstein between two Gaussians → W2 (meters)
                v = ot.gaussian.bures_wasserstein_distance(
                    mu_prev[a, 0], mu_curr[a, 0], C_prev[a, 0], C_curr[a, 0]
                )
            else:
                # Full GMM OT on reduced mixtures → typically W2 (meters)
                v = ot.gmm.gmm_ot_loss(
                    mu_prev[a], mu_curr[a], C_prev[a], C_curr[a], w_prev[a], w_curr[a]
                )
            # Ensure tensor, right device/dtype
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, device=device, dtype=dtype)
            else:
                v = v.to(device=device, dtype=dtype)
            vals.append(v)

        W = torch.stack(vals, dim=0).reshape(A, 1, 1) / denom # [A,1,1], meters
        per_sample = W.reshape(A)

        return per_sample # [A]                   # seconds if denom=m/s; else meters
    


# TODO: might require introduction of power parameter
class GeometricConsistency(tm.Metric):
    """
    Consistency between trajectory ensembles at consecutive timesteps using POT.

    - k=1 → Dirac: direct Minkowski distance between top-1 modes.
    - k>1 (or None) → Discrete mixture OT (DMM) using POT's EMD.
    - Optionally normalized by velocity to yield time units [s] or [s^p].

    Args:
      t: which t to compare (prev=t-1, curr=t)
      k: top-K modes to consider (<=0 uses all modes)
      metric: 'euclidean', 'sqeuclidean', 'cityblock', etc. (POT-compatible)
      eps: small number for weight normalization
    """
    full_state_update = False

    def __init__(
        self,
        t: int = -1,
        k: int = -1,
        metric: str = "euclidean",
        eps: float = 1e-8,
        normalize: bool = False,
    ):
        super().__init__()
        self.t = int(t)
        self.k = k
        self.metric = metric
        self.eps = float(eps)
        self.normalize = normalize

        self.add_state("sum_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0),   dist_reduce_fx="sum")

    @torch.no_grad()
    def update(
        self,
        traj_prev: torch.Tensor,   # [A,N,T,D]
        traj_curr: torch.Tensor,   # [A,N,T,D]
        prob_prev: torch.Tensor,   # [A,N]
        prob_curr: torch.Tensor,   # [A,N]
        curr_vel: Optional[torch.Tensor] = None,  # [A]
        position_idx: Union[int, slice, list[int]] = slice(0, 2),
        **kwargs
    ):
        A, N, T, D = traj_prev.shape
        device, dtype = traj_prev.device, traj_prev.dtype

        t_curr = _resolve_final_timestep_as_int(T, self.t)
        t_prev = _resolve_final_timestep_as_int(T, t_curr - 1)

        # normalize and select modes
        idx_prev = _select_topk(prob_prev, self.k)
        idx_curr = _select_topk(prob_curr, self.k)

        xy_prev = _extract_xy_coordinates(traj_prev, t = t_prev, position_idx = position_idx, traj_idx = idx_prev)  # [A,Kp,2]
        xy_curr = _extract_xy_coordinates(traj_curr, t= t_curr, position_idx = position_idx, traj_idx = idx_curr)  # [A,Kc,2]
        w_prev = _normalize_weights(prob_prev.gather(1, idx_prev), self.eps)
        w_curr = _normalize_weights(prob_curr.gather(1, idx_curr), self.eps)

        denom = _get_denominator(A, device, dtype, curr_vel, self.normalize).view(A, 1, 1)

        per_sample = self._consistency(xy_prev, xy_curr, w_prev, w_curr, denom)  # [A]

        self.sum_value += per_sample.sum()
        self.n_samples += torch.tensor(A, device=device)

    def compute(self) -> torch.Tensor:
        return self.sum_value / self.n_samples.clamp_min(1).float()
    
    def _consistency(
        self,
        xy_prev: torch.Tensor,             # [A,Kp,2]
        xy_curr: torch.Tensor,             # [A,Kc,2]
        w_prev: torch.Tensor,              # [A,Kp]
        w_curr: torch.Tensor,              # [A,Kc]
        denom: torch.Tensor,               # [A,1,1], time normalizer (v or ones)
    ) -> torch.Tensor:
        A = xy_prev.shape[0]
        device, dtype = xy_prev.device, xy_prev.dtype
        Kp, Kc = xy_prev.shape[1], xy_curr.shape[1]

        vals = []
        for a in range(A):
            # POT expects numpy or torch on CPU
            cost = ot.dist(xy_prev[a], xy_curr[a], metric=self.metric,)
            if Kp == 1 and Kc == 1:
                # Dirac distribution case (just one element)
                v = cost[0, 0]
            else:
                # Discrete mixture distribution case: earth mover’s distance
                v = ot.emd2(w_prev[a], w_curr[a], cost,)
            vals.append(torch.tensor(v, device=device, dtype=dtype))

        W = torch.stack(vals, dim=0).reshape(A, 1, 1) / denom
        per_sample = W.reshape(A)

        return per_sample