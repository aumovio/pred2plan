"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import torch
from typing import Union, Optional, Dict
import torchmetrics as tm
from hydra.utils import instantiate

def build_metric_collection(cfg_section: Dict, prefix: str, separator: str = "_") -> tm.MetricCollection:
    """
    cfg_section like:
      {"minNLL": {"_target_": "predictor.metrics.minNLL", "k": 6}, ...}
    """
    if not cfg_section:
        return tm.MetricCollection({})
    
    metrics = {}
    for name, cfg in cfg_section.items():
        if isinstance(cfg, tm.Metric):
            metric = cfg
        else:
            metric = instantiate(cfg)
        metrics[name] = metric

    prefix = prefix if prefix.endswith(separator) else prefix + separator
    return tm.MetricCollection(metrics, prefix=prefix)

def _resolve_final_timestep_as_int(T: int, t: int,) -> int:
    """
    Clamp/resolve t into [0, T-1], supporting negative indexing.
    """
    t_resolved = t if t >= 0 else T + t
    t_resolved = min(t_resolved, T - 1)
    return max(0, t_resolved)

def _resolve_final_timestep_as_batch(
        t: int,  
        mask: torch.Tensor # [B, T]
    ) -> torch.Tensor: # [B]
    """
    Resolve a global timestep t into per-batch valid timesteps.

    Returns the minimum between the global timestep (after handling negatives)
    and each batch’s last valid timestep from mask.
    """
    B, T = mask.shape
    t_max = _resolve_final_timestep_as_int(T, t) # int between 0 and T
    t_batch = torch.full((B,), t_max, device=mask.device, dtype=torch.long) # [B]
    t_valid = _get_final_valid_idx(mask).to(torch.long) # [B]
    t_resolved = torch.minimum(t_valid.clamp_min(0), t_batch)
    return t_resolved.long() # [B]

def _resolve_timestep_as_mask(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Create a [B, T] mask that keeps timesteps up to t and zeroes out the rest.

    Combines the given validity mask with a cutoff at the resolved timestep.
    """
    B, T = mask.shape
    t_max = _resolve_final_timestep_as_int(T, t) # int between 0 and T
    t_batch = torch.full((B,), t_max, device=mask.device, dtype=torch.float) # [B]
    idx = torch.arange(T, device=mask.device).unsqueeze(0).expand(B, -1) # [B, T]
    mask_cut = (idx <= t_batch.unsqueeze(1)) # [B, T]
    mask_resolved = mask.bool() & mask_cut # [B, T]
    return mask_resolved


def _make_covariance(std_x: torch.Tensor, std_y: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    sxx = std_x ** 2
    syy = std_y ** 2
    sxy = rho * std_x * std_y
    Sigma = torch.stack([
        torch.stack([sxx, sxy], dim=-1),
        torch.stack([sxy, syy], dim=-1)
    ], dim=-2)  # [..., 2, 2]
    return Sigma

def _extract_gaussian_params(
        traj: torch.Tensor,  # [A, N, T, D]
        traj_idx: Optional[torch.Tensor] = None, # [A, K] 
        param_idx: Union[int, slice, list[int]] = slice(0,5),
        t: Optional[int | torch.Tensor] = None,
        eps: float = 1e-3,
    ):
        A, N, T, D = traj.shape

        if traj_idx is not None:
            K = traj_idx.shape[1]
            traj = traj.gather(1, traj_idx[..., None, None].expand(A, K, T, D))  # [A,K,T,D]
        params = traj[..., param_idx]  # [A, K, T, 5]

        if t is not None:
            if isinstance(t, int):
                params = params[..., t, :] # [A, K, 5]
            else:
                params = params.gather(2, t.view(A, 1, 1, 1).expand(A, K, 1, 5)).squeeze(2) # [A, K, 5]

        mu = params[..., 0:2]  # [A, K, T, 2] or [A, K, 2]
        sx = torch.clamp(params[..., 2], min=eps)
        sy = torch.clamp(params[..., 3], min=eps)
        rho = torch.clamp(params[..., 4], min=-(1-eps), max=(1-eps))
        Sigma = _make_covariance(sx, sy, rho)  # [A, K, T, 2, 2] or [A, K, 2, 2]
        return mu, sx, sy, rho, Sigma

def _extract_xy_coordinates(
    traj: torch.Tensor,                       # [A, N, T, D]
    traj_idx: Optional[torch.Tensor] = None,       # [A, K] per-batch indices or None (all N)
    t: Optional[int | torch.Tensor] = -1,                                 
    position_idx: Union[int, slice, list[int]] = slice(0, 2),
) -> torch.Tensor:
    """Return positions at t as [A, K, 2] (K=N if idx=None)."""
    A, N, T, D = traj.shape

    if traj_idx is not None:
        K = traj_idx.shape[1]
        traj = traj.gather(1, traj_idx[..., None, None].expand(A, K, T, D))  # [A,K,T,D]
    xy = traj[..., position_idx]  # [A, K, T, 2]

    if t is not None:
        if isinstance(t, int):
            xy = xy[..., t, :] # [A, K, 2]
        else:
            xy = xy.gather(2, t.view(A, 1, 1, 1).expand(A, K, 1, 5)).squeeze(2) # [A, K, 2]
    return xy

def _select_topk(k, pred_prob: torch.Tensor, M: int) -> torch.Tensor:
    # prob: [A,N] -> indices: [A,K]
    K = min(k, M) if k>0 else M
    return torch.topk(pred_prob, k=K, dim=1).indices  # [A, K]

def _normalize_weights(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return w / w.clamp_min(eps).sum(dim=1, keepdim=True)

def _get_denominator(A: int, device: torch.device, dtype: torch.dtype, curr_vel: Optional[torch.Tensor], normalize: Optional[bool] = False,
) -> torch.Tensor:
    if not normalize:
        denom = torch.ones(A, device=device, dtype=dtype) # [A]
    else:
        denom = curr_vel.to(device=device, dtype=dtype).clamp_min(1.0) # [A]
    return denom

def _get_final_valid_idx(gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Returns the last valid index t in [0, T-1] where mask==1 for
    gt_mask: [B, T] (bool or 0/1). 
    """
    idx = torch.arange(1, gt_mask.shape[1] + 1).expand(gt_mask.shape[0], -1).to(gt_mask.device)
    gt_mask_int = gt_mask*idx
    gt_final_valid_idx = torch.argmax(gt_mask_int, dim=1)
    return gt_final_valid_idx # [B]


#### OLD STUFF

# def unwrap(theta):
#     d        = theta[..., 1:] - theta[..., :-1]
#     d_mod    = (d + torch.pi) % (2*torch.pi) - torch.pi
#     theta0   = theta[..., :1]
#     return torch.cat([theta0, theta0 + torch.cumsum(d_mod, dim=-1)], dim=-1)

# def compute_derivative(arr, dt=0.1):
#     """
#     Compute the time derivative along the last axis using central differences.
#     For endpoints, use forward/backward differences.
#     """
#     d_arr = torch.zeros_like(arr)
#     # Central differences for interior points
#     d_arr[..., 1:-1] = (arr[..., 2:] - arr[..., :-2]) / (2*dt)
#     # Forward difference for first timestep
#     d_arr[..., 0] = (arr[..., 1] - arr[..., 0]) / dt
#     # Backward difference for last timestep
#     d_arr[..., -1] = (arr[..., -1] - arr[..., -2]) / dt
#     return d_arr

# def compute_kinematics(trajectories, dt=0.1):
#     """
#     Compute kinematics for trajectories of shape [batch_size, modes, horizon, 2],
#     where the last dimension corresponds to x and y.
    
#     This function computes:
#     - Velocities (vx, vy) and speed.
#     - Heading (yaw angle theta).
#     - Accelerations (ax, ay) and decomposes them into longitudinal (a_lon) and lateral (a_lat).
#     - Yaw rate (theta_dot) and yaw acceleration (theta_ddot).
#     - Jerk (both overall magnitude and longitudinal jerk).
    
#     For each kinematic quantity, the maximum absolute value along the horizon is returned.
#     """
#     # Extract x and y from trajectories
#     # trajectories shape: [batch_size, modes, horizon, 2]
#     x = trajectories[..., 0]  # shape: [batch_size, modes, horizon]
#     y = trajectories[..., 1]  # shape: [batch_size, modes, horizon]
#     # Compute velocities
#     vx = compute_derivative(x, dt)
#     vy = compute_derivative(y, dt)
#     speed = torch.sqrt(vx**2 + vy**2)
    
#     # Compute heading (yaw)
#     theta = unwrap(torch.arctan2(vy, vx))
    
#     # Compute accelerations
#     ax = compute_derivative(vx, dt)
#     ay = compute_derivative(vy, dt)
    
#     # Decompose acceleration into body-fixed frame components:
#     # longitudinal acceleration: along the heading direction
#     a_lon = ax * torch.cos(theta) + ay * torch.sin(theta)
#     # lateral acceleration: perpendicular to heading
#     a_lat = -ax * torch.sin(theta) + ay * torch.cos(theta)
    
#     # Compute yaw rate and yaw acceleration
#     theta_dot = compute_derivative(theta, dt)
#     theta_ddot = compute_derivative(theta_dot, dt)
    
#     # Compute jerk: derivative of acceleration
#     jx = compute_derivative(ax, dt)
#     jy = compute_derivative(ay, dt)
#     jerk_mag = torch.sqrt(jx**2 + jy**2)
    
#     # Compute longitudinal jerk: derivative of a_lon
#     jerk_lon = compute_derivative(a_lon, dt)
    
#     # Aggregate the kinematics by taking the maximum absolute value along the horizon (axis=-1)
#     def aggregate(arr):
#         return torch.amax(torch.abs(arr[..., 1:-1]), dim=-1)
    
#     kinematics = {
#         'kin_vx_abs_max': aggregate(vx),
#         'kin_vy_abs_max': aggregate(vy),
#         'kin_speed_abs_max': aggregate(speed),
#         'kin_theta_abs_max': aggregate(theta),
#         'kin_ax_abs_max': aggregate(ax),
#         'kin_ay_abs_max': aggregate(ay),
#         'kin_a_lon_max': torch.amax(a_lon[..., 1:-1], dim=-1),
#         'kin_a_lon_min': torch.amin(a_lon[..., 1:-1], dim=-1),
#         'kin_a_lat_abs_max': aggregate(a_lat),
#         'kin_theta_dot_abs_max': aggregate(theta_dot),
#         'kin_theta_ddot_abs_max': aggregate(theta_ddot),
#         'kin_jerk_lon_abs_max': aggregate(jerk_lon),
#         'kin_jerk_mag_abs_max': aggregate(jerk_mag)
#     }
#     return kinematics

# def compute_kinematic_comfort(trajectories, dt=0.1):
#     """
#     Given a dictionary 'kinematics' containing maximum absolute values for each quantity
#     (with keys: 'a_lon_max', 'a_lat_max', 'theta_dot_max', 'theta_ddot_max',
#     'jerk_lon_max', and 'jerk_mag_max'), compute the fraction of modes that satisfy all the thresholds.
    
#     The kinematics values are assumed to have shape [batch_size, modes].
#     """
#     # NuPlan comfort thresholds
#     min_lon_accel    = -4.05  # m/s^2
#     max_lon_accel    =  2.40  # m/s^2
#     max_abs_lat_accel = 4.89  # m/s^2
#     max_abs_yaw_accel = 1.93  # rad/s^2
#     max_abs_yaw_rate  = 0.95  # rad/s
#     max_abs_lon_jerk  = 4.13  # m/s^3
#     max_abs_mag_jerk  = 8.37  # m/s^3

#     kinematics = compute_kinematics(trajectories)

#     # Check each condition for feasibility
#     cond_lon_accel = (kinematics['kin_a_lon_min'] >= min_lon_accel) & (kinematics['kin_a_lon_max'] <= max_lon_accel)
#     cond_lat_accel = torch.abs(kinematics['kin_a_lat_abs_max']) <= max_abs_lat_accel
#     cond_yaw_rate  = torch.abs(kinematics['kin_theta_dot_abs_max']) <= max_abs_yaw_rate
#     cond_yaw_accel = torch.abs(kinematics['kin_theta_ddot_abs_max']) <= max_abs_yaw_accel
#     cond_lon_jerk = torch.abs(kinematics['kin_jerk_lon_abs_max']) <= max_abs_lon_jerk
#     cond_jerk_mag = kinematics['kin_jerk_mag_abs_max'] <= max_abs_mag_jerk
    
#     # Combine all conditions; a mode is feasible if all conditions are met.
#     feasible_mask = (cond_lon_accel & cond_lat_accel & cond_yaw_rate &
#                     cond_yaw_accel & cond_lon_jerk & cond_jerk_mag)
#     # Compute the fraction of modes that are feasible for each sample in the batch.
#     fraction = torch.mean(feasible_mask.to(torch.float32), dim=1)
#     return fraction, kinematics


# def compute_kinematics_metrics(inputs, prediction):
#     trajectory = prediction['predicted_trajectory'][..., :2]
#     #last_pos = inputs["obj_trajs_last_pos"][:,0,:].unsqueeze(1).expand(-1, trajectory.shape[1], -1).unsqueeze(2)
#     #trajectory = torch.cat([last_pos, trajectory], dim=2)
#     kinematic_comfort, kinematics = compute_kinematic_comfort(trajectory) 
#     return {
#         "avgKinFeasibility": kinematic_comfort
#     }

# def compute_mean_nll(log_pdf, mask, mode="gmm"):
#     # Depending on the mode, combine the mixture components.
#     if mode.lower() in ['gmm', 'full']:
#         # Full GMM: sum the weighted PDFs.
#         log_mixture_pdf = torch.logsumexp(log_pdf, dim=1)  # (B, T)
#     elif mode.lower() in ['best', 'max']:
#         # Best component: take the maximum likelihood across mixture components.
#         log_mixture_pdf, _ = torch.max(log_pdf, dim=1)  # (B, T)
#     else:
#         raise ValueError("Unknown mode {}. Choose 'gmm' (or 'full') or 'best' (or 'max').".format(mode))

#     # Compute negative log-likelihood for each time step: -log(p)
#     nll = -log_mixture_pdf  # shape (B, T)
#     # Apply the ground-truth mask to consider only valid timesteps.
#     nll = nll * mask
#     # Average over valid timesteps for each batch.
#     valid_counts = torch.sum(mask, dim=1)  # shape (B,)
#     nll_per_sample = torch.sum(nll, dim=1) / valid_counts  # shape (B,)
#     return nll_per_sample



# def construct_covariance(sigma_x, sigma_y, rho):
#     """Construct covariance matrices."""
#     cov = torch.stack([
#         torch.stack([sigma_x**2, rho * sigma_x * sigma_y], dim=-1),
#         torch.stack([rho * sigma_x * sigma_y, sigma_y**2], dim=-1)
#     ], dim=-2)
#     return cov  # shape: [..., 2, 2]

# def wasserstein_similarity_pairwise(params, threshold=0.1, inflection=2.0):
#     """
#     Compute pairwise Wasserstein distances for batches of Gaussians.
    
#     params: Tensor of shape [B, num_gaussians, 5]
#             Each Gaussian: [mu_x, mu_y, sigma_x, sigma_y, rho]
#     Returns:
#         distances: Tensor of shape [B, num_gaussians, num_gaussians]
#     """
#     B, G, _ = params.shape
    
#     # Extract parameters
#     mu = params[..., :2]                           # [B, G, 2]
#     sigma_x = torch.exp(params[..., 2])                       # [B, G]
#     sigma_y = torch.exp(params[..., 3])                       # [B, G]
#     rho = torch.clip(params[..., 4] , -0.5, 0.5)                          # [B, G]

#     # Construct covariance matrices
#     cov = construct_covariance(sigma_x, sigma_y, rho)  # [B, G, 2, 2]

#     # Prepare tensors for pairwise operations
#     mu1 = mu.unsqueeze(2)          # [B, G, 1, 2]
#     mu2 = mu.unsqueeze(1)          # [B, 1, G, 2]
#     cov1 = cov.unsqueeze(2)        # [B, G, 1, 2, 2]
#     cov2 = cov.unsqueeze(1)        # [B, 1, G, 2, 2]

#     # Compute mean differences squared
#     mean_diff_sq = torch.sum((mu1 - mu2)**2, dim=-1)  # [B, G, G]

#     # Matrix square roots via eigen-decomposition for stability
#     def sqrtm_psd(mat):
#         with autocast(device_type="cuda" if mat.is_cuda else "cpu", enabled=False):
#             original_dtype = mat.dtype
#             eigvals, eigvecs = torch.linalg.eigh(mat.to(dtype=torch.float32))
#             eigvals_clamped = eigvals.clamp(min=1e-12).sqrt()
#             sqrtm = (eigvecs * eigvals_clamped.unsqueeze(-2)) @ eigvecs.transpose(-2, -1)
#         return sqrtm.to(original_dtype)

#     sqrt_cov1 = sqrtm_psd(cov1)  # [B, G, 1, 2, 2]

#     # Intermediate term
#     middle_term = sqrt_cov1 @ cov2 @ sqrt_cov1  # [B, G, G, 2, 2]
#     sqrt_middle_term = sqrtm_psd(middle_term)   # [B, G, G, 2, 2]

#     # Trace terms
#     trace_term = torch.diagonal(
#         cov1 + cov2 - 2 * sqrt_middle_term, dim1=-2, dim2=-1
#     ).sum(-1)  # [B, G, G]

#     # Wasserstein-2 distance
#     dist_squared = mean_diff_sq + trace_term.clamp(min=0)
#     distances = torch.sqrt(dist_squared)
#     distances = torch.relu(distances - threshold)
    
#     similarities = torch.exp(-(distances**2)/((inflection-threshold)**2)) 
#     return similarities # [B, G, G]

# def compute_wasserstein_diversity(similarities, temperature=1.0, eps=1e-8):
#     with autocast(device_type="cuda" if similarities.is_cuda else "cpu", enabled=False):
#         original_dtype=similarities.dtype
#         # Compute eigenvalues (guaranteed symmetric)
#         eigvals = torch.linalg.eigvalsh(similarities.to(torch.float32))  # [B, G]

#         # Ensure eigenvalues are non-negative (numerical stability)
#         eigvals = eigvals.clamp(min=eps)

#         # Normalize eigenvalues into probabilities
#         eigvals_normalized = eigvals / eigvals.sum(dim=1, keepdim=True)  # [B, G]

#         # Compute entropy
#         entropy = -torch.sum(eigvals_normalized * torch.log(eigvals_normalized + eps), dim=1)  # [B]

#         diversity = torch.exp(entropy)
#     return diversity.to(original_dtype)

# def bivariate_gaussian_entropy(sigma_x, sigma_y, rho):
#     det_cov = sigma_x**2 * sigma_y**2 * (1 - rho**2)
#     entropy = torch.log(2 * torch.pi * torch.e * torch.sqrt(det_cov))
#     return entropy

# def estimate_gmm_entropy(mu_x,
#                          mu_y,
#                          sigma_x,
#                          sigma_y,
#                          rho,
#                          log_pi,
#                          num_samples=10000):
#     """
#     Monte-Carlo approximation of entropy H(p) for a bivariate Gaussian mixture.
    
#     Args:
#         mu_x:      Tensor of shape (B, m), component means in x.
#         mu_y:      Tensor of shape (B, m), component means in y.
#         sigma_x:   Tensor of shape (B, m), positive stddevs in x.
#         sigma_y:   Tensor of shape (B, m), positive stddevs in y.
#         rho:       Tensor of shape (B, m), correlation coefficients (in [-1,1]).
#         log_pi:    Tensor of shape (B, m), log mixture-weights (unnormalized).
#         num_samples: int, number of MC samples per batch entry.
        
#     Returns:
#         entropy_est: Tensor of shape (B,), the estimated entropy per batch.
#     """
#     B, m = mu_x.shape

#     # Normalize mixture logits → weights π_k
#     pi = torch.softmax(log_pi, dim=-1)  # (B, m)

#     # Sample component indices according to π
#     mix_idx = torch.multinomial(pi, num_samples=num_samples, replacement=True)  # (B, num_samples)

#     # Gather parameters for each sample
#     mu_x_s = torch.gather(mu_x,    1, mix_idx)  # (B, S)
#     mu_y_s = torch.gather(mu_y,    1, mix_idx)
#     sx_s   = torch.gather(sigma_x,1, mix_idx)
#     sy_s   = torch.gather(sigma_y,1, mix_idx)
#     rho_s  = torch.gather(rho,    1, mix_idx)

#     # Draw base normals and construct correlated samples
#     u = torch.randn_like(mu_x_s)
#     v = torch.randn_like(mu_x_s)
#     x_s = mu_x_s + sx_s * u
#     y_s = mu_y_s + sy_s * (rho_s * u + torch.sqrt(1 - rho_s**2) * v)

#     # Prepare for density eval: expand back to (B, S, m)
#     x = x_s.unsqueeze(-1)             # (B, S, 1)
#     y = y_s.unsqueeze(-1)             # (B, S, 1)
#     mu_x_e  = mu_x.unsqueeze(1)       # (B, 1, m)
#     mu_y_e  = mu_y.unsqueeze(1)
#     sx_e    = sigma_x.unsqueeze(1)
#     sy_e    = sigma_y.unsqueeze(1)
#     rho_e   = rho.unsqueeze(1)
#     log_pi_e= log_pi.unsqueeze(1)

#     # Compute the exponent term of each component
#     zx = (x - mu_x_e) / sx_e
#     zy = (y - mu_y_e) / sy_e
#     denom = 1 - rho_e**2
#     exp_term = -0.5 * (zx**2 - 2*rho_e*zx*zy + zy**2) / denom

#     # Normalizing constant
#     norm_const = 2 * torch.pi * sx_e * sy_e * torch.sqrt(denom)
#     log_comp = exp_term - torch.log(norm_const)  # (B, S, m)

#     # Log-density of mixture via log-sum-exp
#     log_mix = torch.logsumexp(log_pi_e + log_comp, dim=-1)  # (B, S)

#     # MC estimate of entropy: H ≈ -E[log p(x)]
#     entropy_est = -log_mix.mean(dim=1)  # (B,)
#     return entropy_est

# def compute_probabilistic_metrics(inputs, prediction, future_required):
#     log_std_range=(-1.609, 5.0)

#     # Get ground-truth trajectory and mask. gt_traj shape: (B, T, D)
#     gt_traj = inputs['center_gt_trajs']
#     gt_traj_mask = inputs['center_gt_trajs_mask']
    
#     # Use only the first two dimensions (x,y) from ground truth. (B, T, 2)
#     gt_xy = gt_traj[..., :2]

#     # Get predicted mixture parameters.
#     predicted_traj = prediction['predicted_trajectory']  # shape (B, m, T, 5)
#     predicted_log_prob = prediction['predicted_log_probability'] # shape (B, m)

#     # Unpack the bivariate Gaussian parameters.
#     mu_x    = predicted_traj[..., 0]  # shape (B, m, T)
#     mu_y    = predicted_traj[..., 1]  # shape (B, m, T)
#     sigma_x = torch.exp(torch.clip(predicted_traj[..., 2], min=log_std_range[0], max=log_std_range[1]))  # shape (B, m, T)
#     sigma_y = torch.exp(torch.clip(predicted_traj[..., 3], min=log_std_range[0], max=log_std_range[1]))  # shape (B, m, T)
#     rho     = torch.clip(predicted_traj[..., 4], -0.5, 0.5)  # shape (B, m, T)

#     # Broadcast the predicted mixture weights to shape (B, m, T)
#     log_pi = predicted_log_prob.unsqueeze(-1).expand_as(mu_x)

#     eps = 1e-12  # Small constant for numerical stability

#     # Expand ground truth (x,y) to shape (B, 1, T, 2) for broadcasting over m.
#     gt_xy_expanded = gt_xy.unsqueeze(1)  # shape: (B, 1, T, 2)
#     x = gt_xy_expanded[..., 0]  # (B, 1, T)
#     y = gt_xy_expanded[..., 1]  # (B, 1, T)

#     # Calculate the log PDF for each Gaussian component using the bivariate Gaussian formula.
#     # Normalization factor: -log(2*pi*sigma_x*sigma_y*sqrt(1-rho^2))
#     norm = -torch.log(2 * torch.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2) + eps)  # (B, m, T)

#     # Compute standardized differences
#     dx = (x - mu_x) / (sigma_x + eps)  # (B, m, T)
#     dy = (y - mu_y) / (sigma_y + eps)  # (B, m, T)

#     # Compute the quadratic term in the exponent:
#     # dx^2 - 2*rho*dx*dy + dy^2
#     z = dx**2 - 2 * rho * dx * dy + dy**2
#     denom = 2 * (1 - rho**2) + eps

#     log_pdf = norm - (z / denom)  # (B, m, T)
#     # Multiply each component's PDF by its mixture weight.
#     log_weighted_pdf = log_pi + log_pdf  # (B, m, T)
#     mask = gt_traj_mask.float()  # (B, T)
#     mixtureNLL = compute_mean_nll(log_weighted_pdf, mask, mode="gmm")
#     minNLL = compute_mean_nll(log_pdf, mask, mode="best")
#     softmaxEntropy = -torch.sum(prediction["predicted_probability"] * prediction["predicted_log_probability"], dim=1)

#     wassersteins = wasserstein_similarity_pairwise(predicted_traj[:,:,future_required,:5])
#     diversity = compute_wasserstein_diversity(wassersteins)
#     sqrt_pi = torch.sqrt(prediction["predicted_probability"]).unsqueeze(-1) 
#     wassersteins_weighted = sqrt_pi * wassersteins * sqrt_pi.transpose(-2, -1) 
#     diversity_weighted = compute_wasserstein_diversity(wassersteins_weighted)
     
#     best_modes = torch.argmax(prediction["predicted_probability"], dim=1)
#     B=predicted_traj.shape[0]
#     best_sigma_x = sigma_x[torch.arange(B), best_modes, future_required] 
#     best_sigma_y = sigma_y[torch.arange(B), best_modes, future_required] 
#     best_rho = rho[torch.arange(B), best_modes, future_required] 
#     best_mode_entropy = bivariate_gaussian_entropy(best_sigma_x, best_sigma_y, best_rho)
#     mixture_entropy = estimate_gmm_entropy(mu_x[..., future_required], 
#                                         mu_y[..., future_required], 
#                                         sigma_x[..., future_required], 
#                                         sigma_y[..., future_required], 
#                                         rho[..., future_required], 
#                                         log_pi[..., future_required])

#     return {
#         "minNLL": minNLL, 
#         "mixtureNLL": mixtureNLL, 
#         "softmaxEntropy": softmaxEntropy, 
#         "minFEntropy_hat": best_mode_entropy,
#         "mixtureFEntropy_hat": mixture_entropy,
#         "spectralDiversity": diversity,
#         "weightedSpectralDiversity": diversity_weighted,
#         "avgWassersteinSimilarity": torch.mean(wassersteins, dim=(1,2))
#     }

# def compute_displacement_metrics(inputs, prediction, future_required):
#     gt_traj = inputs['center_gt_trajs'].unsqueeze(1)  # .transpose(0, 1).unsqueeze(0) #[A_tar, T, F]
#     gt_traj_mask = inputs['center_gt_trajs_mask'].unsqueeze(1)  #[A_tar, T]
#     center_gt_final_valid_idx = inputs['center_gt_final_valid_idx'] #[A_tar]

#     predicted_traj = prediction['predicted_trajectory']
#     predicted_prob = prediction['predicted_probability'].detach()
#     idx_best_mode = torch.argmax(predicted_prob, dim=1)
#     # Calculate ADE losses  #[A_tar, K, T, F]

#     ade_diff = torch.norm(predicted_traj[:, :, :, :2] - gt_traj[:, :, :, :2], 2, dim=-1)
#     bs, modes, future_len = ade_diff.shape

#     ade_losses = torch.sum(ade_diff * gt_traj_mask, dim=-1) / torch.sum(gt_traj_mask, dim=-1)
#     minade = ade_losses.gather(dim=1, index=idx_best_mode.unsqueeze((1))).squeeze(-1)
#     minade6 = torch.amin(ade_losses, dim=1)
#     # Calculate FDE losses
#     center_gt_final_required_idx = torch.full((bs, 1), future_required, device=ade_diff.device, dtype=torch.int64).view(-1, 1, 1).repeat(1, modes, 1)
#     center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, modes, 1).to(torch.int64)

#     fde_losses = torch.gather(ade_diff, dim=-1, index=center_gt_final_valid_idx).squeeze(-1)
#     minfde = fde_losses.gather(dim=1, index=idx_best_mode.unsqueeze(1)).squeeze(-1)         #minfde = fde_losses[np.arange(bs), idx_best_mode]
#     minfde6 = torch.amin(fde_losses, dim=-1)

#     fde_hat_losses = torch.gather(ade_diff, -1, center_gt_final_required_idx).squeeze(-1)
#     minfde_hat = fde_hat_losses.gather(dim=1, index=idx_best_mode.unsqueeze(1)).squeeze(-1)
#     minfde6_hat = torch.amin(fde_hat_losses, dim=-1)

#     best_fde_idx = torch.argmin(fde_losses, dim=-1)
#     predicted_prob = predicted_prob.gather(dim=1, index=best_fde_idx.unsqueeze(1)).squeeze(1) #predicted_prob[np.arange(bs), best_fde_idx]

#     miss_rate = (minfde6 > 2.0).float()
#     brier_fde6 = minfde6 + (1 - predicted_prob).pow(2)

#     return {
#         'minADE': minade,
#         'minFDE': minfde,
#         'minFDE_hat': minfde_hat,
#         'minADE6': minade6,
#         'minFDE6': minfde6,
#         'minFDE6_hat': minfde6_hat,
#         'miss_rate': miss_rate,
#         'brier_minFDE6': brier_fde6,
#     }
