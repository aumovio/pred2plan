"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import torch
import torchmetrics as tm
from typing import Optional, Union

class Kinematic(tm.Metric):
    """
    Mean comfortability per predicted trajectory (global over all seen B×M trajectories).

    update() expects:
      trajectories: Tensor [B, M, T, D] with (x, y) over time.
      position_idx: idx int, list or slice identifying the (x, y) dim
      dt: optional timestep override (float), defaults to self.dt

    compute() returns:
      scalar tensor in [0, 1]: (# comfortable trajectories) / (# total trajectories)
    """
    full_state_update = False

    def __init__(
        self,
        dt: float = 0.1,
        # NuPlan comfort thresholds
        min_lon_accel: float = -4.05,   # m/s^2
        max_lon_accel: float =  2.40,   # m/s^2
        max_abs_lat_accel: float = 4.89,# m/s^2
        max_abs_yaw_accel: float = 1.93,# rad/s^2
        max_abs_yaw_rate: float  = 0.95,# rad/s
        max_abs_lon_jerk: float  = 4.13,# m/s^3
        max_abs_mag_jerk: float  = 8.37 # m/s^3
    ):
        super().__init__()
        self.dt = float(dt)

        self.min_lon_accel     = float(min_lon_accel)
        self.max_lon_accel     = float(max_lon_accel)
        self.max_abs_lat_accel = float(max_abs_lat_accel)
        self.max_abs_yaw_accel = float(max_abs_yaw_accel)
        self.max_abs_yaw_rate  = float(max_abs_yaw_rate)
        self.max_abs_lon_jerk  = float(max_abs_lon_jerk)
        self.max_abs_mag_jerk  = float(max_abs_mag_jerk)

        # global counters across updates (DDP-safe)
        self.add_state("comfortable_traj", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total_traj",       default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    # ---------------- public API ----------------
    def compute_per_sample_values(self, 
        trajs: 
        torch.Tensor,
        position_idx: Union[int, slice, list[int]] = (0,2), 
        dt: float = None,
        **kwargs
        ) -> torch.Tensor:

        assert trajs.ndim == 4 and trajs.size(-1) >= 2, \
            f"Expected [B, M, T, >=2], got {tuple(trajs.shape)}"
        
        trajs = trajs[..., position_idx]

        dts = float(self.dt if dt is None else dt)

        feasible_mask = self._feasible_mask(trajs, dts).float().mean(dim=1)  # [B] float

        return feasible_mask

    @torch.no_grad()
    def update(self, trajs: torch.Tensor, position_idx: Union[int, slice, list[int]] = (0,2), dt: float = None, **kwargs):
        """
        trajs: [B, M, T, D] including (x, y)
        """
        feasible_mask = self.compute_per_sample_values(trajs, position_idx, dt, **kwargs)

        self.comfortable_traj += feasible_mask.sum()
        self.total_traj += feasible_mask.numel()

    def compute(self):
        return (self.comfortable_traj.float()
                / self.total_traj.clamp_min(1).float())

    # ---------------- internals ----------------

    @staticmethod
    @torch.no_grad()
    def _unwrap(theta: torch.Tensor) -> torch.Tensor:
        d = theta[..., 1:] - theta[..., :-1]
        d_mod = (d + torch.pi) % (2 * torch.pi) - torch.pi
        theta0 = theta[..., :1]
        return torch.cat([theta0, theta0 + torch.cumsum(d_mod, dim=-1)], dim=-1)

    @staticmethod
    @torch.no_grad()
    def _derivative(arr: torch.Tensor, dt: float) -> torch.Tensor:
        d_arr = torch.zeros_like(arr)
        d_arr[..., 1:-1] = (arr[..., 2:] - arr[..., :-2]) / (2 * dt)
        d_arr[..., 0]    = (arr[..., 1] - arr[..., 0])     / dt
        d_arr[..., -1]   = (arr[..., -1] - arr[..., -2])   / dt
        return d_arr

    @torch.no_grad()
    def _kinematics(self, trajectories: torch.Tensor, dt: float):
        # based on kinematic bicycle model
        # trajectories: [B, M, T, 2]
        x = trajectories[..., 0]
        y = trajectories[..., 1]

        vx = self._derivative(x, dt)
        vy = self._derivative(y, dt)
        speed = torch.sqrt(vx**2 + vy**2 + 1e-12)

        theta = self._unwrap(torch.atan2(vy, vx))

        ax = self._derivative(vx, dt)
        ay = self._derivative(vy, dt)

        # Decompose acceleration into body-fixed frame components:
        # longitudinal acceleration: along the heading direction
        # lateral acceleration: perpendicular to heading
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        a_lon = ax * cos_t + ay * sin_t
        a_lat = -ax * sin_t + ay * cos_t

        theta_dot  = self._derivative(theta, dt)
        theta_ddot = self._derivative(theta_dot, dt)

        jx = self._derivative(ax, dt)
        jy = self._derivative(ay, dt)
        jerk_mag = torch.sqrt(jx**2 + jy**2 + 1e-12)
        jerk_lon = self._derivative(a_lon, dt)

        # Aggregate the kinematics by taking the maximum absolute value along the horizon (axis=-1)
        def agg_abs_max(arr):
            # drop endpoints to reduce FD edge artifacts
            return torch.amax(torch.abs(arr[..., 1:-1]), dim=-1)

        kin = {
            "kin_vx_abs_max":         agg_abs_max(vx),
            "kin_vy_abs_max":         agg_abs_max(vy),
            "kin_speed_abs_max":      agg_abs_max(speed),
            "kin_theta_abs_max":      agg_abs_max(theta),
            "kin_ax_abs_max":         agg_abs_max(ax),
            "kin_ay_abs_max":         agg_abs_max(ay),
            "kin_a_lon_max":          torch.amax(a_lon[..., 1:-1], dim=-1),
            "kin_a_lon_min":          torch.amin(a_lon[..., 1:-1], dim=-1),
            "kin_a_lat_abs_max":      agg_abs_max(a_lat),
            "kin_theta_dot_abs_max":  agg_abs_max(theta_dot),
            "kin_theta_ddot_abs_max": agg_abs_max(theta_ddot),
            "kin_jerk_lon_abs_max":   agg_abs_max(jerk_lon),
            "kin_jerk_mag_abs_max":   agg_abs_max(jerk_mag),
        }
        return kin

    @torch.no_grad()
    def _feasible_mask(self, trajectories: torch.Tensor, dt: float):
        kin = self._kinematics(trajectories, dt)
        cond_lon_accel = (kin["kin_a_lon_min"] >= self.min_lon_accel) & \
                         (kin["kin_a_lon_max"] <= self.max_lon_accel)
        cond_lat_accel = kin["kin_a_lat_abs_max"].abs() <= self.max_abs_lat_accel
        cond_yaw_rate  = kin["kin_theta_dot_abs_max"].abs() <= self.max_abs_yaw_rate
        cond_yaw_accel = kin["kin_theta_ddot_abs_max"].abs() <= self.max_abs_yaw_accel
        cond_lon_jerk  = kin["kin_jerk_lon_abs_max"].abs()  <= self.max_abs_lon_jerk
        cond_jerk_mag  = kin["kin_jerk_mag_abs_max"]        <= self.max_abs_mag_jerk

        feasible = cond_lon_accel & cond_lat_accel & cond_yaw_rate & \
                   cond_yaw_accel & cond_lon_jerk & cond_jerk_mag    # [B, M]
        return feasible