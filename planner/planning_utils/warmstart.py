"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import numpy as np
from typing import Optional

class Warmstart:
    """
    Vectorized warm-start generator with consistent API and output.

    State (len=7): [x, y, psi, v, ax, delta, s]
      x,y:  world position
      psi:  heading [rad] 
      v:    speed [m/s]
      ax:   longitudinal acceleration [m/s^2]
      delta: steering [rad/s]
      s:    arc-length traveled

    Three different modes:
        - const_vx
        - const_ax
        - const-jx
    with delta = 0 and psi = constant

    Output: array shape (7, N+1), k=0..N
    """

    def __init__(self, mode: str = "const_ax"):
        self.mode = mode

    def generate(
        self,
        x0: np.ndarray,      # shape (7,)
        N: int,
        dt: float,
        vmax: Optional[float] = None,  # used by const_ax/const_jx
        amax: Optional[float] = None,  # used by const_jx
        jx: Optional[float] = 0.0,     # jerk for const_jx
    ) -> np.ndarray:
        """
        Generate warm-start with the selected mode.

        Returns:
            traj: np.ndarray, shape (7, N+1), columns are time steps k=0..N
        """
        if x0.shape[0] < 7:
            raise ValueError("x0 must be length 7: [x, y, psi, v, ax, delta, s].")

        if self.mode == "const_vx":
            return self._const_vx(x0, N, dt)
        elif self.mode == "const_ax":
            if vmax is None:
                raise ValueError("const_ax requires vmax.")
            return self._const_ax(x0, N, dt, vmax)
        elif self.mode == "const_jx":
            if vmax is None or amax is None:
                raise ValueError("const_jx requires vmax and amax.")
            return self._const_jx(x0, N, dt, vmax, amax, jx)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")

    # ---------------- internal, all return (7, N+1) ----------------

    @staticmethod
    def _straight_line_pose(
        x: float, y: float, psi: float, s_path: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project arc-length s_path onto world (x,y) with fixed heading psi."""
        c, s = np.cos(psi), np.sin(psi)
        xw = x + c * s_path
        yw = y + s * s_path
        psi_vec = np.full_like(s_path, psi, dtype=float)
        return xw, yw, psi_vec

    def _const_vx(self, x0: np.ndarray, N: int, dt: float) -> np.ndarray:
        x, y, psi, v0, ax0, delta0, s0 = x0
        # Velocity constant
        v = np.full(N + 1, v0, dtype=float)
        # Distance traveled (left Riemann sum): s_{k+1} = s_k + v_k dt
        s_incr = dt * np.concatenate(([0.0], np.cumsum(v[:-1])))
        xw, yw, psi_vec = self._straight_line_pose(x, y, psi, s_incr)

        traj = np.zeros((7, N + 1), dtype=float)
        traj[0, :] = xw
        traj[1, :] = yw
        traj[2, :] = psi_vec
        traj[3, :] = v
        traj[4, :] = np.full(N + 1, ax0, dtype=float)
        traj[5, :] = np.full(N + 1, delta0, dtype=float)
        traj[6, :] = s0 + s_incr
        return traj[:,1:]

    def _const_ax(self, x0: np.ndarray, N: int, dt: float, vmax: Optional[float]) -> np.ndarray:
        x, y, psi, v0, ax0, delta0, s0 = x0
        # Acceleration constant
        a = np.full(N + 1, ax0, dtype=float)
        # v_{k+1} = v_k + a_k dt => use cumulative sum of a[:-1]
        v_unclipped = v0 + dt * np.concatenate(([0.0], np.cumsum(a[:-1])))
        if vmax is not None:
            v = np.clip(v_unclipped, 0.0, vmax)
        else:
            v = np.maximum(v_unclipped, 0.0)

        s_incr = dt * np.concatenate(([0.0], np.cumsum(v[:-1])))
        xw, yw, psi_vec = self._straight_line_pose(x, y, psi, s_incr)

        # Optional: zero out accel exactly if we're pegged at bounds (keeps things neat)
        a_eff = a.copy()
        if vmax is not None:
            at_floor = v <= 1e-2
            at_ceiling = v >= vmax - 5e-2
            a_eff[at_floor | at_ceiling] = 0.0

        traj = np.zeros((7, N + 1), dtype=float)
        traj[0, :] = xw
        traj[1, :] = yw
        traj[2, :] = psi_vec
        traj[3, :] = v
        traj[4, :] = a_eff
        traj[5, :] = np.full(N + 1, delta0, dtype=float)
        traj[6, :] = s0 + s_incr
        return traj[:,1:]

    def _const_jx(
        self, x0: np.ndarray, N: int, dt: float, vmax: float, amax: float, jx: float
    ) -> np.ndarray:
        x, y, psi, v0, ax0, delta0, s0 = x0
        t = np.arange(N + 1, dtype=float) * dt

        # Acceleration increases linearly with jerk, capped
        a = np.minimum(ax0 + jx * t, amax)
        # Velocity via cumulative sum of a[:-1]
        v_unclipped = v0 + dt * np.concatenate(([0.0], np.cumsum(a[:-1])))
        v = np.clip(v_unclipped, 0.0, vmax)

        s_incr = dt * np.concatenate(([0.0], np.cumsum(v[:-1])))
        xw, yw, psi_vec = self._straight_line_pose(x, y, psi, s_incr)

        traj = np.zeros((7, N + 1), dtype=float)
        traj[0, :] = xw
        traj[1, :] = yw
        traj[2, :] = psi_vec
        traj[3, :] = v
        traj[4, :] = a
        traj[5, :] = np.full(N + 1, delta0, dtype=float)
        traj[6, :] = s0 + s_incr
        return traj[:,1:]