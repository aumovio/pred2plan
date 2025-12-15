'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


import numpy as np

import utils.ep_utils.preprocess_utils.cmotap.interfaces.motionmodel as interface


class TrajectoryMotion(interface.MotionModel):
    def __init__(self, trajectory, Q, Prior=None):
        """Returns an instance of MotionModel applicable to Trajectories

        Parameters:
        -----------
        trajectory: instance of type Trajectory

        Q: nd.array
        Process noise covariane

        Prior: nd.array [optional]
        Matrix of prior parameters
        """
        assert trajectory.statedim == Q.shape[0] == Q.shape[1]
        assert Prior is None or Q.shape == Prior.shape

        if Prior is None:
            self._invPrior = np.eye(Q.shape[0]) * 1e-8
        else:
            self._invPrior = np.linalg.solve(Prior, np.eye(Q.shape[0]))

        self._trajectory = trajectory
        self._Qmat = Q
        # a simple Least Recently Used (LRU) cache
        self._lru_cache = [None, None, None]

    def _make_F(self, x, dt):
        """Generates F in x(t+dt) = F x(t)"""
        T = self._trajectory.timescale
        basis = self._trajectory.basisfunctions
        spacedim = self._trajectory.spacedim
        t = np.linspace(0, T - dt, basis.size) / T
        tp = np.linspace(dt, T, basis.size) / T
        B = np.kron(basis.get(t), np.eye(spacedim))
        Bp = np.kron(basis.get(tp), np.eye(spacedim))
        return np.linalg.solve(B.T @ B + self._invPrior, B.T @ Bp)

    def _make_Q(self, x, dt):
        """Generates Q via F @ Q @ F.T"""
        return self._F(x, dt) @ self._Qmat @ self._F(x, dt).T

    def _update_lru(self, x, dt):
        """Updates the LRU cache - F before Q!"""
        self._lru_cache[0] = dt
        self._lru_cache[1] = self._make_F(x, dt)
        self._lru_cache[2] = self._make_Q(x, dt)

    def _f(self, x, dt):
        """Returns x(t+dt) = f(x(t)), i.e. the non-linear motion model"""
        # we shift the start point of the predicted trajectory to 0,0 before
        # applying the motion model in order for the zero mean prior to make sense
        x0 = np.kron(
            self._trajectory.basisfunctions.get(dt / self._trajectory.timescale),
            np.eye(self._trajectory.spacedim)
        ) @ x
        x0 = np.kron(np.ones(self._trajectory.basisfunctions.size), x0)
        return self._F(x, dt) @ (x - x0) + x0

    def _F(self, x, dt):
        """Returns the F in x(t+dt) = F x(t), i.e. the linearized motion model"""
        if self._lru_cache[0] != dt:
            self._update_lru(x, dt)
        return self._lru_cache[1]

    def _Q(self, x, dt):
        """Returns the process noise covariance"""
        if self._lru_cache[0] != dt:
            self._update_lru(x, dt)
        return self._lru_cache[2]
