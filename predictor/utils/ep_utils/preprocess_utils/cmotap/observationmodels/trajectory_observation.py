'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import numpy as np
from scipy.linalg import block_diag

import predictor.utils.ep_utils.preprocess_utils.cmotap.interfaces.observationmodel as interface


class TrajectoryObservation(interface.ObservationModel):
    """Observation model for trajectories"""

    def __init__(self, trajectory, t, derivatives, R):
        """Returns an instance of ObservationModel for fixed time t

        Parameters
        -----------

        trajectory: instance of Trajectory to be observed

        t: float or ndarray
            the timepoint(s) 0 <= t <= timescale at which the trajectory is observed

        derivatives: list of int or list of list of int
            the derivatives to be calculated at every time point
            must be a list if more than one derivative is required for each timepoint

        R: np.array or list of ndarray
            the observation noise covariance for every timepoint
        """

        assert isinstance(t, (float, list, np.ndarray))
        if isinstance(t, float):
            t = np.array([t])
        assert np.all(0 <= t) and np.all(t <= trajectory.timescale)

        assert isinstance(derivatives, (int, list))
        if isinstance(derivatives, int):
            derivatives = [derivatives]
        if isinstance(derivatives, list):
            assert np.all([isinstance(d_at_t, (int, list)) for d_at_t in derivatives])
        if np.all([isinstance(d_at_t, int) for d_at_t in derivatives]):
            derivatives = [derivatives]

        assert isinstance(R, (np.ndarray, list))
        if isinstance(R, np.ndarray):
            R = [R]
        assert len(t) == len(derivatives) == len(R)

        assert np.all(
            len(d_t)
            == R_t.shape[0] * trajectory.spacedim
            == R_t.shape[1] * trajectory.spacedim
            for d_t, R_t in zip(derivatives, R)
        )

        def _H_at_t(t, derivatives):
            T = trajectory.timescale
            return np.row_stack(
                [
                    trajectory.basisfunctions.get(t / T, derivative=d) / T**d
                    for d in derivatives
                ]
            )

        self._Rmat = block_diag(*R)

        Hmat = np.row_stack(
            [_H_at_t(t_i, derivatives_i) for t_i, derivatives_i in zip(t, derivatives)]
        )

        self._Hmat = np.kron(Hmat, np.eye(trajectory.spacedim))

    def _R(self, x, z):
        """Returns the observation noise covariance"""
        return self._Rmat

    def _H(self, x):
        """Returns H from z(t) = H x(t), i.e. the linearized observation model"""
        return self._Hmat

    def _h(self, x):
        """Returns z(t) = h(x(t)), i.e. the non-linear observation model"""
        return self._H(x) @ x
