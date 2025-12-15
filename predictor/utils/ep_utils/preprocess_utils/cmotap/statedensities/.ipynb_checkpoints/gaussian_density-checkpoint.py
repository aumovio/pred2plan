'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import utils.ep_utils.preprocess_utils.cmotap.interfaces.density as interface
import utils.ep_utils.preprocess_utils.cmotap.cmotap_utils as utils


class GaussianDensity(interface.Density):
    """A Gaussian State density"""

    def __init__(self, x=None, P=None):
        self._x = x
        self._P = P

    def __repr__(self):
        return "x:\n{}\nP:\n{}".format(self._x, self._P)

    @property
    def x(self):
        return self._x

    @property
    def P(self):
        return self._P

    @property
    def statedim(self):
        return self.x.size

    def predict(self, motion_model, dt):

        x_pred = motion_model.f(x=self.x, dt=dt)
        F = motion_model.F(x=self.x, dt=dt)
        P_pred = F @ self.P @ F.T + motion_model.Q(x=self.x, dt=dt)

        return self.__class__(x_pred, P_pred)

    def update(self, z, observation_model):

        Hx = observation_model.H(x=self.x)
        S = Hx @ self._P @ Hx.T + observation_model.R(x=self.x, z=z)
        S = (S + S.T) / 2

        K = np.linalg.lstsq(S, Hx @ self.P, rcond=None)[0].T
        # K = self.P @ Hx.T @ np.linalg.solve(S, np.eye(S.shape[0]))

        x_new = self.x + K @ (z - observation_model.h(x=self.x))
        P_new = (np.eye(x_new.size) - K @ Hx) @ self.P
        P_new = (P_new + P_new.T) / 2

        return self.__class__(x_new, P_new)

    def observationLogLikelihood(self, z, observation_model):
        Hx = observation_model.H(self.x)
        S = Hx @ self.P @ Hx.T + observation_model.R(self.x)
        S = (S + S.T) / 2

        return multivariate_normal.logpdf(z.T, mean=observation_model.h(self.x), cov=S)

    def ellipsoidal_gating(self, z, observation_model, gating_size):
        if np.ndim(z) < 2:
            z = np.expand_dims(z, -1)
        Hx = observation_model.H(self.x)
        S = Hx @ self.P @ Hx.T + observation_model.R(self.x)
        S = (S + S.T) / 2

        z_hat = np.expand_dims(observation_model.h(self.x), -1)
        nu = z - z_hat
        # dist = np.diag(nu.T @ np.linalg.solve(S, nu))
        dist = np.sum(nu * np.linalg.solve(S, nu), axis=0)

        meas_in_gate = dist <= gating_size
        z_in_gate = z[:, meas_in_gate]
        return z_in_gate, meas_in_gate

    @classmethod
    def from_mixture_density(cls, log_w, state_densities):
        w = np.exp(np.array(log_w))
        x_new = np.sum([w_i * c_i.x for (w_i, c_i) in zip(w, state_densities)], axis=0)
        P_new = np.sum(
            [
                w_i * (c_i.P + np.outer(c_i.x - x_new, c_i.x - x_new))
                for (w_i, c_i) in zip(w, state_densities)
            ],
            axis=0,
        )
        return cls(x_new, P_new)

    @classmethod
    def mixture_reduction(cls, log_w, state_densities, threshold=2):
        reduced_log_w = []
        reduced_state_densities = []

        to_reduce = state_densities
        to_reduce_log_w = np.array(log_w)
        z = np.column_stack([c.x for c in state_densities])
        to_merge = np.zeros(len(state_densities)).astype(bool)

        assert len(to_reduce) == to_reduce_log_w.size

        while len(to_reduce) > 0:
            z = z[:, ~to_merge].reshape((-1, len(to_reduce)))
            c0 = to_reduce[np.argmax(to_reduce_log_w)]
            x0 = np.expand_dims(c0.x, -1)

            dist = np.sum((z - x0) * np.linalg.solve(c0.P, z - x0), axis=0)
            to_merge = dist <= threshold

            relative_log_w, merged_weight = utils.normalize_logweights(
                to_reduce_log_w[to_merge]
            )

            reduced_state_densities.append(
                GaussianDensity.from_mixture_density(
                    relative_log_w, [c for i, c in enumerate(to_reduce) if to_merge[i]]
                )
            )
            reduced_log_w.append(merged_weight)

            to_reduce = [c for i, c in enumerate(to_reduce) if not to_merge[i]]
            to_reduce_log_w = to_reduce_log_w[~to_merge]
            assert len(to_reduce) == to_reduce_log_w.size

        return np.array(reduced_log_w), reduced_state_densities

    def draw(self, ax=None, **kwargs):
        """Draw the density into axis ax"""

        if ax is None:
            ax = plt.subplot(111)

        x_pos = self._x[kwargs.get("startdim", 0): kwargs.get("enddim", 2)]
        P_pos = self._P[
            kwargs.get("startdim", 0): kwargs.get("enddim", 2),
            kwargs.get("startdim", 0): kwargs.get("enddim", 2),
        ]

        utils.draw_confidence_ellipse(x_pos, P_pos, ax, **kwargs)

        return ax
