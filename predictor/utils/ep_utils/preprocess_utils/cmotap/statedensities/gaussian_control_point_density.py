'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import numpy as np
import matplotlib.pyplot as plt

from predictor.utils.ep_utils.preprocess_utils.cmotap.statedensities.gaussian_density import GaussianDensity
import predictor.utils.ep_utils.preprocess_utils.cmotap.cmotap_utils as utils


class GaussianControlPointDensity(GaussianDensity):
    def __init__(self, x, P, spacedim=2):
        assert x.size % spacedim == 0
        self._spacedim = spacedim
        super(GaussianControlPointDensity, self).__init__(x, P)

    @property
    def spacedim(self):
        """spatial dimension of state space"""
        return self._spacedim

    def compensate_sensor_movement(self, sensor_movement):
        """Compensate density for 2D sensor movement

        Parameters
        ----------
        sensor_movement: tuple (dx, dy, dphi)
            translation dx, dy and rotation dpih
            seen from the _old_ sensor pose

        Example:
            at time t, sensor pose is x=0, y=0, phi=pi / 4 in world coordinates
            at time t+dt, sensor pose is x=1, y=0, phi=pi / 2 in world coordinates
            then, sensormovement would be
            (np.sqrt(1/2), -np.sqrt(2), pi / 4)
        """
        dx, dy, dtheta = sensor_movement
        # unit vectors in new ego frame
        ex = np.array([np.cos(dtheta), np.sin(dtheta)])
        ey = np.array([-np.sin(dtheta), np.cos(dtheta)])
        R1 = np.column_stack([ex, ey])
        R = np.kron(np.eye(self.x.size // self.spacedim), R1)

        dxy = np.kron(np.ones(self.statedim // self.spacedim), np.array([dx, dy]))

        x_new = (self.x - dxy) @ R
        P_new = R.T @ self.P @ R

        return GaussianControlPointDensity(x_new, P_new)

    def draw(self, ax=None, **kwargs):
        """Draw the density into axis ax"""
        if ax is None:
            ax = plt.subplot(111)

        for k, cp in enumerate(self._x.reshape(-1, self.spacedim)):

            cov = self._P[
                k * self.spacedim: (k + 1) * self.spacedim,
                k * self.spacedim: (k + 1) * self.spacedim,
            ]

            utils.draw_confidence_ellipse(cp, cov, ax, **kwargs)
            kwargs.update({"markerlabel": None})
            kwargs.update({"confidencelabel": None})
        return ax
