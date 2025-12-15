'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import numpy as np
import matplotlib.pyplot as plt

from utils.ep_utils.preprocess_utils.cmotap.statedensities.gaussian_density import GaussianDensity
import utils.ep_utils.preprocess_utils.cmotap.cmotap_utils as utils

class GaussianPositionVelocityDensity2D(GaussianDensity):
    def __init__(self, x, P, spacedim=2):
        assert x.size % spacedim == 0
        self._spacedim = spacedim
        super(GaussianPositionVelocityDensity2D, self).__init__(x, P)

    @property
    def spacedim(self):
        return self._spacedim

    def compensate_sensor_movement(self, sensor_movement):
        """Currently on in 2D"""
        dx, dy, dtheta = sensor_movement
        # unit vectors in new ego frame
        ex = np.array([np.cos(dtheta), np.sin(dtheta)])
        ey = np.array([-np.sin(dtheta), np.cos(dtheta)])
        R1 = np.column_stack([ex, ey])
        R = np.kron(np.eye(self.x.size // self.spacedim), R1)

        dxy = np.array([dx, dy, 0, 0])

        x_new = (self.x - dxy) @ R
        P_new = R.T @ self.P @ R

        return GaussianPositionVelocityDensity2D(x_new, P_new)

    def draw(self, ax=None, **kwargs):
        """Draw the density into axis ax"""
        if ax is None:
            ax = plt.subplot(111)

        x_pos = self._x[:2]
        P_pos = self._P[:2, :2]

        x_vel = self._x[2:]
        # P_vel = self._P[2:, 2:]

        utils.draw_confidence_ellipse(x_pos, P_pos, ax, **kwargs)

        ax.arrow(
            *x_pos,
            *x_vel,
            edgecolor=kwargs.get("edgecolor", kwargs.get("color", "b")),
            facecolor=kwargs.get("facecolor", kwargs.get("color", "b"))
        )

        return ax
