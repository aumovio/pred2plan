'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


import numpy as np

import .interfaces.motionmodel as interface


class ConstantVelocity(interface.MotionModel):
    """A simple 2D constant velocity point mass model"""

    def __init__(self, sigma=1):
        """Initializes the motion model

        Parameters
        -----------

        sigma: float
            Process noice standard deviation
        """
        self._sigma = sigma

    def _f(self, x, dt):
        """Returns f in x(t+dt) = f(x(t)), i.e. the non-linear motion model"""
        return self._F(x, dt) @ x

    def _F(self, x, dt):
        """Returns F in x(t+dt) = F x(t), i.e. the linearized motion model"""
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

    def _Q(self, x, dt):
        """Returns the process noise covariance"""
        return self._sigma**2 * np.array(
            [
                [dt**4 / 4, 0, dt**3 / 2, 0],
                [0, dt**4 / 4, 0, dt**3 / 2],
                [dt**3 / 2, 0, dt**2, 0],
                [0, dt**3 / 2, 0, dt**2],
            ]
        )
