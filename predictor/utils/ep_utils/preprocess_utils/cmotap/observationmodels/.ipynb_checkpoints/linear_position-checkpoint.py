'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import numpy as np

import utils.ep_utils.preprocess_utils.cmotap.interfaces.observationmodel as interface


class LinearPosition(interface.ObservationModel):
    """A simple linear observation model for position only with constant R"""

    def __init__(self, sigma):
        """
        Parameters:
        -----------
        sigma: float
            observation noise std_dev
        """
        self._sigma = sigma

    def _h(self, x):
        """Returns z(t) = h(x(t)), i.e. the non-linear observation model"""
        return self._H(x) @ x

    def _H(self, x):
        """Returns H from z(t) = H x(t), i.e. the linearized observation model"""
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    def _R(self, x, z):
        """Returns the observation noise covariance"""
        return self._sigma**2 * np.eye(2)
