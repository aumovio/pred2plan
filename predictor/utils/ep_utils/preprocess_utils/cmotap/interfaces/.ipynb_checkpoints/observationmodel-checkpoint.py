'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


class ObservationModel(object):
    """Interface definition of an observation model"""

    def h(self, x):
        """Non-Linear observation model z(t) = h(x(t))"""
        return self._h(x).squeeze()

    def H(self, x=None):
        """Linearized observation model z(t) = H x(t)"""
        return self._H(x)

    def R(self, x=None, z=None):
        """Observation noise covariance when observing z(t) = h(x(t))"""
        return self._R(x=x, z=z)

    # To be implemented in derived class
    def _h(self, x):
        raise NotImplementedError

    def _H(self, x):
        raise NotImplementedError

    def _R(self, x=None, z=None):
        raise NotImplementedError
