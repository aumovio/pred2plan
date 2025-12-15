'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


class MotionModel(object):
    """Interface definition of a motion model"""

    def f(self, x, dt):
        """Takes object state x(t) and returns x(t+dt)"""
        return self._f(x, dt).squeeze()

    def F(self, x, dt):
        """Returns matrix linearized motion model x(t+dt) = F x(t)"""
        return self._F(x, dt)

    def Q(self, x=None, dt=None):
        """Returns process noise matrix Q for x(t) with time horizon dt"""
        return self._Q(x, dt)

    # To be implemented in derived class
    def _f(self, x, dt):
        raise NotImplementedError

    def _F(self, x, dt):
        raise NotImplementedError

    def _Q(self, x=None, dt=None):
        raise NotImplementedError
