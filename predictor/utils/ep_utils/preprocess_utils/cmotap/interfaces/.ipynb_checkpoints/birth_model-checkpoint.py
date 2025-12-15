'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


import numpy as np


class BirthModel(object):
    """A birthmodel for state densities"""

    def __init__(self, log_w, state_densities):
        """Initialize a birth model from a mixture model
        Parameters
        ----------
        log_w: list, array
               log probabilities of mixture components

        state_densities: list
               list of component densities
        """
        self._log_w = np.array(log_w)
        self._state_densities = state_densities

    def __repr__(self):
        return "log_w: {}\n{}".format(self.log_w, self.state_densities)

    @property
    def log_w(self):
        """log probabilities of mixture components"""
        return self._log_w

    @property
    def state_densities(self):
        """component densities"""
        return self._state_densities
