'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


class Density(object):
    """Interface definition for a state density"""

    def __repr__(self):
        raise NotImplementedError

    def predict(self, motion_model, dt=None):
        """Returns a predicted state Density"""
        raise NotImplementedError

    def update(self, z, observation_model):
        """Returns an updated state density"""
        raise NotImplementedError

    def observationLogLikelihood(self, z, observation_model):
        """Return the likelihood of observing z under observation_model"""
        raise NotImplementedError

    def ellipsoidal_gating(self, z, observation_model, gating_size):
        """Returns whether z is likely to be associated as an observation"""
        raise NotImplementedError

    @classmethod
    def from_mixture_density(cls, log_w, state_densities):
        """Generates a density by merging a mixture density"""
        raise NotImplementedError

    @classmethod
    def mixture_reduction(cls, log_w, state_densities, threshold=2):
        """Reduces a mixture distribution to fewer components"""
        raise NotImplementedError

    def draw(self, ax=None, **kwargs):
        """Draws the density"""
        raise NotImplementedError

    def compensate_sensor_movement(self, sensor_movement):
        """Compensates the density for sensor movement"""
        raise NotImplementedError
