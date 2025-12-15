'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import matplotlib.pyplot as plt
import numpy as np


from utils.ep_utils.preprocess_utils.cmotap.interfaces.basisfunctions import BasisFunctions
from utils.ep_utils.preprocess_utils.cmotap.statedensities.gaussian_control_point_density import (
    GaussianControlPointDensity,
)


class Trajectory(object):
    """An object describing a trajectory over a fixed time horizon"""

    def __init__(self, basisfunctions, spacedim, timescale):
        """constructs an object that models a trajectory
        over a fixed time horizon as a linear combination of basis functions

        Parameters
        ----------

        basisfunction: obj
            an instance of a BasisFunction object to be used

        spacedim: int
            spatial dimension of state space

        timescale: float
            the timescale (horizon) over which the trajectory is modeled
        """
        assert timescale > 0
        assert issubclass(basisfunctions.__class__, BasisFunctions)

        self._basisfunctions = basisfunctions
        self._spacedim = spacedim
        self._timescale = timescale

    @property
    def spacedim(self):
        """Spatial dimension of state space"""
        return self._spacedim

    @property
    def statedim(self):
        """Dimensionality of the state vector"""
        return self._basisfunctions.size * self._spacedim

    @property
    def timescale(self):
        """Timescale / timehorizon of the trajectory"""
        return self._timescale

    @property
    def basisfunctions(self):
        """Basisfunctions used in trajectory"""
        return self._basisfunctions

    def estimate(self, density, t, derivatives=0):
        """Estimate kinematic properties

        Parameters
        ----------

        density: GaussianControlPointDensity
            the state density for which the

        t: float
            the time along the trajectory at which to make the estimate

        derivatives: int or list of int
            the temporal derivatives to estimate
        """
        assert np.all(0 <= t) and np.all(t <= self.timescale)
        if not isinstance(derivatives, list):
            derivatives = [derivatives]

        T = self.timescale
        H = np.row_stack(
            [self.basisfunctions.get(t / T, derivative=d) / T**d for d in derivatives]
        )
        H = np.kron(H, np.eye(self.spacedim))

        return H @ density.x, H @ density.P @ H.T

    def draw(self, density, ax=None, **kwargs):
        """draws a the trajectory corresponding to the control point density"""
        if ax is None:
            ax = plt.subplot(111)

        t = np.linspace(0, self.timescale, 51)

        if kwargs.get("with_CI", False):
            density.draw(ax=ax, **kwargs)

        if kwargs.get("samples", 0):
            samples = np.random.multivariate_normal(
                mean=density.x, cov=density.P, size=kwargs.get("samples")
            )
            for sample in samples:
                dummy = GaussianControlPointDensity(x=sample, P=np.eye(sample.size))
                ss, _ = self.estimate(dummy, t, derivatives=0)
                ax.plot(*ss.reshape(-1, self.spacedim).T, "k--", alpha=0.5)

        mean_traj, _ = self.estimate(density, t, derivatives=0)

        ax.plot(
            *mean_traj.reshape(-1, 2).T,
            ls=kwargs.get("ls", "-"),
            c=kwargs.get("color", "b"),
            lw=2,
            alpha=kwargs.get("alpha", 1.0),
            label=kwargs.get("label", None)
        )
