'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


import numpy as np


class SensorModel(object):
    """A generic sensor model used in Poission Point Processes"""

    def __init__(self, P_D, lambda_c, range_c):
        """Initializes a Sensor Model

        Parameters
        ----------
        P_D: float
            Probability of detection

        lambda_c: float
            Mean number of clutter detections in sensor range

        range_c: nd.array
            [[xmin, xmax], [ymin, ymax]]
        """
        self._P_D = P_D
        self._lambda_c = lambda_c
        self._range_c = range_c
        self._pdf_c = 1 / np.prod(range_c[:, 1] - range_c[:, 0])
        self._intensity_c = lambda_c * self._pdf_c

    @property
    def P_D(self):
        """Probability of detection"""
        return self._P_D

    @property
    def intensity_c(self):
        """Clutter intensity, i.e. mean density of clutter per detection area"""
        return self._intensity_c

    @property
    def lambda_c(self):
        """Mean number of clutter detection over range"""
        return self._lambda_c

    @property
    def range_c(self):
        """Sensor range"""
        return self._range_c
