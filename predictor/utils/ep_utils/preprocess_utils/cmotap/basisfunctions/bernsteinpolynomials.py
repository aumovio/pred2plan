'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


import numpy as np
from scipy.special import binom

from predictor.utils.ep_utils.preprocess_utils.cmotap.basisfunctions.monomials import Monomials


class BernsteinPolynomials(Monomials):
    """Bernstein Polynomial basis functions"""

    def __init__(self, degree):
        """
        Parameters
        ----------
        degree: int
                maximum polynomial degree in basis, size of basis = degree + 1
        """
        self._degree = degree
        # see https://en.wikipedia.org/wiki/Bernstein_polynomial
        self._basisfunctions = [
            np.polynomial.Polynomial(
                [
                    0 if i < k else (-1) ** (i - k) * binom(degree, i) * binom(i, k)
                    for i in range(degree + 1)
                ]
            )
            for k in range(0, degree + 1)
        ]
