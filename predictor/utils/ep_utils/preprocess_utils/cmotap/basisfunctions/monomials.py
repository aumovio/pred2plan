'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


import numpy as np

import predictor.utils.ep_utils.preprocess_utils.cmotap.interfaces.basisfunctions as interface


class Monomials(interface.BasisFunctions):
    """Monomial basis function"""

    def __init__(self, degree):
        """
        Parameters
        ----------
        degree: int
                maximum polynomial degree in basis, size of basis = degree + 1
        """
        self._degree = degree
        self._basisfunctions = [
            np.polynomial.Polynomial(np.eye(1, degree + 1, n).squeeze())
            for n in range(degree + 1)
        ]

    def _get(self, t, derivative):
        return np.row_stack([bf.deriv(derivative)(t) for bf in self._basisfunctions]).T
