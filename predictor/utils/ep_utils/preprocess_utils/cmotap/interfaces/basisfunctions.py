'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


class BasisFunctions(object):
    """
    Interface definition of basis functions
    """

    def __init__(self):
        self._basisfunctions = []

    @property
    def basis_functions(self):
        """The list of basis functions"""
        return self._basisfunctions

    @property
    def size(self):
        """Number of basis functions"""
        return len(self._basisfunctions)

    def get(self, t, derivative=0):
        """Evaluates all basis functions or their derivative at t"""
        return self._get(t, derivative)

    def _get(self, t, derivative):
        """Evaluates basis functions or their derivatives at t"""
        raise NotImplementedError
