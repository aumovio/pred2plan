'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''


import numpy as np
from scipy.optimize import linear_sum_assignment


def assign2D(L):
    row_ind, col_ind = linear_sum_assignment(L)
    cost = np.sum(L[row_ind, col_ind])

    col4row = -np.ones((L.shape[0], 1), dtype=int)
    row4col = -np.ones((L.shape[1], 1), dtype=int)
    col4row[row_ind, 0] = col_ind
    row4col[col_ind, 0] = row_ind

    return col4row, row4col, [cost]
