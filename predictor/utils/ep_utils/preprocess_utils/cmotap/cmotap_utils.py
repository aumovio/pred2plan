'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import json
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from scipy.special import binom


def normalize_logweights(log_w):
    """Normalize log probabilities

    Parameters
    ----------
    log_w: list
        log probabilities

    Returns
    -------
        tuple of normalizedl log probabilities and log of sum of weights before normalization
    """
    if len(log_w) <= 1:
        log_sum_w = np.sum(log_w)
        return np.array(log_w) - log_sum_w, log_sum_w
    tmp = np.sort(log_w)[::-1]
    log_sum_w = tmp[0] + np.log(1 + np.sum(np.exp(tmp[1:] - tmp[0])))
    return log_w - log_sum_w, log_sum_w


def prune_multi_hypotheses(
    log_w, multi_hypotheses, threshold=np.log(1e-3), normalize_log_w=False
):
    """Remove mixture components of low probability

    Parameters
    ----------
    log_w: list
        log_probabilies of components

    multi_hypotheses: list
        components

    threshold:  float
        minimum log probability to keep component

    normalize_log_w: bool
        whether to normalize log probabilities after reduction
    """
    keep_indices = np.where(log_w >= threshold)[0]
    if not normalize_log_w:
        return (log_w[keep_indices], [multi_hypotheses[idx] for idx in keep_indices])
    else:
        return (
            normalize_logweights(log_w[keep_indices])[0],
            [multi_hypotheses[idx] for idx in keep_indices],
        )


def cap_multi_hypotheses(log_w, multi_hypotheses, M=100, normalize_log_w=False):
    """Limit the size of a mixture to only the M most probable components

    Parameters
    ----------
    log_w: list
        log_probabilies of components

    multi_hypotheses: list
        components

    M: int
        maximum number of components to keep

    normalize_log_w: bool
        whether to normalize log probabilities after reduction
    """
    log_w = np.array(log_w)
    keep_indices = np.argsort(log_w)[-M:]
    if not normalize_log_w:
        return (log_w[keep_indices], [multi_hypotheses[idx] for idx in keep_indices])
    else:
        return (
            normalize_logweights(log_w[keep_indices])[0],
            [multi_hypotheses[idx] for idx in keep_indices],
        )


def merge_multi_hypotheses(log_w, multi_hypotheses, threshold=2):
    """Reduce Mixture distribution by merging overlaps

    Parameters
    ----------
    log_w: list of float
        log probabilities of densities

    multi_hypotheses: list of densities

    threshold:
        threshold for densities to be joined

    Returns
    -------
        log_w, densities of reduced mixture
    """
    return multi_hypotheses[0].__class__.mixture_reduction(
        log_w, multi_hypotheses, threshold
    )


def get_mapextension(xy, expansion=(10, 10, 10, 10)):
    """Return extension of trajectory coordinates

    Parameters
    ----------
    xy: ndarray
        t x 2 array of trajectory coordinates
    extension: tuple
        dxmin, dxmax, dymin, dymax

    Returns
    -------
    ndarray of [xmin, xmax, ymin, ymax] to be passed to ax.axis to set axis limits
    """
    min_x, min_y = np.min(xy, axis=0)
    max_x, max_y = np.max(xy, axis=0)
    return [
        min_x - expansion[0],
        max_x + expansion[1],
        min_y - expansion[2],
        max_y + expansion[3],
    ]


def draw_confidence_ellipse(mean, cov, ax, **kwargs):
    """
    draws a 2d 95% confidence ellipse + centroid
    """
    assert mean.size == 2
    assert cov.shape == (2, 2)

    ax.plot(
        *mean,
        kwargs.get("marker", "x"),
        c=kwargs.get("color", "b"),
        alpha=kwargs.get("alpha", 1.0),
        ms=kwargs.get("markersize", 10),
        mew=kwargs.get("markeredgewidth", 2),
        label=kwargs.get("markerlabel", None)
    )

    d, U = np.linalg.eigh(cov)
    z = chi2.ppf(kwargs.get("confidence", 0.95), 2)  # 2 DoF
    # for a confidence level of 0.95 in 2d, z=5.991
    height, width = 2 * np.sqrt(z * d)
    angle = np.arctan2(U[1, -1], U[0, -1]) / np.pi * 180

    e = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=kwargs.get("linewidth", 2),
        alpha=kwargs.get("alpha", 1.0),
        color=kwargs.get("color", "b"),
    )
    ax.add_artist(e)

    return ax


def cartesian_noise_from_polar_noise(
    r, alpha, sigma_r, sigma_alpha, sigma_c, ego_heading
):
    """Generate a Cartesian Observation Noise covariance from a polar one

    Parameters
    ----------
    r: float
        distance between sensor and object

    alpha: float, radians
        bearing angle of object seen from sensor

    sigma_r: float
        standard deviation of r

    sigma_alpha: float
        standard deviation of alpha

    sigma_c: float
        standard deviation for timing jitter

    ego_heading: float, radians
        sensor heading in world coordinate system

    Returns
    -------
    R_ego: ndarray
        cartesian position noise covariance in sensor coordinates

    R_world: ndarray
        cartesian position noise covariance in world coordinates,
        i.e. rotated by ego_heading
    """

    s_xx = (sigma_r * np.cos(alpha)) ** 2 + (sigma_alpha * r * np.sin(alpha)) ** 2
    s_yy = (sigma_r * np.sin(alpha)) ** 2 + (sigma_alpha * r * np.cos(alpha)) ** 2
    s_xy = (sigma_r**2 - sigma_alpha**2 * r**2) * np.sin(alpha) * np.cos(alpha)

    R_ego = np.array([[s_xx, s_xy], [s_xy, s_yy]])

    Rot = np.array(
        [
            [np.cos(ego_heading), np.sin(ego_heading)],
            [-np.sin(ego_heading), np.cos(ego_heading)],
        ]
    )

    R_world = Rot.T @ (R_ego + sigma_c**2 * np.eye(2)) @ Rot

    return R_ego, R_world

def load_prior(filename, degree, timescale, spacedim=2):
    """Load Prior from Empirical Bayes Analysis
    
    Parameters
    ----------
    filename: str
        json file with prior
    degree: int
        Maximum degree of polynomial in Prior
    timescale: float
        time horizon of trajectories
    spacedim: int
        trajectory spatial dimension
    
    Returns
    -------
    futureprior, historyprior: ndarray
        prior matrices for trajectories starting at 0,0 or ending at 0,0, resp.
    """

    # The prior was calculatd for monomials and is
    # organized in the following way
    # x0x0 x0x1 x0x2 x0y0 x0y1 x0y2
    # x1x0 x1x1 x1x2 x1y0 x1y1 x1y2
    # x2x0 x2x1 x2x2 x2y0 x2y1 x2y2
    # y0x0 y0x1 y0x2 y0y0 y0y1 y0y2
    # y1x0 y1x1 y0x2 y1y0 y1y1 y1y2
    # y2x0 y2x1 y2x2 y2y0 y2y1 y2y2
    prior_data = json.load(open(filename, 'r'))

    # We reorganize the prior into the following structure
    # x0x0 x0y0 x0x1 x0y1 x0x2 x0y2
    # y0x0 y0y0 y0x1 y0y1 y0x2 y0y2
    # x1x0 x1y0 x1x1 x1y1 x1x2 x1y2
    # ....
    perm = np.zeros((2 * degree, 2 * degree))
    for i in range(degree):
        perm[2 * i, i] = 1
        perm[2 * i + 1, degree + i] = 1

    # all trajectories in EB analysis start at (0, 0), so there is little uncertainty about start point
    monomial_cov_unscaled = np.diag(np.block([0.7**2, 0.7**2, np.zeros(spacedim * degree)]))
    monomial_cov_unscaled[2:, 2:] = np.array(perm @ prior_data['A_list'][degree - 1] @ perm.T)
    
    monomial_scale = np.diag([timescale ** deg for deg in range(0, degree + 1)])
    monomial_scale = np.kron(monomial_scale, np.eye(spacedim))
    monomial_cov =  monomial_scale @ monomial_cov_unscaled @ monomial_scale.T
    
    monomial_mean = monomial_scale @ np.kron(np.zeros(degree + 1), np.ones(spacedim))
    
    # Now we transform to Bernstein Polynomials
    M = np.zeros((degree + 1, degree + 1))

    for k in range(0, degree + 1):
        for i in range(k, degree + 1):
            M[i, k] = (-1)**(i - k) * binom(degree, i) * binom(i, k)
        
    M_inv = np.linalg.solve(M, np.eye(degree + 1))
    future_prior = np.kron(M_inv, np.eye(2)) @ monomial_cov @ np.kron(M_inv, np.eye(2)).T
    
    # reorder entries for history
    perm = np.zeros((degree + 1, degree + 1))
    perm[np.arange(degree + 1), np.arange(degree, -1, -1)] = 1
    perm = np.kron(perm, np.eye(2))
    history_prior = perm @ future_prior @ perm.T
    
    return future_prior, history_prior
