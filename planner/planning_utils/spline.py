"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import numpy as np
import scipy.linalg as LA
from scipy.optimize import fminbound

class BernsteinMatrix(object):
    def __init__(self, degree):
        self.degree = degree
        # a quick test if all works well
        t = np.linspace(0, 1, 101)
        assert np.linalg.norm(
            self.matrix_(t, self.degree) -
            self.Tmatrix_(t, self.degree) @ self.Mmatrix_(self.degree)
        ) < 1e-8
        assert np.linalg.norm(
            self.Fmatrix(0, degree) - np.eye(degree + 1)
        ) < 1e-8

    def binom(self, n, k):
        return np.prod([
            (n - j) / (j + 1)
            for j in range(0, min(k, n - k))
        ])

    def matrix_(self, t, degree=None):
        # Calculate Bernstein Matrix,
        # column j is nchoosek(degree, j) * t^j * (1-t)^(degree - j)
        if degree is None:
            degree = self.degree
        return np.array([
            self.binom(degree, j) * (1 - t)**(degree - j) * t ** j
            for j in range(degree + 1)
        ]).T

    def Tmatrix_(self, t, degree=None, derivative=0):
        '''
        the matrix of basis polynomials
        '''
        if np.ndim(t) < 1:
            t = [t]
        if isinstance(t, list):
            t = np.array(t)
        if degree is None:
            degree = self.degree

        factor = lambda dn, n: np.prod(range(n, n-dn, -1))

        return np.array([
            factor(derivative, n) * t**(n - derivative) if n >= derivative else 0 * t
            for n in range(degree + 1)
        ]).T

    def Dmatrix_(self, dt, degree=None):
        '''
        The \Delta matrix needed for the calculation of the advancement along the curve
        '''
        if degree is None:
            degree = self.degree
        return np.array([
            [(
                self.binom(j, i)  * dt**(j - i)
            ) if j >= i else 0 for j in range(degree + 1)
            ] for i in range(degree + 1)
        ])

    def Mmatrix_(self, degree=None):
        '''
        see https://interval.louisiana.edu/reliable-computing-journal/volume-17/reliable-computing-17-pp-40-71.pdf
        equation (8)
        '''
        if degree is None:
            degree = self.degree
        return np.array([
            [(
                self.binom(degree, j) * self.binom(degree - j, i - j) *
                (-1)**((i + j) % 2)
            ) if j <= i else 0 for j in range(degree + 1)
            ] for i in range(degree + 1)
        ])

    def Smatrices_(self, tp, degree=None):
        '''
        The returns the matrices used to split a Bezier
        '''
        assert 0 <= tp and tp <= 1
        if degree is None:
            degree = self.degree
        M = self.Mmatrix_(degree)
        Sprime = np.diag([tp**n for n in range(degree + 1)])
        # since M is triangular, we can solve much easier
        S1 = LA.solve_triangular(M, Sprime @ M, lower=True)
        Sprime = np.diag([(1-tp)**n for n in range(degree + 1)])
        # M * Pi would reverse the row orders of M so we can do that directly
        # Similarly Pi^-1 can be done directly with flipup
        # but now, the flipped M is in the 
        S2 = np.flipud(LA.solve_triangular(M, Sprime @ np.fliplr(M), lower=True))
        return S1, S2

    def Fmatrix(self, dt, degree=None):
        if degree is None:
            degree = self.degree
        M = self.Mmatrix_(degree)
        # since M is triangular, we can solve much easier
        return LA.solve_triangular(M, self.Dmatrix_(dt, degree) @ M, lower=True)

    def __call__(self, t, derivative=0):
        return (
            self.Tmatrix_(t, self.degree, derivative=derivative) @
            self.Mmatrix_(self.degree)
        )


class Spline(object):
    def __init__(self, degree=3):
        self.degree = degree
        self.Bernstein = BernsteinMatrix(degree)

    def Fmatrix(self, dt):
        return self.Bernstein.Fmatrix(dt, degree=self.degree)

    def from_control_points(self, cp, t=None, N=101):
        '''
        Generate a spline from control points, each ROW of cp is one point
        '''
        if t is None:
            t = np.linspace(0, 1, N)
        degree = cp.shape[0] - 1
        B = self.Bernstein(t)
        return B @ cp

    def tangent_from_control_points(self, cp, t=None, normalized=True, N=101):
        if t is None:
            t = np.linspace(0, 1, N)
        degree = cp.shape[0] - 1
        Bprime = self.Bernstein(t, derivative=1)
        T = Bprime @ cp
        if normalized:
            return T / np.linalg.norm(T, axis=-1, keepdims=True)
        else:
            return T

    
    
    def phi_from_control_points(self, cp, t=None, N=101):
        assert cp.shape[1] == 2
        dx, dy = self.tangent_from_control_points(cp, t=t, N=N).T
        return np.arctan2(dy, dx)

    def dphi_from_control_points(self, cp, t=None, N=101):

        assert cp.shape[1] == 2
        if t is None:
            t = np.linspace(0, 1, N)
        dx_dt, dy_dt = self.tangent_from_control_points(cp, normalized=False, t=t).T

        B_dtt = self.Bernstein(t, derivative=2)
        dx_sq_dt_sq, dy_sq_dt_sq = (B_dtt @ cp).T

        return (dx_dt * dy_sq_dt_sq  -dy_dt * dx_sq_dt_sq) / (dx_dt**2 + dy_dt**2)

    
    def curvature_from_control_points(self, cp, t=None, N=101):
        #added by moh
        assert cp.shape[1] == 2
        dx, dy = self.tangent_from_control_points(cp, t=t, N=N).T
        v = np.sqrt(dx**2 +dy**2)
        dphi_dt = self.dphi_from_control_points( cp, t=t, N=N)
        kappa = dphi_dt*v #np.absolute((self.Bernstein(t, derivative=1) @ cp)* (self.Bernstein(t, derivative=2)@ cp))/(np.norm?(self.Bernstein(t, derivative=1)@ cp))**3
        return kappa
    
    def pathlength(self, cp, t=None, N=101):
        '''
        return an Nx2 matrix of t and correponding s
        '''
        if t is None:
            t = np.linspace(0, 1, N)
        tangent = self.tangent_from_control_points(cp, t, normalized=False)
        s = np.cumsum(
            np.linalg.norm(tangent, axis=1) * np.block([0, np.diff(t)])
        )
        return np.column_stack([t, s])

    def split(self, cp, ts):
        S1, S2 = self.Bernstein.Smatrices_(ts)
        return S1 @ cp, S2 @ cp

    def extend(self, cp, x_new, weight):
        if np.ndim(x_new) < 2:
            x_new = x_new.reshape(1, -1)
        assert cp.shape[1] == x_new.shape[1]
        assert 0 < weight and weight <= 1
        return self.find_control_points(
            np.row_stack([
                self.from_control_points(cp, N=int(1 / weight - 0.5)),
                x_new
            ])
        )

    def normal_from_control_points(self, cp, t=None, N=101):
        assert cp.shape[1] in [2, 3]
        if cp.shape[1] > 2:
            rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        else:
            rot = np.array([[0, 1], [-1, 0]])
        degree = cp.shape[0] - 1
        return self.tangent_from_control_points(cp @ rot, t=t, N=N)

    def rational_from_control_points(self, cp, t=None, N=101):
        '''
        Generate a rational spline from control points,
        each ROW of cp is one point, weights in the last column
        '''
        degree = cp.shape[0] - 1
        S = self.from_control_points(cp, t, N)
        return S / S[:, -1].reshape(-1, 1)

    def find_control_points(self, D, t=None, W=None, m0=None, S0=None, reg=1e-8):
        '''
        Estimate Control points from data, assuming equidistant t if not given
        D is an N x d of N sample points in d dimensions
        t is an N vector
        W is an d x N Matrix of weights for each data point, e.g. use 1/sigma
        m0 is an n x d matrix, representing the prior mean of the control points
        S0 is a d x (n x d) Tensor representing the prior covariance of the control points
        '''
        if t is None:
            t = np.linspace(0, 1, D.shape[0])

        B = self.Bernstein(t)

        if W is None:
            W = np.ones((D.shape[1], B.shape[0]))

        if m0 is None:
            m0 = np.zeros((B.shape[1], D.shape[1]))
            #m0 = np.tile(np.mean(D, axis=0), (B.shape[1], 1))

        if S0 is None:
            S0 = [1 / reg * np.eye(B.shape[1])] * D.shape[1]

        S0_inv = [np.linalg.solve(S, np.eye(S.shape[0])) for S in S0]

        # Bayesian least squares
        return np.column_stack([
            np.linalg.solve(
                (B.T * W[d]) @ B + S0_inv[d],
                S0_inv[d] @ m0[:, d] + (B.T * W[d]) @ D[:, d]
            )
            for d in range(D.shape[1])
        ])
    
    
    def find_control_points_from_derivative(self, D, t=None, W=None, m0=None, S0=None, reg=1e-8):
        '''
        Estimate Control points from data, assuming equidistant t if not given
        D is an N x d of N sample points in d dimensions
        t is an N vector
        W is an d x N Matrix of weights for each data point, e.g. use 1/sigma
        m0 is an n x d matrix, representing the prior mean of the control points
        S0 is a d x (n x d) Tensor representing the prior covariance of the control points
        '''
        if t is None:
            t = np.linspace(0, 1, D.shape[0])

        B = self.Bernstein(derivative=2,t=t)

        if W is None:
            W = np.ones((D.shape[1], B.shape[0]))

        if m0 is None:
            m0 = np.zeros((B.shape[1], D.shape[1]))
            #m0 = np.tile(np.mean(D, axis=0), (B.shape[1], 1))

        if S0 is None:
            S0 = [1 / reg * np.eye(B.shape[1])] * D.shape[1]

        S0_inv = [np.linalg.solve(S, np.eye(S.shape[0])) for S in S0]

        # Bayesian least squares
        return np.column_stack([
            np.linalg.solve(
                (B.T * W[d]) @ B + S0_inv[d],
                S0_inv[d] @ m0[:, d] + (B.T * W[d]) @ D[:, d]
            )
            for d in range(D.shape[1])
        ])
    
    def find_control_points_with_derivatives(self, D, t=None, W=None, m0=None, S0=None, reg=1e-8):
        '''
        Estimate Control points from data, assuming equidistant t if not given
        D is an N x d of N sample points in d dimensions
        t is an N vector
        W is an d x N Matrix of weights for each data point, e.g. use 1/sigma
        m0 is an n x d matrix, representing the prior mean of the control points
        S0 is a d x (n x d) Tensor representing the prior covariance of the control points
        '''
        if t is None:
            t = np.linspace(0, 1, D.shape[0])

        B0 = self.Bernstein(derivative=0,t=t)
        B1 = self.Bernstein(derivative=1,t=t)
        B2 = self.Bernstein(derivative=2,t=t)
        B=  np.concatenate((B0, B1, B2), axis=0)

        if W is None:
            W = np.ones((D.shape[1], B.shape[0]))

        if m0 is None:
            m0 = np.zeros((B.shape[1], D.shape[1]))
            #m0 = np.tile(np.mean(D, axis=0), (B.shape[1], 1))

        if S0 is None:
            S0 = [1 / reg * np.eye(B.shape[1])] * D.shape[1]

        S0_inv = [np.linalg.solve(S, np.eye(S.shape[0])) for S in S0]

        # Bayesian least squares
        return np.column_stack([
            np.linalg.solve(
                (B.T * W[d]) @ B + S0_inv[d],
                S0_inv[d] @ m0[:, d] + (B.T * W[d]) @ D[:, d]
            )
            for d in range(D.shape[1])
        ])

    def initial_guess(self, D):
        t = np.linalg.norm(D[1:, :] - D[:-1, :], axis=1)
        t = np.cumsum(t)
        t = np.block([0, t]) / t[-1]
        return t

    def affine_met(self, D):
        Xcov = np.cov(D.T)
        V = D[1:, :] - D[:-1, :]
        t = np.diag(V @ np.linalg.solve(Xcov, V.T))
        t = np.block([0, np.cumsum(t)])
        t /= t[-1]
        return t

    def nearest_point_and_distance(self, point, cp, N=1001):
        t = np.linspace(0, 1, N)
        ps = self.from_control_points(cp, t=t)
        s =self.pathlength(cp, t=t)
        dist = np.linalg.norm(ps - point, axis=1)
        idx = np.argmin(dist)
        return s[idx, 1], ps[idx, :], dist[idx], t[idx]

    def initial_guess_from_control_points(self, D, cp, t=None, N=1001):
        '''
        returns an estimate for t from given control points
        '''
        if t is None:
            t = np.linspace(0, 1, N)
        spline = self.from_control_points(cp, t)
        idx = [
            np.argmin(np.linalg.norm(spline - point, axis=1))
            for point in D[:]
        ]
        return (t[idx] - np.min(t[idx])) / (np.max(t[idx]) - np.min(t[idx]) + 1e-8)

    def residual(self, D, t, error_only=False):
        B = self.Bernstein(t)
        if error_only:
            Q, _, = LA.qr(B)
            Q2 = Q[:, self.degree + 1:]
            r = Q2.T @ D
            return np.sum(r**2)
        else:
            Q, R, E = LA.qr(B, mode='full', pivoting=True)

            # The paper uses \Pi, a permutation matrix, but it's easier to use
            # the permuation vector returned by scipy.linalg.qr
            # make sure we have the right permutation
            # assert np.allclose(B[:, E], Q @ R)
            # PI = np.zeros((E.size, E.size))
            # PI[E, np.arange(E.size)] = 1
            # make sure we've set up the permutation matrix correctly
            # assert np.allclose(B @ PI, Q @ R)

            Q2 = Q[:, self.degree + 1:] # m x (m - degree - 1)

            r = Q2.T @ D      # residual vector

        return np.sum(r**2), r, Q, R, E, Q2

    def borgespastva(self, D, t0=None, k=3, maxiter=100):
        '''
        adapted from
        https://www.mathworks.com/matlabcentral/fileexchange/46406-borgespastva-m
        See Borges & Pastva, "Total Least Squares Fitting of Bezier and B-Spline
        Curves to Ordered Data", Computer Aided Geometric Design, 2001
        http://hdl.handle.net/10945/38066
        '''
        if t0 is None:
            t0 = self.initial_guess(D)

        # remove first and last sample, we'll fix those to 0, and 1
        t = t0[1:-1]
        errors = []
        err, r, Q, R, E, Q2 = self.residual(D, np.block([0, t, 1]))
        errors.append(err)
        converged = False

        for i in range(maxiter):

            r = Q2 @ r # residual vector Equation 7

            R = R[:k + 1, :]    # (degree + 1) x (degree + 1)
            Q1 = Q[:, :k + 1] # m x degree + 1

            olderr = err

            DB = self.Bernstein(t, derivative=1)

            #P = (DB @ PI) @ np.linalg.solve(R, Q1.T) # equation 10
            P = DB[:, E] @ np.linalg.solve(R, Q1.T) # equation 10
            dd = P @ D
            Q2p = Q2[1:-1, :].T

            F = np.vstack([
                Q2 @ (Q2p @ np.diag(dd[:, d])) + P.T @ np.diag(r[1:-1, 0])
                for d in range(dd.shape[1])
            ]) # equation 10

            delta_t = np.linalg.lstsq(
                F, r.T.reshape(-1, 1), rcond=None
            )[0].flatten()

            alpha = np.min(
                np.max(np.array([t / delta_t, -(1 - t) / delta_t]), axis=0)
            )
            if alpha <=1:
                delta_t = delta_t * alpha * 0.95

            new_t = t + delta_t

            err, r, Q, R, E, Q2 = self.residual(D, np.block([0, new_t, 1]))

            if err < olderr:
                # error decreases, so we take the full step
                t = new_t
            else:
                alpha = fminbound(
                    lambda x: np.sum(self.residual(
                        D, np.block([0, t + x * delta_t, 1]), error_only=True
                    )**2),
                    0, 1
                )
                t = t + alpha * delta_t
                err, r, Q, R, E, Q2 = self.residual(D, np.block([0, t, 1]))

            errors.append(err)

            # Check for convergence
            rel_err = np.abs(olderr - err) / olderr
            if rel_err < 1e-8 or err < 1e-5:
                converged = True
                break

        return np.block([0, t, 1]), converged, errors