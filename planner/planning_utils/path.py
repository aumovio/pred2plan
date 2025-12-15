"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import autograd
import autograd.numpy as np
import functools
import inspect
import matplotlib.pyplot as plt
from matplotlib import animation, patches, rc, transforms
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import check_grad
from scipy import integrate
import casadi as ca

from . import spline


class Path(object):
    '''
    A Bezier Curve through `cp` providing derivatives with respect to path length
    '''
    def __init__(self, cp, N=101):
        self.cp =cp
        t = np.linspace(0, 1, N)
        degree = cp.shape[0] - 1

        self.S = spline.Spline(degree)
        S =self.S
        # x(t), y(t)
        self.x_ref, self.y_ref = S.from_control_points(cp, t=t).T
        # phi(t)
        self.phi = S.phi_from_control_points(cp, t=t)#% (2 * np.pi)
        # fix angle jump
        diffs = np.diff(self.phi)
        wrapped_diffs = (diffs + ca.pi) % (2 * ca.pi) - ca.pi
        self.phi= np.cumsum(np.concatenate(([self.phi[0]], wrapped_diffs)))
        
        # dx/dt, dy/dt
        self.dx, self.dy = S.tangent_from_control_points(cp, normalized=False, t=t).T
        
        
        self.kappa =  S.curvature_from_control_points(cp, t=t)
        # ds / dt
        #self.ds = S.ds_from_control_points(cp, t)
        # s(t)
        self.s = S.pathlength(cp, t)
        
        
        
        #self.dphi_ds = S.dphi_from_control_points(cp, t) / self.ds

        # dx/ds = dx / dt / (ds / dt)
        #self.dx_ref_ds = self.dx / self.ds
        # dy/ds = dy / dt / (ds / dt)
        #self.dy_ref_ds = self.dy / self.ds

        # d^2 phi / ds^2
        #self.dphi_sq_ds_sq = np.gradient(self.dphi_ds, self.s)
        # d^2 x_ref / ds^2
        #self.dx_ref_sq_ds_sq = np.gradient(self.dx_ref_ds, self.s)
        # d^2 y_ref / ds^2
        #self.dy_ref_sq_ds_sq = np.gradient(self.dy_ref_ds, self.s)

        super().__init__()

    # def get_theta(self, x):
    #     np.interp(x)

    def get_length(self):
        return self.s[-1]
    

    def get_phi(self, theta):
        #return np.interp(theta, self.s[:,1], self.phi)
        lut = ca.interpolant('LUT','linear',[self.s[:,1]], self.phi)
        return lut(theta)

    def get_x_ref(self, theta):
        #return np.interp(theta, self.s, self.x_ref)
        lut = ca.interpolant('LUT','linear',[self.s[:,1]], self.x_ref)
        return lut(theta)

    def get_y_ref(self, theta):
        #return np.interp(theta, self.s, self.y_ref)
        lut = ca.interpolant('LUT','linear',[self.s[:,1]], self.y_ref)
        return lut(theta)
    
    def get_kappa(self, theta):
        #return np.interp(theta, self.s, self.y_ref)
        lut = ca.interpolant('LUT','linear',[self.s[:,1]], self.kappa)#np.interp(theta, self.s[:,1], self.kappa) #ca.interpolant('LUT','linear',[self.s[:,1]], self.kappa)
        return lut(theta)   
 
"""
    def get_kappa(self, theta):
        #return np.interp(theta, self.s, self.y_ref)
        lut = np.interp(theta, self.s[:,1], self.kappa) #ca.interpolant('LUT','linear',[self.s[:,1]], self.kappa)
        return lut  
        
    def get_dphi_ds(self, theta):
        return np.interp(theta, self.s, self.dphi_ds)

    def get_dx_ref_ds(self, theta):
        return np.interp(theta, self.s, self.dx_ref_ds)

    def get_dy_ref_ds(self, theta):
        return np.interp(theta, self.s, self.dy_ref_ds)

    def get_dphi_sq_ds_sq(self, theta):
        return np.interp(theta, self.s, self.dphi_sq_ds_sq)

    def get_dx_ref_sq_ds_sq(self, theta):
        return np.interp(theta, self.s, self.dx_ref_sq_ds_sq)

    def get_dy_ref_sq_ds_sq(self, theta):
        return np.interp(theta, self.s, self.dy_ref_sq_ds_sq)"""


class Path_Constraint(object):
    '''
    A Bezier Curve through `cp` providing derivatives with respect to path length
    '''
    def __init__(self, cp, width = 2, N=101):

        t = np.linspace(0, 1, N)
        degree = cp.shape[0] - 1

        S = spline.Spline(degree)

        # x(t), y(t)
        self.x_ref, self.y_ref = S.from_control_points(cp, t=t).T + width *  S.normal_from_control_points(cp, t=t).T
        # phi(t)
        self.phi = S.phi_from_control_points(cp, t=t)
        # dx/dt, dy/dt
        self.dx, self.dy = S.tangent_from_control_points(cp, normalized=False, t=t).T

        # ds / dt
        #self.ds = S.ds_from_control_points(cp, t)
        # s(t)
        self.s = S.pathlength(cp, t)

        #self.dphi_ds = S.dphi_from_control_points(cp, t) / self.ds

        # dx/ds = dx / dt / (ds / dt)
        #self.dx_ref_ds = self.dx / self.ds
        # dy/ds = dy / dt / (ds / dt)
        #self.dy_ref_ds = self.dy / self.ds

        # d^2 phi / ds^2
        #self.dphi_sq_ds_sq = np.gradient(self.dphi_ds, self.s)
        # d^2 x_ref / ds^2
        #self.dx_ref_sq_ds_sq = np.gradient(self.dx_ref_ds, self.s)
        # d^2 y_ref / ds^2
        #self.dy_ref_sq_ds_sq = np.gradient(self.dy_ref_ds, self.s)

        super().__init__()

    # def get_theta(self, x):
    #     np.interp(x)

    def get_length(self):
        return self.s[-1]
    

    def get_phi(self, theta):
        #return np.interp(theta, self.s[:,1], self.phi)
        lut = ca.interpolant('LUT','linear',[self.s[:,1]], self.phi)
        return lut(theta)

    def get_x_ref(self, theta):
        #return np.interp(theta, self.s, self.x_ref)
        lut = ca.interpolant('LUT','linear',[self.s[:,1]], self.x_ref)
        return lut(theta)

    def get_y_ref(self, theta):
        #return np.interp(theta, self.s, self.y_ref)
        lut = ca.interpolant('LUT','linear',[self.s[:,1]], self.y_ref)
        return lut(theta)
    