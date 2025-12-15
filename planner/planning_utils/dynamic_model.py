"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import casadi as ca
from typing import Optional

class KinematicBicycle:
    """
    Kinematic bicycle model with longitudinal jerk and steering rate inputs in casadi.

    States (7):  [x, y, psi, vx, ax, delta, s]
    Controls (3):[jx, ddelta, vp]
    Dynamics:
        x_dot     = vx * cos(psi)
        y_dot     = vx * sin(psi)
        psi_dot   = vx * tan(delta) / L
        vx_dot    = ax
        ax_dot    = jx
        delta_dot = ddelta
        s_dot     = vp
    """
    def __init__(self, wheelbase: float):
        if wheelbase <= 0:
            raise ValueError("wheelbase must be positive.")
        self.wheelbase = wheelbase
        self.states = self.init_state_symbols()
        self.controls = self.init_control_symbols()
        self.rhs = self.init_rhs()

    def init_state_symbols(self) -> ca.MX:
        x     = ca.MX.sym('x')
        y     = ca.MX.sym('y')
        psi   = ca.MX.sym('psi')
        vx    = ca.MX.sym('vx')
        ax    = ca.MX.sym('ax')
        delta = ca.MX.sym('delta')
        s     = ca.MX.sym('s')
        states = ca.vertcat(x, y, psi, vx, ax, delta, s)
        return states

    def init_control_symbols(self) -> ca.MX:
        jx     = ca.MX.sym('jx')
        ddelta = ca.MX.sym('ddelta')
        vp     = ca.MX.sym('vp')
        controls = ca.vertcat(jx, ddelta, vp)
        return controls

    def init_rhs(self) -> ca.MX:
        X = self.states
        U = self.controls
        L = self.wheelbase

        x, y, psi, vx, ax, delta, s = [X[i] for i in range(7)]
        jx, ddelta, vp              = [U[i] for i in range(3)]
        tan_delta = ca.tan(delta)

        rhs = ca.vertcat(
            vx * ca.cos(psi),        # x_dot
            vx * ca.sin(psi),        # y_dot
            vx * tan_delta / L,      # psi_dot
            ax,                      # vx_dot
            jx,                      # ax_dot
            ddelta,                  # delta_dot
            vp                       # s_dot
        )
        return rhs
    
    def get_f(self, fname: Optional[str]="f", sname: Optional[str]="input_state", cname: Optional[str]="control_input", rhsname: Optional[str]="rhs") -> ca.Function:
        """
        Returns CasADi function f(states, controls) -> rhs with named I/O.
        """
        states  = self.states
        controls = self.controls
        rhs     = self.rhs

        f = ca.Function(
            fname,
            [states, controls],
            [rhs],
            [sname, cname],
            [rhsname]
        )
        return f