"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
#!/usr/bin/env python
import logging
import subprocess
import casadi as ca
import numpy as np
from abc import ABC
from typing import List
from pathlib import Path

from planner.planning_utils.warmstart import Warmstart
from planner.planning_utils.dynamic_model import KinematicBicycle

from nuplan.common.actor_state.ego_state import  EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.trajectory.interpolated_trajectory import  InterpolatedTrajectory

logger = logging.getLogger(__name__)

class BaseMPCC(ABC):
    def __init__(self, time, constraints, controller, reference, solver, verbose):
        self.config = {
            "time": time,
            "constraints": constraints,
            "controller": controller,
            "reference": reference,
            "solver": solver,
            "verbose": verbose,
        }
        self.dynamics = KinematicBicycle(wheelbase=3.083)
        self.warmstarter = Warmstart(mode="const_ax")
        self.flattened_params = time | constraints | controller | reference
        self.verbose = verbose
        if self.verbose: logger.info(f"Initialize {self.__class__.__name__}...")
        self._set_time(time)
        self._set_constraints(constraints)
        self._set_controller(controller)
        self._set_reference_signal(reference)
        self._set_optimization_solver(solver)
        if self.verbose: logger.info(f"Initialize {self.__class__.__name__} configuration...  DONE!")
        self.setup_MPC()
        self._fail_streak = 0
        self._emergency_mode = False
        if self.verbose: logger.info(f"Initialize {self.__class__.__name__}... DONE!")

    def setup_MPC(self):
        """
        Assemble the full nonlinear program defining the model-predictive contouring controller.

        The setup procedure constructs all components required by the MPC:
        system and scenario-tree structure, decision and parameter variables,
        reference-path data, stage and terminal costs, simple box constraints,
        nonlinear constraints such as dynamics and feasibility limits, and
        finally the IPOPT solver that will handle the resulting optimization.

        This method organizes the one-time symbolic assembly of the predictive
        problem; after setup, the solver can be called efficiently with updated
        parameters for each planning cycle.

        Returns
        -------
        None
            All symbolic problem elements and the solver instance are stored
            as attributes of the MPC object.
        """
        # time spent in function can increase if jit is activated
        self.init_system_model()
        self.init_decision_variables()
        self.init_parameter_variables()
        self.init_path_refpoints()
        self.init_costs()
        self.init_box_constraints() 
        self.init_nonlinear_constraints() 
        self.init_ipopt_solver()
        
    def __str__(self):
        return f"{self.__class__!s} with initialization configuration {self.config}"

    def _set_time(self, cfg):
        self.N = cfg["N"]
        self.dt = cfg["dt"]

    def _set_constraints(self, cfg):
        # box constraints
        self.slack_scale = cfg["box_scale"]
        # vehicle geometry
        self.wheelbase = cfg["wheelbase"]
        self.rob_r = cfg["rob_r"]
        self.front_length = cfg["front_length"]
        self.rear_length = cfg["rear_length"]
        self.length = cfg["length"]
        self.width = cfg["width"]
        # dynamical model
        self.v_max = cfg["v_max"]
        self.a_max = cfg["a_max"]
        self.a_min = cfg["a_min"]
        self.delta_max = cfg["delta_max"]
        self.ddelta_max = cfg["ddelta_max"]
        self.j_max = cfg["j_max"]
        self.ay_max = cfg["ay_max"]
        self.lane_width = cfg["lane_width"]

        self.lbg, self.ubg = None, None
        self.lbx, self.ubx = None, None
        self.lamx0_, self.lamg0_ = None, None
        self.x0_, self.u0_ = None, None

    def _set_controller(self, cfg):
        self.q_a = cfg["q_a"] # cost on positive acceleration
        self.q_yaw = cfg["q_yaw"] # cost on yaw acceleration
        self.q_c = cfg["q_c"] # cost on contouring lateral error to reference path
        self.q_l = cfg["q_l"] # cost on contouring lag error to reference path
        self.q_v = cfg["q_v"] # reward on velocity along progress
        self.q_ho = cfg["q_ho"] # cost weight collision avoidance for hard constraints
        self.q_so = cfg["q_so"] # cost weight collision avoidance for soft constraints
        self.q_ps = cfg["q_ps"] # cost weight on path slack
        self.q_lb = cfg["q_lb"] # cost weight road boundary avoidance
        self.q_na = cfg["q_na"] # non-anticipativity soft constraint cost
        self.q_jx = cfg["q_jx"] # costs on positive jerk
        self.Q = ca.diagcat(*cfg["Q"]) # quadratic costs on state reference track error
        self.R = ca.diagcat(*cfg["R"]) # quadratic costs on controls
        self.decay = cfg["decay"] # decay parameter

        self.S = cfg["S"]
        self.N_tree = self.N+1+(self.N-1)*(self.S-1)
        self.max_N_control = int(self.N/2)
        self.num_hard_obs = cfg["MAX_HARD_OBS"]
        self.num_soft_obs = cfg["MAX_SOFT_OBS"]
        self.modes = cfg["MODES"]
        self.n_pred_params = 3

        self.cost = 0
        self.f = None
        self.U = None
        self.P = None
        self.X = None
        self.g = []

    def _set_reference_signal(self, cfg):
        self.M = cfg["M"]
        self.s_final = cfg["s_final"]

        self.X_ref = []
        self.Y_ref = []
        self.Psi_ref = []
        self.W_l = []
        self.W_r = []

    def _set_optimization_solver(self, cfg):
        self.optimization_solver_opts = cfg
        self.solver = None

    def init_system_model(self):
        self.f = self.dynamics.get_f()
        self.n_states = self.dynamics.states.numel()
        self.n_controls = self.dynamics.controls.numel()

    def _kth_node(self, s: int, t: int) -> int:
        # x0,x1 are shared; thereafter blocks of (N-1) per scenario
        return t if t <= 1 else 2 + s * (self.N - 1) + (t - 2)
        
    def _kth_edge(self, s: int, t: int) -> int:
        # u0 is shared; thereafter blocks of (N-1) per scenario, starting at t=1
        return 0 if t == 0 else 1 + (t - 1) + (self.N - 1) * s 
        
    def iter_tree_nodes(self, w):
        """Yield (s, t) for all valid state nodes x_t^(s)."""
        # root (shared)
        yield 0, 0, 1.0 # x0
        yield 0, 1, 1.0 # x1
        # scenario branches
        for s in range(0, self.S):  
            for t in range(2, self.N + 1): 
                yield s, t, w[s]

    def iter_tree_edges(self, w):
        """Yield (s, t) for all valid control edges u_t^(s)."""
        # root (shared)
        yield 0, 0, 1.0 # u0
        # scenario branches
        for s in range(0, self.S): 
            for t in range(1, self.N): 
                yield s, t, w[s]

    def init_ipopt_solver(self):
        """
        Initialize the IPOPT nonlinear programming solver for the assembled MPCC problem.

        The solver is constructed from the symbolic nonlinear program consisting of
        the objective function, decision variables, parameters, and the full set of
        equality and inequality constraints. Solver options supplied during class
        configuration are passed directly to IPOPT, allowing control over tolerances,
        linear solvers, and runtime behavior.

        Once created, the solver instance can be called repeatedly with updated
        parameters for warm-started or receding-horizon optimization.

        Returns
        -------
        None
            The method stores the resulting solver object internally and resets any
            cached optimal trajectories.
        """
        self.x_opt = None
        self.u_opt = None
        opts_setting = self.optimization_solver_opts
        nlp_prob = {'f': self.cost, 'x': self.opt_variables, 'p': self.P, 'g': self.g}
        # self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        file_root = "./planner/compiles/"
        binary_name = f"{self.__class__.__name__!s}_{self.S}.so"
        binary_file = Path(file_root)/binary_name
        binary_file = Path(binary_name)
        if opts_setting["jit"]: # just jit compile
            self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        elif binary_file.exists(): # load precompiled binary
            self.solver = ca.nlpsol("solver", "ipopt", binary_file.as_posix(), opts_setting)
        else: # generate compiled binary
            solver = ca.nlpsol("solver","ipopt",nlp_prob)
            solver.generate_dependencies(binary_file.with_suffix(".c").as_posix())
            cmd_args = ["gcc", "-fPIC", "-shared", opts_setting["jit_options"]["flags"], str(binary_file.with_suffix(".c")), "-o", binary_file.as_posix()]
            subprocess.run(cmd_args)
            self.solver = ca.nlpsol("solver", "ipopt", binary_file.as_posix(), opts_setting)
        if self.verbose: logger.info(f"Initialize {self.__class__.__name__} optimization solver ... DONE!")

    def add_state_box_constraints(self, lbx, ubx):
        """
        Add simple box constraints for all state variables across the scenario tree.

        Each state component is bounded independently to enforce physical or modeling
        limits: position and heading are left unbounded, velocity is restricted to a
        feasible range, longitudinal acceleration is kept within braking and traction
        limits, and the steering angle is clipped to the actuator limits. These bounds
        are replicated for every node in the scenario tree.

        A tighter upper bound on velocity is applied near the beginning of the horizon
        to ensure conservative speed behavior immediately after branching, while later
        nodes revert to the nominal velocity limit.

        Parameters
        ----------
        lbx : dict
            Lower bounds for decision variables. The key `"states"` is populated
            with replicated bounds for each state at each node.
        ubx : dict
            Upper bounds for decision variables. The key `"states"` is populated
            with replicated bounds for each state at each node, with early-horizon
            velocity upper bounds selectively reduced.

        Returns
        -------
        tuple
            Updated `(lbx, ubx)` dictionaries including box constraints on all
            state variables along the prediction horizon.
        """
        X_lbx = ca.DM([-ca.inf, -ca.inf  , -ca.inf, -0, self.a_min, -self.delta_max, 0])
        X_ubx = ca.vertcat(ca.inf , ca.inf  , ca.inf, self.v_max_sym, self.a_max, self.delta_max, ca.inf)
        lbx["states"] = ca.repmat(X_lbx, 1, self.N_nodes) 
        ubx["states"] = ca.repmat(X_ubx, 1, self.N_nodes) 

        v_idx = 3
        for s, t, _ in self.iter_tree_nodes(self.parameters["scenario_weights"]):
            if t <= 4:
               k_node = self._kth_node(s, t)
               ubx["states"][v_idx, k_node] = self.v_max
        
        return lbx, ubx

    def add_control_box_constraints(self, lbx, ubx):
        """
        Add simple box constraints for all control inputs across every scenario–time edge.

        The control vector typically contains longitudinal jerk, steering-rate, and a
        commanded speed or acceleration-like term. Each component is bounded independently:
        jerk and steering-rate limits restrict how quickly the vehicle can change its
        longitudinal or lateral motion, and the final component is capped to prevent
        unrealistic commanded speeds. The bounds are repeated for all edges in the
        scenario tree.

        Parameters
        ----------
        lbx : dict
            Dictionary of lower bounds for decision variables; the key `"controls"`
            is populated with replicated lower bounds for every control instance.
        ubx : dict
            Dictionary of upper bounds for decision variables; the key `"controls"`
            is populated with replicated upper bounds for every control instance.

        Returns
        -------
        tuple
            Updated `(lbx, ubx)` dictionaries including box constraints on all
        control variables along the prediction horizon.
        """
        U_lbx = ca.DM([-2*self.j_max, - self.ddelta_max, 0])
        U_ubx = ca.vertcat(self.j_max, self.ddelta_max, 1.5*self.v_max_sym)
        lbx["controls"] = ca.repmat(U_lbx, 1, self.N_edges) 
        ubx["controls"] = ca.repmat(U_ubx, 1, self.N_edges) 

        return lbx, ubx

    def add_edge_costs(self, cost):
        """
        Add running (edge) costs for each scenario–time transition in the prediction horizon.

        For every edge in the scenario tree, the stage cost combines a quadratic penalty
        on control inputs with an additional term penalizing yaw acceleration. The control
        penalty, `uᵀ R u`, regularizes steering and acceleration commands, while the
        yaw-acceleration term discourages aggressive steering changes that would induce
        high rotational dynamics. A small linear term on the final control component
        can bias the optimizer toward maintaining speed when appropriate.

        Each edge cost is scaled by the scenario probability or weight associated
        with that branch, ensuring that the expected running cost is accumulated
        across the scenario tree.

        Parameters
        ----------
        cost : casadi.MX
            Current accumulated cost to be augmented by the running costs defined
            along all scenario–time edges.

        Returns
        -------
        casadi.MX
            Updated cost including weighted running costs for all edges.
        """
        scenario_weights = self.parameters["scenario_weights"]
        X, U = self.decision_variables["X"], self.decision_variables["U"]

        for s, t, w in self.iter_tree_edges(scenario_weights):
            k = self._kth_node(s,t)
            xk  = X[:, k]
            uk  = U[:, self._kth_edge(s, t)]

            edge_cost = ca.mtimes([uk.T, self.R, uk]) # quadratic cost on control effort
            edge_cost -= self.q_v * ca.log(uk[-1] + 1)/ca.log(2)  # reward on progress velocity
            edge_cost += self.q_jx *((1/10.0) * ca.log1p(ca.exp(10.0 * (uk[0]-0.5)))) # cost on positive jerk
            edge_cost += self.q_yaw * ca.sumsqr((xk[4] * ca.tan(xk[5]) + xk[3] * (1/ca.cos(xk[5])**2) * uk[1]) / self.wheelbase) # cost on yaw acceleration
            # running_cost += self.q_yaw * ca.sumsqr((xk[3] * ca.tan(xk[5])) / self.wheelbase) # cost on yaw_rate
            # running_cost += self.q_yaw * ca.sumsqr((xk[4] * ca.tan(xk[5]) + xk[3] * (1/ca.cos(xk[5])**2) * uk[1]) / self.wheelbase) # cost on yaw acceleration
            # running_cost += self.q_yaw * ca.sumsqr((2.0 * xk[3] * xk[4] * ca.tan(xk[5]) + (xk[3]**2) * (1.0/ca.cos(xk[5])**2) * uk[1]) / self.wheelbase) # cost on lateral jerk
            cost += w * edge_cost
        
        return cost

    def add_equality_constraints(self, g, lbg, ubg):
        """
        Add initial-condition and system-dynamics equality constraints for all
        scenario branches of the predictive horizon.

        The initial-condition constraint fixes the state at the first node to the
        supplied initial state parameter. For each scenario–time edge, the system
        dynamics are enforced through a discrete forward Euler step:
        the next state must equal the current state plus the time-step–scaled
        dynamics evaluated at that state and its corresponding control input.

        These constraints ensure that every scenario branch evolves consistently
        with the model `f(x, u)` and that branching only occurs where prescribed by
        the scenario tree structure.

        Parameters
        ----------
        g : dict
            Dictionary to which equality constraint expressions are appended.
            The keys `"ic_eq"` and `"dynamics_eq"` are added or overwritten.
        lbg : dict
            Dictionary of lower bounds corresponding to each constraint block.
            Equality constraints are given zero lower bounds.
        ubg : dict
            Dictionary of upper bounds corresponding to each constraint block.
            Equality constraints are given zero upper bounds.

        Returns
        -------
        tuple
            Updated `(g, lbg, ubg)` dictionaries including initial-condition and
            dynamics equality constraints and their bounds.
        """
        # initial condition equality
        X = self.decision_variables["X"]
        U = self.decision_variables["U"]
        g["ic_eq"] = X[:, 0] - self.parameters["x_init"]
        lbg["ic_eq"] = ca.DM.zeros(self.n_states, 1)
        ubg["ic_eq"] = ca.DM.zeros(self.n_states, 1)

        dyn_model_constraints = []
        # TODO: exchange scenario_weights
        scenario_weights = self.parameters["scenario_weights"]

        for s, t, _w in self.iter_tree_edges(scenario_weights):
            kx  = self._kth_node(s, t)       # x_t^(s)
            kx_next = self._kth_node(s, t + 1)   # x_{t+1}^(s)
            ku  = self._kth_edge(s, t)       # u_t^(s)

            x_next = X[:, kx] + self.f(X[:, kx], U[:, ku]) * self.dt
            dyn_model_constraints.append(X[:, kx_next] - x_next)
    
        g["dynamics_eq"] = ca.vertcat(*dyn_model_constraints)
        nrows = g["dynamics_eq"].shape[0]
        z = ca.DM.zeros(nrows, 1)
        lbg["dynamics_eq"] = z
        ubg["dynamics_eq"] = z
        
        return g, lbg, ubg

    def add_gg_diamond_constraints(self, g, lbg, ubg):
        """
        Add tire force (g–g diagram) inequality constraints for all scenario–time nodes.

        The constraint enforces that the combined longitudinal and lateral
        accelerations of the ego vehicle stay within an approximated g–g
        capability region. This region is modeled as a convex “diamond”
        described by four linear half-spaces, coupling longitudinal
        acceleration `a_x` with lateral acceleration `a_y`.

        At each scenario–time node, the acceleration components are computed
        from the vehicle state: `a_x` is a decision variable and `a_y`
        derives from bicycle-model geometry through the steering angle.
        The resulting inequalities constrain the feasible acceleration set
        to respect the vehicle’s maximum longitudinal (`a_max`) and lateral
        (`ay_max`) limits as well as its minimum braking capability (`a_min`).

        Parameters
        ----------
        g : dict
            Dictionary of constraint expressions to which the g–g inequalities
            will be appended under the key `"gg_ineq"`.
        lbg : dict
            Dictionary of lower bounds for each constraint block. The g–g
            inequalities are unbounded below.
        ubg : dict
            Dictionary of upper bounds for each constraint block. The g–g
            inequalities are enforced as ≤ 0 conditions.

        Returns
        -------
        tuple
            Updated `(g, lbg, ubg)` dictionaries including the g–g diagram
            constraints and their corresponding bounds.
        """
        scenario_weights = self.parameters["scenario_weights"]

        gg_constraints = []
        for s, t, _ in self.iter_tree_nodes(scenario_weights):
            k = self._kth_node(s, t)        # column in X for x_t^(s)

            vx    = self.decision_variables["X"][3, k]
            delta = self.decision_variables["X"][5, k]
            a_x   = self.decision_variables["X"][4, k]
            a_y   = vx * vx * ca.tan(delta) / self.wheelbase

            # four half-spaces of the g–g “diamond”
            gg_constraints += [
                a_x  + (self.a_max * a_y / self.ay_max) - self.a_max,
                a_x  - (self.a_max * a_y / self.ay_max) - self.a_max,
                -a_x + (self.a_min * a_y / self.ay_max) + self.a_min,
                -a_x - (self.a_min * a_y / self.ay_max) + self.a_min,
            ]
        
        g["gg_ineq"] = ca.vertcat(*gg_constraints)
        nrows = g["gg_ineq"].shape[0]
        lbg["gg_ineq"] = -ca.inf * ca.DM.ones(nrows, 1)
        ubg["gg_ineq"] = ca.DM.zeros(nrows, 1)

        return g, lbg, ubg

    def add_nonanticipativity_costs(self, cost):
        """
        Add soft non-anticipativity costs to the objective function for a scenario-based MPCC.

        Non-anticipativity enforces that control inputs across multiple scenario branches
        remain identical up to the point where scenarios diverge. Here, the constraint is
        imposed softly by penalizing deviations between each scenario’s control input and
        the reference branch (scenario index 0).

        For each time step `t` where the control is still shared among scenarios,
        a weighted quadratic penalty is added for the difference in control actions.

        Parameters
        ----------
        cost : casadi.MX
            Current accumulated objective function value to which the non-anticipativity
            penalties will be added.

        Returns
        -------
        casadi.MX
            Updated objective function including soft non-anticipativity penalty terms.
        """
        na_w = self.parameters["non_anticipatory_weights"]
        U = self.decision_variables["U"]

        for t in range(1, self.max_N_control): # iterate over bounded controls
            w_t = na_w[t - 1]
            u_ref = U[:, self._kth_edge(0, t)]       # reference branch is s=0
            for s in range(1, self.S):                    # compare s=1..S-1 to s=0
                u_s = U[:, self._kth_edge(s, t)]
                diff = u_s - u_ref
                cost += w_t * self.q_na * ca.sumsqr(diff)

        return cost     

    def check_road_boundary(self, w):
        """
        Ensure consistent representation of the road boundary width across all modes.

        If `w` is a scalar or a single value, it is broadcast into a vector of length `M`,
        representing a uniform boundary width for all trajectory branches or modes.
        If `w` is already a vector or array, it is returned unchanged.

        Parameters
        ----------
        w : float, int, or array-like
            Road boundary width or widths, possibly scalar.

        Returns
        -------
        numpy.ndarray or casadi.DM
            Boundary width vector of shape (M,), representing the lateral boundary at each mode.
        """
        if isinstance(w, (int, float)) or len(w)<1:
            return w*np.ones((self.M,))
        else:
            return w
   
    def get_e_c(self,  px, py, x_ref, y_ref, phi):
        """
        Compute the contouring (lateral) error between the vehicle position and the reference path.

        The contouring error `e_c` measures how far the vehicle has deviated laterally from
        the path centerline — that is, perpendicular to the path’s heading direction.

        Parameters
        ----------
        px : float or casadi.MX
            Vehicle x-position in world coordinates.
        py : float or casadi.MX
            Vehicle y-position in world coordinates.
        x_ref : float or casadi.MX
            Reference x-position along the path.
        y_ref : float or casadi.MX
            Reference y-position along the path.
        phi : float or casadi.MX
            Reference path heading angle (orientation).

        Returns
        -------
        casadi.MX
            Contouring error: lateral deviation from the path centerline.
        """
        e_c = -ca.sin(phi) * (px - x_ref) + ca.cos(phi) * (py - y_ref)
        return e_c

    def get_e_l(self, px, py, x_ref, y_ref, phi):
        """
        Compute the lag (longitudinal) error between the vehicle position and the reference path.

        The lag error `e_l` measures the deviation along the path’s tangent direction — 
        that is, how far ahead or behind the vehicle is relative to the reference point 
        projected onto the path.

        Parameters
        ----------
        px : float or casadi.MX
            Vehicle x-position in world coordinates.
        py : float or casadi.MX
            Vehicle y-position in world coordinates.
        x_ref : float or casadi.MX
            Reference x-position along the path.
        y_ref : float or casadi.MX
            Reference y-position along the path.
        phi : float or casadi.MX
            Reference path heading angle (orientation).

        Returns
        -------
        casadi.MX
            Lag error: longitudinal deviation along the path direction.
        """
        e_l =  ca.cos(phi) * (px - x_ref) + ca.sin(phi) * (py - y_ref)
        return e_l
        
    def get_contouring_costs(self, x, x_ref, y_ref, psi_ref, q_c, q_l):
        """
        Compute the contouring and lag cost for the vehicle with respect to a reference path.

        The contouring cost penalizes the **lateral deviation** (orthogonal error) from the 
        reference path, while the lag cost penalizes the **longitudinal deviation** 
        (distance along the path) between the vehicle's current position and its 
        projection onto the reference.

        Parameters
        ----------
        x : casadi.MX or casadi.DM
            Current vehicle state vector [px, py, ...].
        x_ref : float
            Reference x-position on the path at the corresponding timestep.
        y_ref : float
            Reference y-position on the path at the corresponding timestep.
        psi_ref : float
            Reference heading (orientation) on the path at the corresponding timestep.
        q_c : float or casadi.MX
            Weight for contouring (lateral) error.
        q_l : float or casadi.MX
            Weight for lag (longitudinal) error.

        Returns
        -------
        casadi.MX
            Scalar contouring and lag cost term: q_c * e_c² + q_l * e_l².
        """
        px, py= x[0], x[1]
        e_c = self.get_e_c(px, py, x_ref, y_ref, psi_ref)
        e_l = self.get_e_l(px, py, x_ref, y_ref, psi_ref)
        return ca.mtimes([e_c, q_c, e_c]) + ca.mtimes([e_l, q_l, e_l]) 

    def get_lane_costs(self, x, x_ref, y_ref, psi_ref):
        """
        Compute the lane potential field cost, modeling soft repulsion from adjacent lanes.

        The potential field introduces exponential "walls" centered at multiple lane 
        midlines on both sides of the reference trajectory. The effect is to softly 
        penalize large lateral deviations from the lane center, discouraging lane departures 
        while still allowing flexibility for overtaking or obstacle avoidance.

        Parameters
        ----------
        x : casadi.MX or casadi.DM
            Current vehicle state vector [px, py, ...].
        x_ref : float
            Reference x-position on the path at the corresponding timestep.
        y_ref : float
            Reference y-position on the path at the corresponding timestep.
        psi_ref : float
            Reference heading (orientation) on the path at the corresponding timestep.

        Returns
        -------
        casadi.MX
            Scalar potential field cost, increasing with lateral deviation from lane centers.
        """
        px, py = x[0], x[1]
        U_lm = 0
        for i in range(3):
            lm = (i+ 0.5) * self.lane_width
            e_lm = lm - (self.get_e_c(px, py, x_ref, y_ref, psi_ref))
            U_lm = U_lm + ca.exp(-(e_lm/(2/2))**2) 
            lm = (-i- 0.5) * self.lane_width
            e_lm = lm - (self.get_e_c(px, py, x_ref, y_ref, psi_ref))
            U_lm = U_lm + ca.exp(-(e_lm/(2/2))**2) 
        return U_lm

    def get_collision_soft_obstacle_costs(self, xk, v_e, t):
        """
        Compute soft collision-avoidance costs for the ego vehicle at a given time step.

        This term introduces a smooth repulsive potential around predicted obstacle
        positions using a super-Gaussian field. The potential grows rapidly as the ego
        vehicle’s projected front point intrudes into a safety envelope around each
        obstacle, where the envelope is anisotropic and scales with the estimated
        closing speed.

        The cost is evaluated in each obstacle’s local frame to capture its orientation
        and dimensions. Faster closing speeds enlarge both lateral and longitudinal
        radii, producing a more conservative potential when approaching an obstacle
        head-on.

        Parameters
        ----------
        xk : casadi.MX
            Ego state at time index `t`, containing at least position and heading
            `[px, py, psi]`.
        v_e : float or casadi.MX
            Ego speed magnitude used to compute closing-speed components.
        t : int
            Time index within the prediction horizon at which the soft cost is evaluated.

        Returns
        -------
        casadi.MX
            Soft collision-avoidance cost aggregated over all obstacles at time `t`.
        """
        # parameters
        w_plateau = self.get_time_decay_weights()
        preds = self.parameters["soft_obs_preds"]
        geom = self.parameters["soft_obs_geom"]
        # super-Gaussian order
        r = 12

        # ego pose at node (s,t)
        px  = xk[0]
        py  = xk[1]
        psi = xk[2]
        cpsi = ca.cos(psi)
        spsi = ca.sin(psi)

        # front offset 
        off_front = self.wheelbase/2.0 # center of vehicle
        ego_x_front = px + off_front * cpsi
        ego_y_front = py + off_front * spsi

        # ego velocity components for closing speed 
        vx_e = v_e * cpsi
        vy_e = v_e * spsi

        cost = 0
        for o in range(self.num_soft_obs):
            base   = o * 3 * (self.N + 1)
            obs_x  = preds[base + t]
            obs_y  = preds[base + (self.N + 1) + t]
            obs_psi= preds[base + 2 * (self.N + 1) + t]

            # geometry: [w, l]
            obs_w = geom[o * 2 + 0]
            obs_l = geom[o * 2 + 1]

            # obstacle frame rotation
            cO = ca.cos(obs_psi)
            sO = ca.sin(obs_psi)

            # world delta from obstacle to ego FRONT point
            ex_w = ego_x_front - obs_x
            ey_w = ego_y_front - obs_y

            # express in obstacle frame
            ex_v = ex_w * cO + ey_w * sO
            ey_v = ex_w * sO - ey_w * cO

            # closing-speed–scaled radii
            dx   = obs_x - px
            dy   = obs_y - py
            dist = ca.sqrt(dx * dx + dy * dy) + 1e-6
            v_closing = ca.fmax(0.0, (vx_e * dx + vy_e * dy) / dist)

            ego_radius = self.width / 2.0
            d_lat  = obs_w / 2.0 + ego_radius * (0.99 + 0.002 * v_closing)
            d_long =  obs_l / 2.0 + obs_l / 10.0 + ego_radius * (1 + 0.12 * v_closing)

            # super-Gaussian "rectangle" potential; 
            phi_o_k = w_plateau[t] * ca.exp(
                -(( (ex_v / d_long) ** r + (ey_v / d_lat) ** r + 1e-6 ) ** (1.0 / r)) ** 4
            )

            cost += phi_o_k

        return cost  

    def solve(self, x0, xref,  hard_obs, soft_obs, path, wl, wr, iteration, vmax=None, **kwargs):
        """
        Solve the MPC optimization problem for the current snapshot.

        The solver builds an initial guess for states and controls, assembles the
        parameter vector from the current state, reference path, and obstacle
        predictions, substitutes the current velocity limit into the box constraints,
        and then calls the nonlinear solver to compute an optimal control sequence
        and state trajectory over the scenario tree.

        If the solver fails to converge consistently, a simple emergency strategy is
        triggered: the previous solution (or warm-start guess) is shifted forward,
        and an emergency braking profile is generated when the vehicle is already
        moving slowly. This prevents unbounded behavior when the optimizer does
        not return a valid solution.

        Parameters
        ----------
        x0 : array_like
            Current ego state used as the initial condition.
        xref : array_like
            Target (terminal) reference state for the horizon.
        hard_obs : list
            List of hard obstacle objects used to form hard prediction scenarios.
        soft_obs : list
            List of soft obstacle objects used for soft potential-field costs.
        path : array_like, shape (M, 3)
            Discretized reference path `[x_ref, y_ref, psi_ref]`.
        wl : array_like
            Left corridor boundary widths or offsets passed to the scenario selector.
        wr : array_like
            Right corridor boundary widths or offsets passed to the scenario selector.
        vmax : float, optional
            Maximum velocity constraint for the current solve. If `None`, a default
            value is taken from the controller configuration.
        **kwargs :
            Additional options forwarded to warm-start and scenario-selection
            routines.

        Returns
        -------
        tuple
            u_full : ndarray or casadi.DM
                Selected control sequence over the horizon (possibly shifted or
                emergency-braking profile in failure cases).
            x_full : ndarray or casadi.DM
                Corresponding state trajectory over the horizon.
            success : bool
                Flag indicating whether the NLP solver reported success.
            f_val : float or ndarray
                Final objective value reported by the solver.
        """
        if vmax == None:
            vmax = self.config["controller"]["v_max"]

        ### initial decision variable values
        opt0_, x0_, u0_, lamx0, lamg0 = self.build_mpc_initial_values(x0, vmax, **kwargs)

        ### update box constraints
        lbx, ubx = self.update_box_constraints(vmax, iteration.index)

        ### optimization parameters
        c_p = self.build_mpc_param_values(x0, xref, hard_obs, soft_obs, path, wl, wr, vmax, **kwargs)
        ### call solver
        res = self.solver(x0=opt0_,  p=c_p, lbg=self.lbg, #lam_x0=self.lamx0_, lam_g0=self.lamg0_, 
                     lbx=lbx, ubg=self.ubg, ubx=ubx)
        estimated_opt = res['x'].full()        

        # handle success/emergency braking
        if self.solver.stats()["success"] or float(ca.norm_inf(u0_)) == 0:
            self._fail_streak = 0
            self._emergency_mode = False

            u_full = estimated_opt[0: self.n_controls * self.N_edges].reshape(self.n_controls, self.N_edges, order="F").T
            x_full = estimated_opt[self.n_controls * self.N_edges: self.n_controls * self.N_edges + self.n_states * self.N_nodes].reshape(self.n_states, self.N_nodes, order="F").T
        else:
            self._fail_streak += 1
            #print(check_bounds(np.array(res['lam_g'].full()), np.array(self.lbg), np.array(self.ubg)) )
            if self._fail_streak > 3 and x0[3] < 0.3:
                self._emergency_mode = True
                u_full = np.concatenate((u0_[1:], u0_[-1:]))
                x_full = np.concatenate((x0_[1:], x0_[-1:]))
                x0[4] = self.a_min # force maximum break acceleration
                x0[5] = 0 # force straight line stop
                x_full[:self.N,:] = self.warmstarter.generate(np.array(x0).reshape(-1,), vmax = vmax, N = self.N, dt = self.dt).T
            else:
                u_full = np.concatenate((u0_[1:], u0_[-1:]))
                x_full = np.concatenate((x0_[1:], x0_[-1:]))

        self.u_opt =  estimated_opt[0: self.n_controls * self.N_edges].reshape(self.n_controls, self.N_edges, order="F").T
        self.x_opt = estimated_opt[self.n_controls * self.N_edges: self.n_controls * self.N_edges + self.n_states * self.N_nodes].reshape(self.n_states, self.N_nodes, order="F").T

        return self.transform_state_to_trajectory(x_full=x_full, iteration=iteration, **kwargs), u_full, x_full, self.solver.stats()["success"], res["f"]

    def transform_state_to_trajectory(self, x_full, ego_state, **kwargs):
        trajectory: List[EgoState] = []
        time_point = ego_state.time_point
        for i in range(0, self.N):
            state = EgoState.build_from_rear_axle( 
                    rear_axle_pose=StateSE2(float(x_full[i,0]) , float(x_full[i,1]),    float(x_full[i,2])),
                    rear_axle_velocity_2d=StateVector2D(float(x_full[i,3]), 0),
                    rear_axle_acceleration_2d=StateVector2D(float(x_full[i,4]), 0),
                    tire_steering_angle=float(x_full[i,5]),
                    time_point=time_point,
                    vehicle_parameters=ego_state.car_footprint.vehicle_parameters)
            trajectory.append(state)
            time_point += TimePoint(int(self.dt * 1e6))
        traj_interpolated = InterpolatedTrajectory(trajectory)
        return traj_interpolated

    def get_time_decay_weights(self):
        # timestep decay with plateau
        Tp = self.N/5
        w_end = self.decay
        lam_plateau = -np.log(w_end) / (self.N - Tp)
        w_plateau = np.ones(self.N + 1)
        k = np.arange(self.N + 1)
        w_plateau[k > Tp] = np.exp(-lam_plateau * (k[k > Tp] - Tp))
        return w_plateau
    

    def update_box_constraints(self, vmax, iteration):
        """
        Build numeric box constraints (lbx, ubx) for the current solve.

        This performs:
        - substitution of the symbolic velocity bound `v_max_sym` into `self.ubx`
        - conversion of symbolic bounds to numeric `DM`
        - optional tightening of control bounds in the first few edges,
            depending on the simulation tick.

        Parameters
        ----------
        vmax : float
            Maximum velocity for this MPC solve (used to substitute `v_max_sym`).
        iteration : int
            Current simulation tick. For the first few ticks, the bounds on the
            first control edges are tightened.

        Returns
        -------
        tuple of (lbx, ubx) as casadi.DM
            Numeric lower and upper bounds for the decision vector.
        """
        # substitute symbolic parameter and transform it to numerical
        lbx = ca.DM(self.lbx)  # just convert to numeric
        ubx = ca.DM(ca.substitute(self.ubx, self.v_max_sym, vmax))

        if iteration < 3:
            n_tight_edges = min(5, self.N_edges)   # at most 5, but not more than N_edges
            n_u = self.n_controls                  # controls per edge
            alpha_floor = 0.1
            alpha = alpha_floor + (iteration / 3.0) * (1 - alpha_floor)
            alpha = max(alpha_floor, min(alpha, 1.0))

            for k in range(n_tight_edges):
                beta = k / (n_tight_edges - 1)

                factor = alpha + beta * (1 - alpha)
                factor = max(factor, 1)

                tight_lbx_u = ca.DM([
                    (-2 * self.j_max),        # reduced negative jerk
                    factor * (-self.ddelta_max),   # reduced steering rate
                    0.0 
                ])
                tight_ubx_u = ca.vertcat(
                    factor * self.j_max,         # reduced positive jerk
                    factor * self.ddelta_max,    # reduced steering rate
                    (1.5*vmax)
                )

                offset = k * n_u
                lbx[offset : offset + n_u] = tight_lbx_u
                ubx[offset : offset + n_u] = tight_ubx_u

        return lbx, ubx


##### Functions implemented in sublcasses #####

    def init_decision_variables(self):
        raise NotImplementedError()

    def init_parameter_variables(self):
        raise NotImplementedError()

    def init_path_refpoints(self):
        raise NotImplementedError()

    def init_costs(self):
        raise NotImplementedError()
    
    def init_nonlinear_constraints(self):
        raise NotImplementedError()

    def init_box_constraints(self):
        raise NotImplementedError()

    def add_node_costs(self, cost):
        raise NotImplementedError()
    
    def add_driving_corridor_constraints(self, g, lbg, ubg):
        raise NotImplementedError()

    def build_mpc_initial_values(self, x0, v_max, **kwargs):
        raise NotImplementedError()

    def build_mpc_param_values(self, x0, xref, hard_obs, soft_obs, path, wl, wr, v_max, **kwargs):
        raise NotImplementedError()

###### Old Functions

    # def get_collision_soft_constraint_costs(self):
    #     """
    #     Add soft collision-avoidance costs for predicted obstacles without
    #     hard epigraph constraints.

    #     For each soft obstacle, a super-Gaussian potential `phi_s_o_k` is
    #     constructed in the obstacle frame at every scenario–time node of the
    #     tree, using the same L^r (super-elliptic) rectangle approximation
    #     used for hard obstacles. The radii in longitudinal and lateral
    #     directions are scaled by the ego–obstacle closing speed and obstacle
    #     geometry, yielding a conservative, velocity-aware safety envelope in
    #     front of the ego vehicle.

    #     For each obstacle `o`, the per-node potentials `phi_s_o_k` are
    #     aggregated into a single scalar soft cost using a smooth maximum
    #     based on the log-sum-exp operator:

    #         max_k phi_k  ≈  (1/α) * log( Σ_k exp( α * phi_k ) ),

    #     where `α > 0` controls the sharpness of the approximation (larger
    #     α → closer to the true max, smaller α → closer to an averaged
    #     potential). The resulting obstacle-wise soft max is then added to
    #     the global objective, encouraging the optimizer to avoid regions of
    #     high collision potential without enforcing hard feasibility.

    #     Returns
    #     -------
    #     casadi.MX or casadi.SX
    #         Scalar objective contribution representing the aggregated soft
    #         collision costs for all hard obstacles.
    #     """
    #     obs_preds = self.parameters["soft_obs_preds"]
    #     obs_geom = self.parameters["soft_obs_geom"]
    #     X = self.decision_variables["X"]

    #     r = 12 # L^r aggregator order (super-Gaussian)
    #     alpha_soft_max = 6.0 # sharpness of log-sum-exp operation
    #     cost = 0

    #     # timestep decay with plateau
    #     w_plateau = self.get_time_decay_weights()

    #     for o in range(self.num_soft_obs):
    #         obstacle_costs = []
    #         for s, t, w in self.iter_tree_nodes(self.parameters["scenario_weights"]):
    #             k = self._kth_node(s, t)

    #             # Ego pose 
    #             px = X[0, k]
    #             py = X[1, k]
    #             psi  = X[2, k]
    #             cpsi = ca.cos(psi)
    #             spsi = ca.sin(psi)

    #             # front offset 
    #             off_front = 3.0
    #             ego_x_front = px + off_front * cpsi
    #             ego_y_front = py + off_front * spsi

    #             # Ego velocity (root speed, as in your code)
    #             v_e   = X[3, 0]
    #             vx_e  = v_e * cpsi
    #             vy_e  = v_e * spsi

    #             # obstacle state at (o, s, t)
    #             base = o * self.n_pred_params * (self.N + 1)
    #             obs_x   = obs_preds[base + t]
    #             obs_y   = obs_preds[base + (self.N + 1) + t]
    #             obs_psi = obs_preds[base + 2 * (self.N + 1) + t]

    #             # geometry (length then width) and trigs
    #             obs_l = obs_geom[o * 2 + 1]         # length
    #             obs_w = obs_geom[o * 2]     # width
    #             cO = ca.cos(obs_psi)
    #             sO = ca.sin(obs_psi)

    #             # world deltas to ego front
    #             ex_w = ego_x_front - obs_x
    #             ey_w = ego_y_front - obs_y

    #             # express in obstacle frame
    #             ex_v = ex_w * cO + ey_w * sO
    #             ey_v = ex_w * sO - ey_w * cO

    #             # closing-speed–scaled radii 
    #             ego_radius = self.width / 2.0
    #             dx   = obs_x - px
    #             dy   = obs_y - py
    #             dist = ca.sqrt(dx * dx + dy * dy) + 1e-6
    #             v_closing = ca.fmax(0.0, (vx_e * dx + vy_e * dy) / dist)

    #             d_lat  = 0.99 * (ego_radius + obs_w / 2.0) + 0.01 * v_closing
    #             d_long = ego_radius + obs_l / 2.0 + obs_l / 10.0 + 0.1 * v_closing

    #             # super-Gaussian L^r rectangle approximation 
    #             # φ = decay^(2*t+1) * w_s * exp( - ( (|ex_v|/d_long)^r + (|ey_v|/d_lat)^r + eps )^(4/r) )
    #             phi_s_o_k = w_plateau[t] * w * ca.exp(
    #                 -(( (ex_v / d_long) ** r + (ey_v / d_lat) ** r + 1e-6) ** (1.0 / r)) ** 4
    #             )

    #             obstacle_costs.append(phi_s_o_k)
    #         cost_vec = ca.vertcat(*obstacle_costs) # (num_nodes_for_o, 1)
    #         # obstacle_max_cost = (1.0 / alpha_soft_max) * ca.log(ca.sum1(ca.exp(alpha_soft_max * cost_vec)) + 1e-9)
    #         # cost += obstacle_max_cost
    #         cost += ca.norm_inf(cost_vec)

    #     return cost 