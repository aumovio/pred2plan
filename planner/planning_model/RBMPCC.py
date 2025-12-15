"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import numpy as np
import casadi as ca
import logging

from planner.planning_model.BaseMPCC import BaseMPCC

logger = logging.getLogger(__name__)

from planner.planning_utils.ScenarioSelection import ReachScenarioSelection as rss

class RBMPCC(BaseMPCC):

    def init_decision_variables(self):
        """
        Create all decision variables for the RBMPCC.

        The decision set consists of the full collection of state vectors at every
        node of the scenario tree and the control vectors defined on each tree edge.
        States are arranged as a matrix whose columns correspond to scenario–time
        nodes, while controls are arranged similarly over the tree edges. The total
        number of nodes and edges follows from the branching structure of the
        scenario tree.

        All decision variables are concatenated into a single vector in a fixed
        order to define the optimization variable `x` for the NLP. This vector forms
        the backbone of the symbolic formulation used by the solver.

        Returns
        -------
        None
            The method populates `self.decision_variables` and `self.opt_variables`
            with the symbolic structures required by the nonlinear program.
        """
        # total number of nodes/edges in tree
        self.N_nodes = (self.N - 1) * self.S + 2
        self.N_edges = (self.N - 1) * self.S + 1
        decision_variables = {}
        # states (one state vector per column) over all time steps * scenarios + 2 shared states at root (x0 and x1)
        decision_variables["X"] = ca.MX.sym('X', self.n_states, self.N_nodes)
        # controles (one control vector per column) over all times steps * scenarios + 1 shared control at root (u0)
        decision_variables["U" ]= ca.MX.sym('U', self.n_controls, self.N_edges)
        # slacks for path
        decision_variables["path_contouring_slack"] = ca.MX.sym("sigma_c", self.S)
        decision_variables["path_halfspace_slack"] = ca.MX.sym("sigma_hs", self.S)

        self.decision_variables = decision_variables

        #concat_order = ["U", "X", "collision_slack"]
        concat_order = ["U", "X", "path_contouring_slack", "path_halfspace_slack"]

        self.opt_variables = ca.vertcat(*[ca.vec(decision_variables[key]) for key in concat_order])

    def init_parameter_variables(self):
        """
        Create all symbolic parameter variables used by the RBMPCC problem.

        Parameters include the initial and terminal reference states, scenario
        probabilities, predicted obstacle trajectories and geometries for both
        hard and soft constraints, non-anticipativity weights, path reference
        data, and reachability-based bounds. These quantities are treated as
        fixed inputs to the optimization problem and can be updated at each
        solve without rebuilding the symbolic structure.

        All parameter vectors are concatenated into a single symbol `P`, which
        serves as the parameter input to the nonlinear program passed to IPOPT.

        Returns
        -------
        None
            The method populates `self.parameters` with individual symbolic
            parameter blocks and constructs `self.P` as their ordered
            concatenation.
        """
        parameters = {}
        parameters["x_init"] = ca.MX.sym("x_init", self.n_states)
        parameters["x_goal"] = ca.MX.sym("x_goal", self.n_states) # reference terminal state
        parameters["scenario_weights"]  = ca.MX.sym("w_s", self.S)
        parameters["hard_obs_preds"] = ca.MX.sym("hard_obs_preds", self.num_hard_obs * self.n_pred_params * (self.N+1) * self.S) # [x,y,psi]
        parameters["hard_obs_geom"] = ca.MX.sym("hard_obs_geom", self.num_hard_obs * 2) # [w, l]
        parameters["soft_obs_preds"] = ca.MX.sym("soft_obs_preds", self.num_soft_obs * self.n_pred_params * (self.N+1)) # [x,y,psi]
        parameters["soft_obs_geom"] = ca.MX.sym("soft_obs_geom", self.num_soft_obs * 2) # [w, l]
        parameters["non_anticipatory_weights"] = ca.MX.sym("w_na", self.max_N_control - 1)
        parameters["path_ref"] = ca.MX.sym("path", self.M * 3) # [x, y, psi, w_left, w_right]
        parameters["reachbased"] = ca.MX.sym("rb", self.N_nodes * 8) # [e_c_min, e_c_max, mx_min, my_min, c_min, mx_max, my_max, c_max]
        self._scenario_selector_initialized = False
        self.parameters = parameters
        concat_order = [
            "x_init",
            "x_goal",
            "hard_obs_preds",
            "hard_obs_geom",
            "scenario_weights",
            "non_anticipatory_weights",
            "path_ref",
            "soft_obs_preds",
            "soft_obs_geom",
            "reachbased"
        ]
        self.P = ca.vertcat(*[parameters[key] for key in concat_order])

    def init_path_refpoints(self):
        """
        Initialize symbolic reference-path values along the arc-length coordinate.

        The path reference is supplied as discretized samples of position and heading
        over an arc-length grid. These samples are converted into CasADi interpolants
        so that, for each scenario–time node, the reference values can be evaluated
        at the ego vehicle’s current arc-length state.

        The resulting symbolic expressions `X_ref`, `Y_ref`, and `Psi_ref` provide the
        reference path centerline at every node of the scenario tree, enabling
        contouring-error and heading-error terms to be computed consistently within
        the optimization problem.

        Returns
        -------
        None
            The method populates internal lists `X_ref`, `Y_ref`, and `Psi_ref`
            with symbolic interpolants evaluated at each node’s arc-length state.
        """
        # retrieve path reference variables
        path_ref = self.parameters["path_ref"]
        #path_mat = ca.reshape(path_ref, 5, self.M).T
        M = self.M
        X_path   = path_ref[0*M : 1*M]
        Y_path   = path_ref[1*M : 2*M]
        Psi_path = path_ref[2*M : 3*M]

        # define arc-length grid
        S_path = np.linspace(0.0, float(self.s_final), int(self.M))
        # create casadi parametric interpolants
        lut_x   = ca.interpolant("LUT_x",   "linear", [S_path])
        lut_y   = ca.interpolant("LUT_y",   "linear", [S_path])
        lut_psi = ca.interpolant("LUT_psi", "linear", [S_path])

        # interpolate along the vehicle arc-length s = self.X[6, k]
        s_idx = 6  # s is the 7th state: [x,y,psi,vx,ax,delta,s]

        for k in range(self.N_nodes):
            s_k = self.decision_variables["X"][s_idx, k]

            # interpolate symbolically
            x_ref   = lut_x(s_k,   X_path)
            y_ref   = lut_y(s_k,   Y_path)
            psi_ref = lut_psi(s_k, Psi_path)

            # store references
            self.X_ref.append(x_ref)
            self.Y_ref.append(y_ref)
            self.Psi_ref.append(psi_ref)

        if self.verbose: logger.info(f"Initialize {self.__class__.__name__} reference signal ... DONE!")

    def init_costs(self):
        """
        Assemble the full objective function of the MPC problem.

        The total cost is built from contributions defined at scenario–time nodes
        and along scenario–time edges, together with soft penalties enforcing
        non-anticipativity across branches of the scenario tree. Node costs typically
        encode tracking objectives, while edge costs capture control effort and
        motion smoothness. Non-anticipativity terms regularize deviations between
        scenarios before divergence points.

        The resulting symbolic expression defines the objective used by the NLP
        solver and is stored internally for use during solver construction.

        Returns
        -------
        None
            The method assigns the composed symbolic objective to `self.cost`.
        """
        cost = 0
        cost = self.add_slack_costs(cost)
        cost = self.add_node_costs(cost)
        cost = self.add_edge_costs(cost)
        cost = self.add_nonanticipativity_costs(cost)
        self.cost = cost

    def add_slack_costs(self, cost):
        """
        Add penalty terms for slack variables to the MPC objective.

        Path-slack penalties are quadratic, encouraging small and symmetric
        violations of the driving corridor while avoiding abrupt kinks in the
        optimized trajectory, both in contouring and halfspace constraints.

        Together, these terms regulate the trade-off between strict constraint
        satisfaction and overall problem solvability.

        Parameters
        ----------
        cost : casadi.MX
            Current accumulated cost expression to which slack penalties are added.

        Returns
        -------
        casadi.MX
            Updated objective including path- and collision-slack penalties.
        """
        cost += self.q_ps * ca.sum1(self.decision_variables["path_contouring_slack"]**2) # quadratic cost on path contouring violation
        cost += self.q_ps * ca.sum1(self.decision_variables["path_halfspace_slack"]**2) # quadratic cost on path halfspace violation
        return cost

    def init_nonlinear_constraints(self):
        """
        Initialize all nonlinear equality and inequality constraints of the MPC problem.

        The constraint set includes system-dynamics equalities, g–g diagram bounds
        capturing feasible combined accelerations, and path-feasibility inequalities
        restricting the ego vehicle to its driving corridor. Each constraint block is
        collected into dictionaries along with their corresponding lower and upper
        bounds.

        After all blocks are assembled, they are concatenated in a fixed ordering
        to form the solver’s unified constraint vector and bound vectors. This
        ordering must remain consistent with the nonlinear program supplied to
        the solver.

        Returns
        -------
        None
            The method populates internal dictionaries for individual constraint
            blocks and constructs the concatenated constraint vector `self.g`
            together with its bound vectors `self.lbg` and `self.ubg`.
        """
        g_dict, lbg_dict, ubg_dict = {}, {}, {}
        self.add_equality_constraints(g_dict, lbg_dict, ubg_dict)
        self.add_gg_diamond_constraints(g_dict, lbg_dict, ubg_dict)
        #self.add_collision_hard_constraints(g_dict, lbg_dict, ubg_dict)
        self.add_driving_corridor_constraints(g_dict, lbg_dict, ubg_dict)
        self.nonlinear_constraints = g_dict
        self.nonlinear_constraint_lbg = lbg_dict
        self.nonlinear_constraint_ubg = ubg_dict

        # define concatenation order (important for solver alignment)
        #concat_order = ["ic_eq", "dynamics_eq", "gg_ineq", "path_ineq", "collision_ineq"]
        concat_order = ["ic_eq", "dynamics_eq", "gg_ineq", "path_ineq"]

        # concatenate into solver vectors
        self.g   = ca.vertcat(*[g_dict[key]   for key in concat_order])
        self.lbg = ca.vertcat(*[lbg_dict[key] for key in concat_order])
        self.ubg = ca.vertcat(*[ubg_dict[key] for key in concat_order])

    def init_box_constraints(self):
        """
        Initialize all simple (box) bounds on states and controls.

        Box constraints impose element-wise lower and upper limits on the
        decision variables. State bounds restrict velocity, acceleration,
        and steering angle to physically feasible ranges, while control
        bounds constrain jerk, steering-rate, and commanded velocity. 
        Slack variables are constrained in addition. These limits are 
        assembled separately and then merged into a single pair of lower 
        and upper bound vectors matching the concatenated decision-variable 
        ordering.

        A symbolic parameter for the maximum allowed velocity is also
        created so that velocity limits can be varied at solve time
        without rebuilding the problem.

        Returns
        -------
        None
            The method stores dictionary-based bounds for clarity and
            also constructs flattened vectors `self.lbx` and `self.ubx`
            required by the nonlinear solver.
        """
        self.v_max_sym = ca.SX.sym("v_max")

        lbx_dict, ubx_dict = {}, {}
        lbx_dict, ubx_dict = self.add_state_box_constraints(lbx_dict, ubx_dict)
        lbx_dict, ubx_dict = self.add_control_box_constraints(lbx_dict, ubx_dict)
        lbx_dict, ubx_dict = self.add_slack_box_constraints(lbx_dict, ubx_dict)
        # concatenate things
        self.box_constraints_lbx = lbx_dict
        self.box_constraints_ubx = ubx_dict
        # concatenate to flat solver vectors
        self.lbx = ca.vertcat(ca.vec(lbx_dict["controls"]),
                            ca.vec(lbx_dict["states"]),
                            ca.vec(lbx_dict["path_contouring_slack"]),
                            ca.vec(lbx_dict["path_halfspace_slack"])
                            )
        self.ubx = ca.vertcat(ca.vec(ubx_dict["controls"]),
                            ca.vec(ubx_dict["states"]),
                            ca.vec(ubx_dict["path_contouring_slack"]),
                            ca.vec(ubx_dict["path_halfspace_slack"])
                            )

    def add_slack_box_constraints(self, lbx, ubx):
        """
        Add simple box bounds for all slack variables used to soften feasibility constraints.

        Two categories of slack variables are bounded:
        - Path slacks, which soften the driving-corridor constraints.
        - Collision slacks, which soften hard collision-avoidance constraints.

        Path slacks are allowed to vary symmetrically within a scaled interval,
        while collision slacks are restricted to nonnegative values, controlling
        how much penetration or overlap with obstacles can be tolerated.

        Parameters
        ----------
        lbx : dict
            Dictionary of lower bounds, updated with entries for `"path_contouring_slack"` and
            `"path_halfspace_slack"`.
        ubx : dict
            Dictionary of upper bounds, updated with entries for `"path_contouring_slack"` and
            `"path_halfspace_slack"`.

        Returns
        -------
        tuple
            Updated `(lbx, ubx)` dictionaries including box constraints for all slack variables.
        """
        path_contouring_slack_numel = self.decision_variables["path_contouring_slack"].numel()
        lbx["path_contouring_slack"] = -ca.DM.ones(path_contouring_slack_numel) * self.slack_scale
        ubx["path_contouring_slack"] = ca.SX.ones(path_contouring_slack_numel) * self.slack_scale

        path_halfspace_slack_numel = self.decision_variables["path_halfspace_slack"].numel()
        lbx["path_halfspace_slack"] = -ca.DM.ones(path_halfspace_slack_numel) * self.slack_scale
        ubx["path_halfspace_slack"] = ca.SX.ones(path_halfspace_slack_numel) * self.slack_scale

        return lbx, ubx

    def add_node_costs(self, cost):
        """
        Add node-based running costs for each scenario–time node.

        The node cost aggregates several contributions evaluated at the current
        state and reference-path location. A quadratic tracking term penalizes
        deviations from the desired terminal state, while contouring and lag
        errors measure geometric deviation from the reference path. Lane
        potential-field terms encourage staying near the corridor centerline,
        and soft collision-avoidance potentials penalize proximity to predicted
        obstacles. An additional smoothing term discourages sustained positive
        longitudinal acceleration.

        Terminal nodes receive an increased weight on contouring and lateral
        costs to sharpen final-path adherence. Each node cost is multiplied by
        the scenario weight associated with that branch of the scenario tree.

        Parameters
        ----------
        cost : casadi.MX
            The accumulated objective function to which the node costs are added.

        Returns
        -------
        casadi.MX
            Updated cost including all weighted node contributions.
        """
        scenario_weights = self.parameters["scenario_weights"]
        X, U = self.decision_variables["X"], self.decision_variables["U"]

        x_goal = self.parameters["x_goal"]
        for s, t, w in self.iter_tree_nodes(scenario_weights):
            k = self._kth_node(s,t)
            xk  = X[:, k]
            xr  = self.X_ref[k]
            yr  = self.Y_ref[k]
            pr  = self.Psi_ref[k]

            if t == self.N:
                multiplier = 2
            else:
                multiplier = 1

            node_cost = ca.mtimes([(xk-x_goal).T, self.Q, xk-x_goal]) # quadratic tracking cost on state deviations from terminal x_ref
            node_cost += multiplier * self.get_contouring_costs(xk, xr, yr, pr, self.q_c, self.q_l) # cost on contouring/lag error on reference path
            node_cost += multiplier * self.q_lb * self.get_lane_costs( xk, xr, yr, pr)  # cost on lane potentialfield
            node_cost += self.q_so * self.get_collision_soft_obstacle_costs(xk, X[3, 0], t)
            node_cost += self.q_ho * self.get_collision_hard_obstacle_costs(xk, X[3, 0], t, s)
            node_cost += self.q_a*((1/10.0) * ca.log1p(ca.exp(10.0 * (xk[4]-0.5)))) # additional cost on positive acceleration
            cost += w * node_cost

        return cost

    def get_collision_hard_obstacle_costs(self, xk, v_e, t, s):
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
            Ego state at time index `t` for scenario `s`, containing at least position and heading
            `[px, py, psi]`.
        v_e : float or casadi.MX
            Ego speed magnitude used to compute closing-speed components.
        t : int
            Time index within the prediction horizon at which the soft cost is evaluated.
        s: int
            Scenario/branch index

        Returns
        -------
        casadi.MX
            Soft collision-avoidance cost aggregated over all obstacles at time `t` for scenario `s`.
        """
        # parameters
        w_plateau = self.get_time_decay_weights()
        preds = self.parameters["hard_obs_preds"]
        geom = self.parameters["hard_obs_geom"]
        # super-Gaussian order
        r = 12

        # ego pose at node (s,t)
        px  = xk[0]
        py  = xk[1]
        psi = xk[2]
        cpsi = ca.cos(psi)
        spsi = ca.sin(psi)

        # front offset 
        off_front = 3.0
        ego_x_front = px + off_front * cpsi
        ego_y_front = py + off_front * spsi

        # ego velocity components for closing speed 
        vx_e = v_e * cpsi
        vy_e = v_e * spsi

        cost = 0
        for o in range(self.num_soft_obs):
            base   = o * self.n_pred_params * (self.N + 1) * self.S + s * self.n_pred_params * (self.N + 1)
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
            d_long = obs_l / 2.0 + obs_l / 10.0 + ego_radius * (1 + 0.12 * v_closing)

            # super-Gaussian "rectangle" potential; 
            phi_o_k = w_plateau[t] * ca.exp(
                -(( (ex_v / d_long) ** r + (ey_v / d_lat) ** r + 1e-6 ) ** (1.0 / r)) ** 4
            )

            cost += phi_o_k

        return cost  

    def add_driving_corridor_constraints(self, g, lbg, ubg):
        """
        Add driving-corridor inequality constraints based on reachability bounds.

        These constraints restrict the ego vehicle to remain inside a feasible
        driving corridor around the reference path, using both lateral band limits
        and polygonal half-space bounds derived from reachability analysis.

        For each scenario–time node, the front corners of the vehicle are computed
        from the state and vehicle geometry. Their signed lateral (contouring)
        errors with respect to the reference-path normal are constrained to stay
        between lower and upper bounds `e_c_min` and `e_c_max`. Additional linear
        half-space constraints on the vehicle position `(px, py)` enforce corridor
        boundaries encoded by `(mx_min, my_min, c_min)` and `(mx_max, my_max, c_max)`.

        Early-horizon nodes use slightly relaxed offsets to avoid over-constraining
        the initial states, with tighter bounds applied deeper in the horizon.

        Parameters
        ----------
        g : dict
            Dictionary of nonlinear constraint expressions. The key `"path_ineq"`
            is populated with the stacked corridor inequality constraints.
        lbg : dict
            Dictionary of lower bounds corresponding to each constraint block.
            Corridor constraints may be one-sided (−∞, 0] or [0, +∞).
        ubg : dict
            Dictionary of upper bounds corresponding to each constraint block.

        Returns
        -------
        tuple
            Updated `(g, lbg, ubg)` dictionaries including the driving-corridor
            inequality constraints and their bounds.
        """
        X = self.decision_variables["X"]
        path_constraints, lower_bounds, upper_bounds = [], [], []
        N, S = self.N, self.S

        for s, t, _ in self.iter_tree_nodes(self.parameters["scenario_weights"]):
            k = self._kth_node(s, t)

            sigma_contouring = self.decision_variables["path_contouring_slack"][s]
            sigma_halfspace = self.decision_variables["path_halfspace_slack"][s]
            
            if t < 2:
                ab = 0
                ofs = 0.4
            elif t < 7:
                ab = 1.0
                ofs = 0.3
            else:
                ab = 1.0
                ofs = 0.2

            # Vehicle state at node k
            px   = X[0, k]
            py   = X[1, k]
            psi  = X[2, k]
            cpsi = ca.cos(psi)
            spsi = ca.sin(psi)

            # Front axle position (wheelbase ahead of rear-axle pose)
            x_fa = px + self.wheelbase * cpsi
            y_fa = py + self.wheelbase * spsi

            # Front corners
            half_w = 0.5 * self.width
            x_fl = x_fa - half_w * spsi
            y_fl = y_fa + half_w * cpsi
            x_fr = x_fa + half_w * spsi
            y_fr = y_fa - half_w * cpsi

            # Reference track at node k
            x_ref   = self.X_ref[k]
            y_ref   = self.Y_ref[k]
            psi_ref = self.Psi_ref[k]

            # Contouring error at center (no corners)
            e_c = self.get_e_c(px, py, x_ref, y_ref, psi_ref)

            # Corner contouring errors (signed lateral errors w.r.t. reference normal)
            e_c_left  = self.get_e_c(x_fl, y_fl, x_ref, y_ref, psi_ref)
            e_c_right = self.get_e_c(x_fr, y_fr, x_ref, y_ref, psi_ref)

            # === Bounds and half-space parameters from P =========================
            # Slice layout (each block has size S*(N+1)):
            # 0:            e_c_min
            # 1:            e_c_max
            # 2:            mx_min
            # 3:            my_min
            # 4:            c_min
            # 5:            mx_max
            # 6:            my_max
            # 7:            c_max

            RB_params = self.parameters["reachbased"]
            e_c_min = RB_params[0*self.N_nodes + k]
            e_c_max = RB_params[1*self.N_nodes + k]
            mx_min  = RB_params[2*self.N_nodes + k]
            my_min  = RB_params[3*self.N_nodes + k]
            c_min   = RB_params[4*self.N_nodes + k]
            mx_max  = RB_params[5*self.N_nodes + k]
            my_max  = RB_params[6*self.N_nodes + k]
            c_max   = RB_params[7*self.N_nodes + k]

            # === Contouring band for BOTH front corners ===
            # Left corner: e_c_left <= e_c_max  -> e_c_left - e_c_max <= 0
            path_constraints.append((e_c_left - e_c_max) - sigma_contouring) # either e_c or e_c_left
            lower_bounds.append(-ca.inf)
            upper_bounds.append(0)

            # Right corner: e_c_right >= e_c_min -> e_c_right - e_c_min >= 0
            path_constraints.append((e_c_right - e_c_min) + sigma_contouring) # either e_c or e_c_right
            lower_bounds.append(0)
            upper_bounds.append(ca.inf)

            # === Half-space constraints on (px,py) ===
            # ab*(mx_max*px + my_max*py + c_max) - ofs <= 0
            path_constraints.append(ab*(mx_max*px + my_max*py + c_max) - ofs - sigma_halfspace)
            lower_bounds.append(-ca.inf)
            upper_bounds.append(0)

            # ab*(mx_min*px + my_min*py + c_min) + ofs >= 0
            path_constraints.append(ab*(mx_min*px + my_min*py + c_min) + ofs + sigma_halfspace)
            lower_bounds.append(0)
            upper_bounds.append(ca.inf)

        g["path_ineq"] = ca.vertcat(*path_constraints)
        lbg["path_ineq"] = ca.vertcat(*[ca.DM([lb]) for lb in lower_bounds])
        ubg["path_ineq"] = ca.vertcat(*[ca.DM([ub]) for ub in upper_bounds])

        return g, lbg, ubg
    
    def build_mpc_initial_values(self, x0, v_max, **kwargs):
        """
        Build initial guesses for primal and dual variables of the MPC NLP.

        The initial state trajectory guess is either taken from the previous
        optimization result (warm-start) or, if unavailable, generated by an
        external warm-start module based on the current state `x0`, the velocity
        limit `v_max`, the horizon length, and the sampling time. The control
        trajectory is initialized to zero unless a previous optimal solution is
        available.

        The resulting guesses are stacked into a single decision-vector initial
        value consistent with the internal concatenation order `[U; X; sigma_c, sigma_hs]`.
        Lagrange multipliers for decision variables and nonlinear constraints
        are initialized to zero.

        Parameters
        ----------
        x0 : array_like or casadi.DM
            Current ego state used as the starting point of the warm-start trajectory.
        v_max : float
            Maximum allowed velocity used by the warm-start generator.
        **kwargs :
            Additional keyword arguments reserved for future extensions or for
            passing options to the warm-start routine.

        Returns
        -------
        tuple
            opt0_ : casadi.DM
                Initial guess for the full decision vector `[U; X]`.
            x0_ : casadi.DM
                Initial guess for the state trajectory over all scenario–time nodes.
            u0_ : casadi.DM
                Initial guess for the control trajectory over all scenario–time edges.
            lamx0_ : casadi.DM
                Initial guess for Lagrange multipliers associated with decision variables.
            lamg0_ : casadi.DM
                Initial guess for Lagrange multipliers associated with nonlinear constraints.
        """
        lamx0_ = ca.DM.zeros(self.n_states * self.N_nodes + self.n_controls * self.N_edges + 2*self.S) # states + controls + slacks
        lamg0_ = ca.DM.zeros(self.g.size()) # nonlinear constraints

        path_contouring_slack0_ = ca.DM.zeros(self.S)
        path_halfspace_slack0_ = ca.DM.zeros(self.S)

        if self.x_opt is None:
            x0_ =  np.tile(x0.T,(self.N_nodes,1)) 
            x0_[:self.N,:] = self.warmstarter.generate(x0, vmax = v_max, N = self.N, dt = self.dt).T
            x0_ = ca.DM(x0_)
        else:
            x0_ = self.x_opt

        if self.u_opt is None:
            u0_ = ca.DM.zeros(self.n_controls, self.N_edges) 
        else:
            u0_ = self.u_opt
        
        # concatenate
        opt0_= ca.vertcat(ca.vec(u0_.T), ca.vec(x0_.T), path_contouring_slack0_, path_halfspace_slack0_)

        return opt0_, x0_, u0_, lamx0_, lamg0_

    def build_mpc_param_values(self, x0, xref, hard_obs, soft_obs, path, wl, wr, v_max, pool=None, **kwargs):
        """
        Assemble the full parameter vector for the MPC problem for a given snapshot.

        This routine constructs all time-varying parameters of the MPC from the
        current state, reference path, obstacle predictions, and driving corridor
        information. It performs the following steps:

        - Runs (or reuses) the scenario-selection / reachability module to obtain
        driving-corridor constraints and active modes for hard obstacles.
        - Sets the initial and terminal reference states.
        - Builds the reference path samples (position and heading) along arc-length.
        - Constructs stacked prediction blocks for hard obstacles across scenarios,
        including their geometric dimensions and scenario probabilities.
        - Constructs prediction blocks for soft obstacles, typically using a single
        scenario per obstacle.
        - Derives non-anticipativity weights from the selected control horizon.
        - Packs reachability-based corridor bounds into a flat parameter vector.

        All parameter blocks are concatenated in a fixed order to form the parameter
        vector `P` passed to the nonlinear program.

        Parameters
        ----------
        x0 : array_like
            Current ego state used as the initial condition for the MPC.
        xref : array_like
            Terminal reference state (target pose or goal configuration).
        hard_obs : list
            List of hard obstacle objects providing `prediction(N, dt)` and
            geometry attributes `(w, l)`.
        soft_obs : list
            List of soft obstacle objects treated via potential fields rather than
            hard constraints.
        path : array_like, shape (M, 3)
            Discretized reference path samples `[x_ref, y_ref, psi_ref]` along
            arc-length.
        wl : array_like
            Left corridor boundary widths or offsets used by the scenario selector.
        wr : array_like
            Right corridor boundary widths or offsets used by the scenario selector.
        v_max : float
            Velocity limit passed to the scenario selector and warm-start logic.
        pool : optional
            Optional multiprocessing pool or executor used by the scenario-selection
            routine.

        Returns
        -------
        casadi.DM
            Concatenated parameter vector `P` consistent with `self.init_parameter_variables`
            and the expected ordering:
            `[x_init, x_goal, hard_obs_preds, hard_obs_geom, scenario_weights,
            non_anticipatory_weights, path_ref, soft_obs_preds, soft_obs_geom,
            reachbased]`.
        """
        # generating driving corridor conditions
        if not self._scenario_selector_initialized:
            rss.init_scenario_selector(path, wl , wr, self.num_hard_obs, self.flattened_params) 
            dc_init = np.zeros(((self.N + 1) * self.S, 4))
            acc_brake = -1
            dc_init[:, 0] = x0[-1]
            dc_init[:, 1] = x0[-1] + x0[3] * self.dt + 0.5 * acc_brake * self.dt**2
            dc_init[:, 2] = -self.lane_width/ 2
            dc_init[:, 3] = self.lane_width/ 2
            dc_init = list(rss.get_dc_constraints(dc_init, 0))
            dc_init[0], dc_init[1] = dc_init[0].reshape(-1,1), dc_init[1].reshape(-1,1)
            self.dc_cons = np.stack(dc_init, axis=1)
            self._scenario_selector_initialized = True

        modes, dc_cons, hard_obs, N_control = rss.scenario_selection(
            z0=x0, 
            ObstacleArray = hard_obs, 
            params = self.flattened_params, 
            x0_ = self.x_opt, 
            v_max = v_max, 
            pool =pool)
        if dc_cons != None:
            self.dc_cons = dc_cons 
        else: # roll over last valid dc_cons
            dc_cons  = np.reshape(self.dc_cons,(self.S, self.N + 1, 8))
            first_rows =  dc_cons[:, :1, :]
            last_rows = dc_cons[:, -1:, :]
            last_rows[:,:, 2:] = 0
            self.dc_cons=ca.DM(np.concatenate([first_rows, dc_cons[:, 2:, :], last_rows], axis=1).reshape(-1, 8))
        # states
        params = {}
        params["x_init"] = ca.vec(ca.DM(x0))
        params["x_goal"] = ca.vec(ca.DM(xref))

        # path-ref
        path_mat = ca.hcat([
            path[:, 0],              # x_ref(s)
            path[:, 1],              # y_ref(s)
            path[:, 2],              # psi_ref(s) (continuous/wrap-safe)
        ])
        params["path_ref"] = ca.vec(path_mat)

        # HARD obstacles: predictions per scenario and geometry (w, l)
        hard_blocks = []
        hard_geom = []
        hard_probs = []
        for obs in hard_obs:
            xpreds, ypreds, psipreds, prob = obs.prediction(self.N, self.dt)
            for s in range(self.S):
                s_idx = min(s, len(xpreds) - 1)
                xb = ca.DM(xpreds[s_idx]).reshape((self.N + 1, 1))
                yb = ca.DM(ypreds[s_idx]).reshape((self.N + 1, 1))
                pb = ca.DM(psipreds[s_idx]).reshape((self.N + 1, 1))
                hard_blocks += [xb, yb, pb]
            hard_geom.append(ca.DM([obs.w, obs.l]).reshape((2, 1)))
            hard_probs.append(ca.DM(prob[:self.S]).reshape((self.S, 1)))

        # scenario weights (normalize first hard obs' probs or use your own policy)
        if len(hard_probs) > 0:
            w_s = hard_probs[0]
            params["scenario_weights"] = w_s / ca.fmax(1e-12, ca.sum1(w_s))
        else:
            # fallback: uniform weights
            params["scenario_weights"] = ca.DM.ones(self.S, 1) / self.S

        params["hard_obs_preds"] = ca.vertcat(*hard_blocks)
        params["hard_obs_geom"]  = ca.vertcat(*hard_geom)

        # SOFT obstacles: single scenario (or your design), same layout
        soft_blocks = []
        soft_geom   = []
        for obs in soft_obs:
            xpreds, ypreds, psipreds, _ = obs.prediction(self.N, self.dt)
            xb = ca.DM(xpreds[0]).reshape((self.N + 1, 1))
            yb = ca.DM(ypreds[0]).reshape((self.N + 1, 1))
            pb = ca.DM(psipreds[0]).reshape((self.N + 1, 1))
            soft_blocks += [xb, yb, pb]
            soft_geom.append(ca.DM([obs.w, obs.l]).reshape((2, 1)))

        params["soft_obs_preds"] = ca.vertcat(*soft_blocks)
        params["soft_obs_geom"]  = ca.vertcat(*soft_geom)

        # non-anticipativity weights
        L = self.max_N_control - 1
        t_plateau = 4
        N_control = max(t_plateau+1, min(int(N_control), L))
        w_plateau = ca.DM.ones(t_plateau, 1) 
        w_tail = ca.linspace(1.0, 0.1, N_control - t_plateau + 1)[1:]  
        w_na = ca.vertcat(w_plateau, w_tail).reshape((N_control, 1))
        if L > N_control:
            w_na = ca.vertcat(w_na, ca.DM.zeros(L - N_control, 1))
        params["non_anticipatory_weights"] = w_na

        # reach-based corridor (8 * N_nodes), from dc_cons
        params["reachbased"] = self._pack_reachbased(dc_cons)

        concat_order = [
            "x_init",
            "x_goal",
            "hard_obs_preds",
            "hard_obs_geom",
            "scenario_weights",
            "non_anticipatory_weights",
            "path_ref",
            "soft_obs_preds",
            "soft_obs_geom",
            "reachbased",
        ]

        c_p = ca.vertcat(*[params[k] for k in concat_order])
        return c_p
    
    def _pack_reachbased(self, dc_cons):
        """
        Returns a ca.DM vector of length 8 * self.N_nodes, stacked block-wise:
        [all e_c_min(k), all e_c_max(k), all mx_min(k), my_min, c_min, mx_max, my_max, c_max],
        where the per-node order uses the SAME global node index k as in add_driving_corridor_constraints.
        """
        S, N = self.S, self.N
        stride = self.N_nodes  # number of iterated nodes

        # Normalize input to (S, N+1, 8)
        import numpy as np
        if isinstance(dc_cons, ca.DM):
            dc_np = np.array(dc_cons.full())
        else:
            dc_np = np.asarray(dc_cons)

        if dc_np.ndim == 2 and dc_np.shape == (S*(N+1), 8):
            dc_np = dc_np.reshape(S, N+1, 8)
        elif not (dc_np.ndim == 3 and dc_np.shape == (S, N+1, 8)):
            raise ValueError(f"dc_cons must be (S*(N+1),8) or (S,N+1,8); got {dc_np.shape}")

        # Allocate per-field arrays in global-node order k
        fields = [np.zeros((stride, 1)) for _ in range(8)]

        # Map (s,t) -> global k using your tree, then place values at k
        # Note: if your tree is a full grid (no pruning), k may equal s*(N+1)+t,
        # but we use _kth_node for correctness.
        for s, t, _ in self.iter_tree_nodes(self.parameters["scenario_weights"]):
            k = self._kth_node(s, t)
            vals = dc_np[s, t, :]  # 8-tuple at this scenario-time node
            for j in range(8):
                fields[j][k, 0] = vals[j]

        # Now stack block-wise: all mins first, then max, then lines…
        blocks = [ca.DM(f) for f in fields]  # each (stride,1)
        reach_vec = ca.vertcat(*blocks)      # (8*stride, 1)
        return reach_vec
    
