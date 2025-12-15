"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
#!/usr/bin/env python
import logging
import casadi as ca
import numpy as np
from common_utils.time_tracking import timeit

from planner.planning_model.BaseMPCC import BaseMPCC

logger = logging.getLogger(__name__)

class SCMPCC(BaseMPCC):

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
        # slacks
        decision_variables["path_slack"] = ca.MX.sym("path_slack", self.S) # one per scenario
        decision_variables["collision_slack"] = ca.MX.sym("collision_slack", self.num_hard_obs*self.S) # for each obstacle per scenario
        self.decision_variables = decision_variables

        concat_order = ["U", "X", "path_slack", "collision_slack"]

        self.opt_variables = ca.vertcat(*[ca.vec(decision_variables[key]) for key in concat_order])
    
    def init_parameter_variables(self):
        """
        Create all symbolic parameter variables used by the MPCC problem.

        Parameters include the initial and terminal reference states, scenario
        probabilities, predicted obstacle trajectories and geometries for both
        hard and soft constraints, non-anticipativity weights and path reference
        data. These quantities are treated as fixed inputs to the optimization 
        problem and can be updated at each solve without rebuilding the symbolic 
        structure.

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
        parameters["path_ref"] = ca.MX.sym("path", self.M*5) # [x, y, psi, w_left, w_right]

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
        ]
        self.P = ca.vertcat(*[parameters[key] for key in concat_order])

    def init_path_refpoints(self):
        """
        Initialize symbolic reference-path and lane-width values along the arc-length coordinate.

        The path reference is supplied as discretized samples of position, heading,
        and left/right lane widths over an arc-length grid. These samples are
        converted into CasADi interpolants so that, for each scenario–time node, the
        reference values can be evaluated at the ego vehicle’s current arc-length
        state.

        For every node, this yields the reference centerline pose `(x_ref, y_ref, psi_ref)`
        together with local corridor half-widths `w_l` and `w_r`, which can be used
        by lane-keeping costs and corridor constraints.

        Returns
        -------
        None
            The method populates the internal lists `X_ref`, `Y_ref`, `Psi_ref`,
            `W_l`, and `W_r` with symbolic expressions evaluated at each node’s
            arc-length state.
        """
        # retrieve path reference variables
        path_ref = self.parameters["path_ref"]
        #path_mat = ca.reshape(path_ref, 5, self.M).T
        M = self.M
        X_path   = path_ref[0*M : 1*M]
        Y_path   = path_ref[1*M : 2*M]
        Psi_path = path_ref[2*M : 3*M]
        Wl_path  = path_ref[3*M : 4*M]
        Wr_path  = path_ref[4*M : 5*M]

        # define arc-length grid
        S_path = np.linspace(0.0, float(self.s_final), int(self.M))
        # create casadi parametric interpolants
        lut_x   = ca.interpolant("LUT_x",   "linear", [S_path])
        lut_y   = ca.interpolant("LUT_y",   "linear", [S_path])
        lut_psi = ca.interpolant("LUT_psi", "linear", [S_path])
        lut_wl  = ca.interpolant("LUT_wl",  "linear", [S_path])
        lut_wr  = ca.interpolant("LUT_wr",  "linear", [S_path])

        # interpolate along the vehicle arc-length s = self.X[6, k]
        s_idx = 6  # s is the 7th state: [x,y,psi,vx,ax,delta,s]

        for k in range(self.N_nodes):
            s_k = self.decision_variables["X"][s_idx, k]

            # interpolate symbolically
            x_ref   = lut_x(s_k,   X_path)
            y_ref   = lut_y(s_k,   Y_path)
            psi_ref = lut_psi(s_k, Psi_path)
            w_l     = lut_wl(s_k,  Wl_path)
            w_r     = lut_wr(s_k,  Wr_path)

            # store references
            self.X_ref.append(x_ref)
            self.Y_ref.append(y_ref)
            self.Psi_ref.append(psi_ref)
            self.W_l.append(w_l)
            self.W_r.append(w_r)

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

    def init_nonlinear_constraints(self):
        """
        Initialize all nonlinear equality and inequality constraints of the MPC problem,
        including hard collision-avoidance constraints.

        The assembled constraint set comprises:
        - Initial-condition and system-dynamics equalities.
        - g–g diagram inequalities enforcing feasible combined accelerations.
        - Hard collision constraints to prevent overlap between the ego vehicle and
        predicted obstacle shapes.
        - Driving-corridor inequalities restricting the ego vehicle to a safe tube
        around the reference path.

        Each constraint block is stored with its corresponding lower and upper bounds,
        then concatenated in a fixed order to form the single constraint vector and
        bound vectors expected by the NLP solver. The concatenation order must remain
        consistent with the solver setup and any post-processing of multipliers.

        Returns
        -------
        None
            Populates internal dictionaries for individual constraint blocks and
            constructs the concatenated constraint vector `self.g` along with its
            bounds `self.lbg` and `self.ubg`.
        """
        g_dict, lbg_dict, ubg_dict = {}, {}, {}
        self.add_equality_constraints(g_dict, lbg_dict, ubg_dict)
        self.add_gg_diamond_constraints(g_dict, lbg_dict, ubg_dict)
        self.add_collision_hard_constraints(g_dict, lbg_dict, ubg_dict)
        self.add_driving_corridor_constraints(g_dict, lbg_dict, ubg_dict)
        self.nonlinear_constraints = g_dict
        self.nonlinear_constraint_lbg = lbg_dict
        self.nonlinear_constraint_ubg = ubg_dict

        # define concatenation order (important for solver alignment)
        concat_order = ["ic_eq", "dynamics_eq", "gg_ineq", "path_ineq", "collision_ineq"]

        # concatenate into solver vectors
        self.g   = ca.vertcat(*[g_dict[key]   for key in concat_order])
        self.lbg = ca.vertcat(*[lbg_dict[key] for key in concat_order])
        self.ubg = ca.vertcat(*[ubg_dict[key] for key in concat_order])

    def init_box_constraints(self):
        """
        Initialize all simple (box) bounds on decision variables, including slack variables.

        This routine assembles independent lower and upper limits for controls,
        states, and slack variables. Slack variables soften hard feasibility
        conditions—such as corridor or collision constraints—and their bounds
        determine how much constraint violation is permitted at a given penalty.

        Bounds for each category are collected in dictionaries for clarity, then
        concatenated into flat vectors matching the solver’s decision-variable
        ordering:
            [controls; states; path_slack; collision_slack].

        A symbolic parameter representing the maximum velocity is introduced so
        that velocity bounds can be adapted at solve time without rebuilding the
        symbolic problem.

        Returns
        -------
        None
            The method stores dictionary-form and concatenated vector-form
            bounds in `self.box_constraints_lbx`, `self.box_constraints_ubx`,
            `self.lbx`, and `self.ubx`.
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
                            lbx_dict["path_slack"],
                            lbx_dict["collision_slack"])
        self.ubx = ca.vertcat(ca.vec(ubx_dict["controls"]),
                            ca.vec(ubx_dict["states"]),
                            ubx_dict["path_slack"],
                            ubx_dict["collision_slack"])

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
            Dictionary of lower bounds, updated with entries for `"path_slack"` and
            `"collision_slack"`.
        ubx : dict
            Dictionary of upper bounds, updated with entries for `"path_slack"` and
            `"collision_slack"`.

        Returns
        -------
        tuple
            Updated `(lbx, ubx)` dictionaries including box constraints for all slack variables.
        """
        path_slack_numel = self.decision_variables["path_slack"].numel()
        lbx["path_slack"] = -ca.DM.ones(path_slack_numel) * self.slack_scale
        ubx["path_slack"] = ca.SX.ones(path_slack_numel) * self.slack_scale

        collision_slack_numel = self.decision_variables["collision_slack"].numel()
        lbx["collision_slack"] = -ca.DM.zeros(collision_slack_numel)
        ubx["collision_slack"] = ca.SX.ones(collision_slack_numel) * self.slack_scale

        return lbx, ubx

    def add_slack_costs(self, cost):
        """
        Add penalty terms for slack variables to the MPC objective.

        Path-slack penalties are quadratic, encouraging small and symmetric
        violations of the driving corridor while avoiding abrupt kinks in the
        optimized trajectory. Collision-slack penalties are linear, strongly
        discouraging obstacle penetration but still allowing limited violations
        when necessary for feasibility.

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
        cost += self.q_ps * ca.sum1(self.decision_variables["path_slack"]**2) # quadratic cost on path violation
        cost += self.q_ho * ca.sum1(self.decision_variables["collision_slack"]) # cost on obstacle violation
        return cost

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
            node_cost += self.q_a*((1/10.0) * ca.log1p(ca.exp(10.0 * (xk[4]-0.5)))) # additional cost on positive acceleration
            cost += w * node_cost
        
        return cost

    def add_collision_hard_constraints(self, g, lbg, ubg):
        """
        Add hard collision-avoidance constraints with slack for all scenarios and time nodes.

        For each ego state and each hard obstacle, a super-Gaussian potential
        `phi_s_o_k` is constructed in the obstacle frame using an L^r
        (super-elliptic) rectangle approximation. The radii in longitudinal and
        lateral directions are scaled by the ego–obstacle closing speed and by
        the obstacle geometry, yielding a conservative, velocity-aware safety
        envelope in front of the ego vehicle.

        A slack variable per obstacle–scenario pair relaxes the hard
        non-penetration condition via inequalities of the form

            phi_s_o_k - slack_{s,o} <= 0,

        so that large potentials (indicating proximity or overlap) must be
        compensated by activating slack. Small side-effect cost terms in `phi`
        and `phi^2` are added directly to the global objective for numerical
        conditioning and to bias the optimizer away from high-potential regions.

        Parameters
        ----------
        g : dict
            Dictionary of nonlinear constraint expressions. The key
            `"collision_ineq"` is populated with the stacked collision
            inequality constraints.
        lbg : dict
            Dictionary of lower bounds for each constraint block; collision
            constraints are one-sided (−∞, 0].
        ubg : dict
            Dictionary of upper bounds for each constraint block.

        Returns
        -------
        tuple
            Updated `(g, lbg, ubg)` dictionaries including the hard
            collision-avoidance constraints and their bounds.
        """
        obs_preds = self.parameters["hard_obs_preds"]
        obs_geom = self.parameters["hard_obs_geom"]
        X = self.decision_variables["X"]
        collision_slack = self.decision_variables["collision_slack"]

        collision_constraints, lower_bounds, upper_bounds = [], [], []

        r = 12 # L^r aggregator order (super-Gaussian)

        # timestep decay with plateau
        w_plateau = self.get_time_decay_weights()


        for s, t, w in self.iter_tree_nodes(self.parameters["scenario_weights"]):
            k = self._kth_node(s, t)

            # Ego pose 
            px = X[0, k]
            py = X[1, k]
            psi  = X[2, k]
            cpsi = ca.cos(psi)
            spsi = ca.sin(psi)

            # front offset 
            off_front = 3.0
            ego_x_front = px + off_front * cpsi
            ego_y_front = py + off_front * spsi

            # Ego velocity (root speed, as in your code)
            v_e   = X[3, 0]
            vx_e  = v_e * cpsi
            vy_e  = v_e * spsi

            for o in range(self.num_hard_obs):
                # obstacle state at (o, s, t)
                base = o * 3 * (self.N + 1) * self.S + s * 3 * (self.N + 1)
                obs_x   = obs_preds[base + t]
                obs_y   = obs_preds[base + (self.N + 1) + t]
                obs_psi = obs_preds[base + 2 * (self.N + 1) + t]

                # geometry (length then width) and trigs
                obs_l = obs_geom[o * 2 + 1]         # length
                obs_w = obs_geom[o * 2]     # width
                cO = ca.cos(obs_psi)
                sO = ca.sin(obs_psi)

                # world deltas to ego front
                ex_w = ego_x_front - obs_x
                ey_w = ego_y_front - obs_y

                # express in obstacle frame
                ex_v = ex_w * cO + ey_w * sO
                ey_v = ex_w * sO - ey_w * cO

                # closing-speed–scaled radii 
                ego_radius = self.width / 2.0
                dx   = obs_x - px
                dy   = obs_y - py
                dist = ca.sqrt(dx * dx + dy * dy) + 1e-6
                v_closing = ca.fmax(0.0, (vx_e * dx + vy_e * dy) / dist)

                d_lat  = obs_w / 2.0 + ego_radius * (0.99 + 0.002 * v_closing)
                d_long = obs_l / 2.0 + obs_l / 10.0 + ego_radius * (1 + 0.12 * v_closing)

                # super-Gaussian L^r rectangle approximation 
                # φ = decay^(2*t+1) * w_s * exp( - ( (|ex_v|/d_long)^r + (|ey_v|/d_lat)^r + eps )^(4/r) )
                phi_s_o_k = w_plateau[t] * w * ca.exp(
                    -(( (ex_v / d_long) ** r + (ey_v / d_lat) ** r + 1e-6) ** (1.0 / r)) ** 4
                )

                self.cost = self.cost + 1e-4*self.q_ho * phi_s_o_k + 1e-4*self.q_ho * phi_s_o_k**2 # side effect on cost for conditioning # TODO: necessary?

                collision_constraints.append(phi_s_o_k - collision_slack[o + self.num_hard_obs * s])
                lower_bounds.append(-ca.inf)
                upper_bounds.append(0)

        g["collision_ineq"] = ca.vertcat(*collision_constraints)
        lbg["collision_ineq"] = ca.vertcat(*[ca.DM([lb]) for lb in lower_bounds])
        ubg["collision_ineq"] = ca.vertcat(*[ca.DM([ub]) for ub in upper_bounds])

        return g, lbg, ubg

    def add_driving_corridor_constraints(self, g, lbg, ubg):
        """
        Add driving-corridor inequality constraints with scenario-level slack variables.

        The ego vehicle is restricted to remain within a lane corridor around the
        reference path, expressed in terms of contouring errors of the front
        vehicle corners relative to the path normal. At each scenario–time node,
        the left and right front corners must lie within the left and right lane
        widths `w_l` and `w_r`, respectively.

        Scenario-level slack variables relax these bounds uniformly across all
        nodes of a given scenario, allowing temporary corridor violations at a
        penalized cost. This yields soft corridor constraints of the form

            e_c_left  ≤  w_l − slack_s
            e_c_right ≥ −w_r + slack_s,

        which are encoded as one-sided inequalities in the constraint vector.

        Parameters
        ----------
        g : dict
            Dictionary of nonlinear constraint expressions. The key `"path_ineq"`
            is populated with the stacked corridor inequality constraints.
        lbg : dict
            Dictionary of lower bounds corresponding to each constraint block.
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
        for s, t, _ in self.iter_tree_nodes(self.parameters["scenario_weights"]):
            k = self._kth_node(s, t)

            # vehicle pose
            px   = X[0, k]
            py   = X[1, k]
            psi  = X[2, k]
            cpsi = ca.cos(psi)
            spsi = ca.sin(psi)

            # front axle position
            x_fa = px + self.wheelbase * cpsi
            y_fa = py + self.wheelbase * spsi

            # front corners
            half_w = 0.5 * self.width
            x_fl = x_fa - half_w * spsi
            y_fl = y_fa + half_w * cpsi
            x_fr = x_fa + half_w * spsi
            y_fr = y_fa - half_w * cpsi

            # path references at node k
            x_ref   = self.X_ref[k]
            y_ref   = self.Y_ref[k]
            psi_ref = self.Psi_ref[k]
            w_l     = self.W_l[k]
            w_r     = self.W_r[k]

            # contouring errors for corners
            e_c_left  = self.get_e_c(x_fl, y_fl, x_ref, y_ref, psi_ref)
            e_c_right = self.get_e_c(x_fr, y_fr, x_ref, y_ref, psi_ref)

            # scenario-level slack
            slack_s = self.decision_variables["path_slack"][s]

            # Left:  e_c_left <= w_l - slack  -> e_c_left - w_l + slack <= 0
            path_constraints.append(e_c_left - w_l + slack_s)
            lower_bounds.append(-ca.inf)
            upper_bounds.append(0)
            # Right: e_c_right >= -w_r + slack -> e_c_right + w_r - slack >= 0
            path_constraints.append(e_c_right + w_r - slack_s)
            lower_bounds.append(0)
            upper_bounds.append(ca.inf)

        g["path_ineq"] = ca.vertcat(*path_constraints)
        lbg["path_ineq"] = ca.vertcat(*[ca.DM([lb]) for lb in lower_bounds])
        ubg["path_ineq"] = ca.vertcat(*[ca.DM([ub]) for ub in upper_bounds])

        return g, lbg, ubg

    def build_mpc_initial_values(self, x0, v_max, **kwargs):
        """
        Build initial guesses for all primal and dual variables of the MPC NLP,
        including slack variables.

        The initial state trajectory is either taken from the previous optimal
        solution (warm-start) or generated by an external warm-start routine based
        on the current state, the velocity limit, horizon length, and sampling
        time. Controls are initialized from the previous solution when available
        or set to zero otherwise.

        Scenario-level path slacks and collision slacks are initialized to zero,
        corresponding to a constraint-satisfying configuration if possible.
        Lagrange multipliers for decision variables and nonlinear constraints
        are also initialized to zero.

        All decision variables are then stacked into a single initial guess vector
        consistent with the internal concatenation order:
            [U; X; path_slack; collision_slack].

        Parameters
        ----------
        x0 : array_like or casadi.DM
            Current ego state used as the starting state of the horizon.
        v_max : float
            Maximum allowed velocity used by the warm-start generator.
        **kwargs :
            Additional keyword arguments reserved for future extensions or for
            forwarding options to the warm-start routine.

        Returns
        -------
        tuple
            opt0_ : casadi.DM
                Initial guess for the full decision vector `[U; X; path_slack; collision_slack]`.
            x0_ : casadi.DM
                Initial guess for the state trajectory over all nodes.
            u0_ : casadi.DM
                Initial guess for the control trajectory over all edges.
            lamx0_ : casadi.DM
                Initial guess for multipliers associated with decision variables.
            lamg0_ : casadi.DM
                Initial guess for multipliers associated with nonlinear constraints.
        """
        lamx0_ = ca.DM.zeros(self.n_states * self.N_nodes + self.n_controls * self.N_edges + self.S + self.S * self.num_hard_obs) # states + controls + slacks
        lamg0_ = ca.DM.zeros(self.g.size()) # nonlinear constraints

        path_slack0_ = ca.DM.zeros(self.S)
        collision_slack0_ = ca.DM.zeros(self.num_hard_obs * self.S)

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
        opt0_= ca.vertcat(ca.vec(u0_.T), ca.vec(x0_.T), path_slack0_, collision_slack0_)

        return opt0_, x0_, u0_, lamx0_, lamg0_

    def build_mpc_param_values(self, x0, xref, hard_obs, soft_obs, path, wl, wr, v_max, **kwargs):
        """
        Assemble the parameter vector for the MPC problem for a single-shot scenario setup.

        This routine constructs all time-varying parameters from the current state,
        reference path, obstacle predictions, and lane-width information. It performs:

        - Initialization of initial and terminal (goal) states.
        - Construction of the reference path with associated left/right lane widths.
        - Stacking of predicted trajectories and geometries for hard obstacles over
        all scenarios.
        - Stacking of predicted trajectories and geometries for soft obstacles
        (typically a single scenario per obstacle).
        - Construction of non-anticipativity weights for the control horizon.

        The resulting parameter blocks are concatenated into a single vector in a
        fixed ordering consistent with the symbolic parameter structure defined in
        `init_parameter_variables`.

        Parameters
        ----------
        x0 : array_like
            Current ego state used as the initial condition.
        xref : array_like
            Desired terminal reference state (goal configuration).
        hard_obs : list
            List of hard obstacle objects providing `prediction(N, dt)` and
            geometry `(w, l)`.
        soft_obs : list
            List of soft obstacle objects treated via soft collision costs.
        path : array_like, shape (M, 3)
            Discretized reference path samples `[x_ref, y_ref, psi_ref]`.
        wl : array_like
            Left lane widths or offsets along the path.
        wr : array_like
            Right lane widths or offsets along the path.
        v_max : float
            Maximum velocity for the current scenario (not used directly here,
            but part of the MPC context).
        **kwargs :
            Additional keyword arguments reserved for future extensions.

        Returns
        -------
        casadi.DM
            Concatenated parameter vector containing initial/terminal states,
            obstacle predictions and geometry, scenario weights, non-anticipativity
            weights, and path/lane information.
        """
        N_control = 15
        # states
        params = {}
        params["x_init"] = ca.vec(ca.DM(x0))
        params["x_goal"] = ca.vec(ca.DM(xref))
        # path-ref
        wl = self.check_road_boundary(wl)
        wr = self.check_road_boundary(wr)
        path_mat = ca.hcat([
            path[:, 0],
            path[:, 1],
            path[:, 2],
            ca.DM(wl).reshape((self.M, 1)),
            ca.DM(wr).reshape((self.M, 1))
        ])
        params["path_ref"] = ca.vec(path_mat)

        hard_blocks = []
        hard_geom = []
        hard_probs =[]
        for o, obs in enumerate(hard_obs):
            xpreds, ypreds, psipreds, prob = obs.prediction(self.N, self.dt)
            # per scenario s: [x(0..N); y(0..N); psi(0..N)]
            for s in range(self.S):
                s_idx = min(s, len(xpreds) - 1)
                xb = ca.DM(xpreds[s_idx]).reshape((self.N + 1, 1))
                yb = ca.DM(ypreds[s_idx]).reshape((self.N + 1, 1))
                pb = ca.DM(psipreds[s_idx]).reshape((self.N + 1, 1))
                hard_blocks += [xb, yb, pb]
            # geometry [w, l]
            hard_geom.append(ca.DM([obs.w, obs.l]).reshape((2, 1)))
            hard_probs.append(ca.DM(prob[:self.S]).reshape((self.S, 1)))

        w_s = hard_probs[0]
        params["scenario_weights"] = w_s / ca.fmax(1e-12, ca.sum1(w_s))
        params["hard_obs_preds"] = ca.vertcat(*hard_blocks)
        params["hard_obs_geom"]  = ca.vertcat(*hard_geom)

        soft_blocks = []
        soft_geom   = []
        for o, obs in enumerate(soft_obs):
            xpreds, ypreds, psipreds, _ = obs.prediction(self.N, self.dt)
            xb = ca.DM(xpreds[0]).reshape((self.N + 1, 1))
            yb = ca.DM(ypreds[0]).reshape((self.N + 1, 1))
            pb = ca.DM(psipreds[0]).reshape((self.N + 1, 1))
            soft_blocks += [xb, yb, pb]
            soft_geom.append(ca.DM([obs.w, obs.l]).reshape((2, 1)))

        params["soft_obs_preds"] = ca.vertcat(*soft_blocks)
        params["soft_obs_geom"]  = ca.vertcat(*soft_geom)

        # non-anticipativity
        L = self.max_N_control - 1
        t_plateau = 4
        N_control = max(t_plateau + 1, min(int(N_control), L))
        w_plateau = ca.DM.ones(t_plateau, 1) 
        w_tail = ca.linspace(1.0, 0.1, N_control - t_plateau + 1)[1:]  
        w_na = ca.vertcat(w_plateau, w_tail).reshape((N_control, 1))
        if L > N_control:
            w_na = ca.vertcat(w_na, ca.DM.zeros(L - N_control, 1))
        params["non_anticipatory_weights"] = w_na

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
        ]

        c_p = ca.vertcat(*[params[key] for key in concat_order])
        return c_p
