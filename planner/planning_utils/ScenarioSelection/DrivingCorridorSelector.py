"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import numpy as np
from typing import List, Tuple
import time
import copy 
import casadi as ca
from planner.planning_utils.obstacles import Obstacle

import logging
logger = logging.getLogger(__name__)

class Zonotope:
    def __init__(self, center, generators):
        self.center = center
        self.generators = generators


    def linear_transform(self, A):
        new_center = A @ self.center
        new_generators = A @ self.generators 
        return Zonotope(new_center, new_generators)


    def minkowski_sum(self, other_center, other_generators):
        new_center = self.center + other_center
        new_generators = ca.horzcat(self.generators, other_generators)# !!!!!!!!!!!!!!!!  np.hstack((self.generators, other_generators))
        return Zonotope(new_center, new_generators)


    def propagate(self, A, B, U_center, U_generators):
        U_transformed_center = B @ U_center
        U_transformed_generators = B @ U_generators

        new_center =  A @ self.center + U_transformed_center 
        new_generators =  ca.horzcat( A @ self.generators, U_transformed_generators ) #

        return Zonotope(new_center, new_generators)
    

    def zonotope_intersection(self, Z1, Z2):
        # Extract min/max bounds for Z1
        x_center1 = Z1.center[0]
        x_generators1 = Z1.generators[0, :]
    
        x_radius1 = ca.sum1(ca.sum2(ca.fabs(x_generators1)))
        x_min1 = x_center1 - x_radius1
        x_max1 = x_center1 + x_radius1
        
        v_center1 = Z1.center[1]
        v_generators1 = Z1.generators[1, :]
        v_radius1 = ca.sum1(ca.sum2(ca.fabs(v_generators1)))
        v_min1 = v_center1 - v_radius1
        v_max1 = v_center1 + v_radius1
    
        # Extract min/max bounds for Z2
        x_center2 = Z2.center[0]
        x_generators2 = Z2.generators[0, :]
        x_radius2 = ca.sum1(ca.sum2(ca.fabs(x_generators2)))
        x_min2 = x_center2 - x_radius2
        x_max2 = x_center2 + x_radius2
        
        v_center2 = Z2.center[1]
        v_generators2 = Z2.generators[1, :]
        v_radius2 = ca.sum1(ca.sum2(ca.fabs(v_generators2)))
        v_min2 = v_center2 - v_radius2
        v_max2 = v_center2 + v_radius2
    
        # Calculate the intersection bounds
        x_min_inter = ca.fmax(x_min1, x_min2)
        x_max_inter = ca.fmin(x_max1, x_max2)
        v_min_inter = ca.fmax(v_min1, v_min2)
        v_max_inter = ca.fmin(v_max1, v_max2)

        # Calculate the center and generators of the intersection zonotope
        x_center = (x_min_inter + x_max_inter) / 2
        x_radius = (x_max_inter - x_min_inter) / 2
        v_center = (v_min_inter + v_max_inter) / 2
        v_radius = (v_max_inter - v_min_inter) / 2
        
        center = ca.vertcat(x_center, v_center)
        generators = ca.diag(ca.vertcat(x_radius, v_radius))
        
        return Zonotope(center, generators)


class DrivingCorridor:
    def __init__(
            self, lane_id: int, mode_id: int, N: int, dt: float, 
            v_max: float, v_min: float, a_max: float, a_min: float, f:int,
            ):

        self.reach_sets: List[Zonotope] = []
        self.dc = None
        self.N = N
        self.dt = dt
        self.A = np.array([
                [1, dt],
                [0, 1]
            ])

        self.B = np.array([
                [0.5*dt**2],
                [dt]
                ])

        self.lane_id = lane_id
        self.mode_id = mode_id

        self.v_max = v_max
        self.v_min = v_min
        self.a_max = a_max
        self.a_min = a_min

        self.reach_sets_initial: List[Zonotope] = []

        self.k_lc1 = -1 #self.x_center_lc = []
        self.k_lc2 = -1 #
        #self.x_generators_lc =[]
        self.k_lcb = []#self.x_center_lcb = []
        #self.x_generators_lcb =[]

        self.k_long = []#self.x_center_long = [] #self.x_center_long = []
        
        #self.x_generators_long =[]

        self.k_mode = []#self.x_center_mode = []
        #self.x_generators_mode =[]
        self.f = f

        self.f_dc = None#self.init_casadi_function(obs_num)
        self.f_cost = None#self.init_casadi_function(obs_num)
        self.f_brs = None

        self.cost = None


    def compute_reachable_sets(self, Z0: Zonotope, obstacles: List[Obstacle]) -> List[Zonotope]:
        # Input constraints
        c_u = np.array([(self.a_max + self.a_min) / 2])
        G_u = np.array([[(self.a_max - self.a_min) / 2]])

        # Propagate zonotope over time and store each zonotope
        Z = Z0
        for k in range(len(self.reach_sets),self.N):
            Z = Z.propagate(self.A, self.B, c_u, G_u)
            Z = self.enforce_velocity_constraints(Z)

            if self.f:
                Z = self.get_collisionfree_zonotopes_f(Z, obstacles,k)
            else:
                Z = self.get_collisionfree_zonotopes_b(Z, obstacles,k)

            self.reach_sets.append(Z)

        return self.reach_sets#, self.k_long#, x_center_n2,x_generators_n2, k_n2


    def enforce_velocity_constraints(self, Z: Zonotope) -> Zonotope:
        v_center = Z.center[1]
        v_generators = Z.generators[1, :]
        v_radius = ca.sum1(ca.sum2(ca.fabs(v_generators)))
        v_center_n = v_center
        v_generators_n = v_generators

        v_max =ca.fmin(v_center + v_radius, self.v_max)
        v_min = ca.fmax(v_center - v_radius, self.v_min)
        v_center_n = (v_max + v_min)/2
        v_generators_n = v_generators*(v_max-v_min) / (2*( v_radius) + 1e-9)
        Z.generators[1, :] = v_generators_n
        Z.center[1] = v_center_n
        return Z


    def get_collisionfree_zonotopes_f(self, Z: Zonotope, obstacles: List[Obstacle], k:int) -> List[Zonotope]:

        x_center = Z.center[0]
        x_generators = Z.generators[0, :]
        x_radius = ca.sum1(ca.sum2(ca.fabs(x_generators))) + 1e-9
        x_min = x_center - x_radius
        x_max = x_center + x_radius
        for o, c_ob in enumerate(obstacles):

            a,b = c_ob[k,0], c_ob[k,1]

            x_center_n = ca.if_else(
                ca.logic_or(x_max <= a ,x_min >= b),
                x_center,
                ca.if_else(
                    ca.logic_and(a > x_min, b < x_max),
                    (x_min + a)/2,
                    ca.if_else(x_min < a, (x_min + a)/2, ca.if_else(x_max>b, (x_max + b)/2, x_center))
                )
            )

            x_generators_n = ca.if_else(
                ca.logic_or(x_max <= a ,x_min >= b),
                x_generators,
                ca.if_else(
                    ca.logic_and(a > x_min, b < x_max),
                    x_generators*(a-x_min) / (2*(x_radius)),
                    ca.if_else(
                        x_min < a,
                        x_generators*(a-x_min) / (2*(x_radius)),
                        ca.if_else(x_max>b, x_generators*(x_max-b) / (2*(x_radius)), x_generators)
                    )
                )
            )

            x_center = x_center_n
            x_generators = x_generators_n
            x_radius = ca.sum1(ca.sum2(ca.fabs(x_generators)))
            x_min = x_center - x_radius
            x_max = x_center + x_radius

            Z.generators[0, :] = x_generators_n
            Z.center[0] = x_center_n
        return Z#, self.x_center_n2,self.x_generators_n2, self.k_n2


    def get_collisionfree_zonotopes_b(self, Z: Zonotope, obstacles: List[Obstacle], k:int) -> List[Zonotope]:
        # Placeholder implementation
        # In a real implementation, you would modify the Zonotope to remove occupied space
        #Z_list.append(Z)

        x_center = Z.center[0]
        x_generators = Z.generators[0, :]
        x_radius = ca.sum1(ca.sum2(ca.fabs(x_generators)))
        x_min = x_center - x_radius
        x_max = x_center + x_radius
        for o, c_ob in enumerate(obstacles):

            a,b = c_ob[k,0], c_ob[k,1]

            x_center_n = ca.if_else(
                ca.logic_or(x_max <= a ,x_min >= b),
                x_center,
                ca.if_else(
                    ca.logic_and(a > x_min, b < x_max),
                    (x_max + b)/2,
                    ca.if_else(x_min < a, (x_min + a)/2, ca.if_else(x_max > b, (x_max + b)/2, x_center))
                )
            )

            x_generators_n = ca.if_else(
                ca.logic_or(x_max <= a ,x_min >= b),
                x_generators,
                ca.if_else(
                    ca.logic_and(a > x_min, b < x_max),
                    x_generators*(x_max-b) / (2*(x_radius)),
                    ca.if_else(
                        x_min < a,
                        x_generators*(a-x_min) / (2*(x_radius)),
                        ca.if_else(x_max>b, x_generators*(x_max-b) / (2*(x_radius)), x_generators)
                    )
                )
            )
            
            x_center = x_center_n
            x_generators = x_generators_n
            x_radius = ca.sum1(ca.sum2(ca.fabs(x_generators)))
            x_min = x_center - x_radius
            x_max = x_center + x_radius

            Z.generators[0, :] = x_generators_n
            Z.center[0] = x_center_n
        return Z    #, self.x_center_n2,self.x_generators_n2, self.k_n2


    def backwards_reachable_set(self, target_set, reachable_sets1):
        dt = self.dt
        
        A_inv = ca.DM([[ 1. , -0.1],
                       [ 0. ,  1. ]])
        B = ca.DM([[0.5 * dt**2],
                   [dt]])
        U_center = ca.DM([(self.a_max + self.a_min) / 2])
        U_generators = ca.DM([[(self.a_max - self.a_min) / 2]])
        # Create an empty list to store the backward reachable sets
        reachable_sets = [target_set.zonotope_intersection(target_set, reachable_sets1[-1])] # # Start with the target set
        #reachable_sets  = [target_set]
        # Backward propagation through time
        for k in range(self.N-1, 0, -1):
            Zk = reachable_sets[-1]
            Zk = Zk.linear_transform(A_inv) #Zonotope(A_inv@ Zk.center , A_inv@Zk.generators)
            Zk_1 = reachable_sets1[k-1]
            Zk_b = Zk.minkowski_sum(B @ U_center, B @ U_generators)
            # Propagate the current set backward using the system dynamics
            #Zk_b = Zonotope(A_inv@ Potry_diff.center , A_inv@Potry_diff.generators)
    
            Zk_1 = Zk_1.zonotope_intersection(Zk_1,Zk_b )  # Propagate backward
            
            # Add the backward reachable set to the list
            reachable_sets.append(Zk_1)
        
        reachable_sets = reachable_sets[::-1]
        return reachable_sets


    def get_driving_corridor(self,reach_sets):
        x_min = []
        x_max = []
        v_min = []
        v_max = []
        for Z in reach_sets:
            x_center = Z.center[0]
            x_generators = Z.generators[0, :]
            x_radius = ca.sum1(ca.sum2(ca.fabs(x_generators)))
            x_min = ca.vertcat(x_min, x_center - x_radius)
            x_max = ca.vertcat(x_max, x_center + x_radius)

            v_center = Z.center[1]
            v_generators = Z.generators[1, :]
            v_radius = ca.sum1(ca.sum2(ca.fabs(v_generators)))
            v_min = ca.vertcat(v_min, v_center - v_radius)
            v_max = ca.vertcat(v_max, v_center + v_radius)

        dc = ca.horzcat(x_min,x_max,v_min,v_max)
        return dc 


    def get_reach_sets_from_dc(self, dc, N):

        self.reach_sets = []

        # Precompute centers and radii
        x_center = (dc[:, 0] + dc[:, 1]) / 2
        x_radius = (dc[:, 1] - dc[:, 0]) / 2
        v_center = (dc[:, 2] + dc[:, 3]) / 2
        v_radius = (dc[:, 3] - dc[:, 2]) / 2
    
        # Vertically concatenate centers and radii into CasADi matrices
        centers = ca.horzcat(x_center, v_center)
        radii = ca.horzcat(x_radius, v_radius)
    
        for i in range(N):
            center = centers[i, :]
            generator = ca.diag(radii[i, :])
    
            # center_np = center.full().flatten()
            # generator_np = generator.full()
    
            Z = Zonotope(center, generator)
            self.reach_sets.append(Z)

        return self.reach_sets



    def init_casadi_function(self, obs_num):
        """init initial driving corridor"""
        x0  = ca.SX.sym('x0')
        v0  = ca.SX.sym('v0')
        vmax  = ca.SX.sym('vmax')
        Z_o = ca.SX.sym('Z_o', self.N*obs_num,2)
        
        self.v_max = vmax
        O_list = []
        for k in range(obs_num):
            O_list.append(Z_o[k*self.N :(k+1)*self.N,:])
        
        # Calculate reachable sets
        # Initial state zonotope
        c0 = ca.vertcat(x0, v0)
        G0 = ca.SX.zeros(2,2) #np.array([[0, 0], [0, 0]])  # Modify as needed to define initial uncertainty
        Z0 = Zonotope(c0, G0)
        
        reachable_sets = self.compute_reachable_sets(Z0, O_list)
        
        d_c = self.get_driving_corridor(reachable_sets)
        
        # x_min, x_max = ca.SX.sym('xmin1', N),ca.SX.sym('xmax1', N)
        # # x_min2, x_max2 = ca.SX.sym('xmin2', N),ca.SX.sym('xmax2', N)
        # dc = ca.horzcat(x_min, x_max)
        
        self.f_dc = ca.Function('f_dc', [x0,v0,Z_o, vmax], [d_c]) #,  { "compiler": "shell", "jit": True, "jit_options": {  "flags": ["-O1"], "verbose": True}})
        # , obstacle_zonotopes)

        """init cost calculation"""
        dc_out = ca.SX.sym("dc_out", self.N,4)
        cost = 0
        for k in range(self.N):
            cost = cost - (dc_out[k,1]-dc_out[k,0]) - 0.1*dc_out[k,1] 
            
        self.f_cost = ca.Function('f_cost', [dc_out],[cost]) 

        """"init backwards reachable set calculation"""
        A_inv = ca.DM([[ 1. , -0.1],
                       [ 0. ,  1. ]])
        B = ca.DM([[0.5 * self.dt**2],
                   [self.dt]])
    
        dc_frs =  ca.MX.sym('xN',self.N, 4)
        
        # reachable_sets = self.compute_reachable_sets(Z0, O_list)
        reachable_sets1 = self.get_reach_sets_from_dc(dc_frs, self.N)
        reachable_sets  = self.backwards_reachable_set(reachable_sets1[-1], reachable_sets1)
        dc_brs          = self.get_driving_corridor(reachable_sets)

        self.f_brs = ca.Function('f_brs', [dc_frs], [dc_brs])


    def get_cost(self):
        self.cost = self.f_cost(self.dc) + self.N*7*np.abs(self.lane_id) - self.N*(self.k_lc2-self.k_lc1)/(13-1)    # + how long  ca
        if self.lane_id < 0:
            self.cost = self.cost + self.N*np.abs(self.lane_id) #left lane less expensive
        return self.cost


    def empty_dc(self, obs_num):
        self.reach_sets_initial: List[Zonotope] = []

        self.k_lc = []  #self.x_center_lc = []
        #self.x_generators_lc =[]
        self.k_lcb = [] #self.x_center_lcb = []
        #self.x_generators_lcb =[]

        self.k_long = []    #self.x_center_long = [] #self.x_center_long = []
        
        #self.x_generators_long =[]

        self.k_mode = []    #self.x_center_mode = []


class DrivingCorridorExtractor:
    def __init__(
            self, N=40, dt=0.1, v_max=15.0, v_min=0.0, a_max=3.0, a_min=-6.0, obs_num=3, l=3,
            path_ref=None, w_l=[], w_r=[], num_hard_obs=0, M=1001, s_final=1000, lane_width=3.5
            ):
        #Parameters
        self.ego_len = l*2
        self.obs_num = obs_num  # 48m in 4 sec = 12m/s diff --> max 3 ob
        self.a_min = a_min  #*/0.8
        self.a_max = a_max
        self.v_max = v_max
        self.v_min = v_min
        
        self.dt = dt
        self.N = N
        self.M = M 
        self.s_final = s_final
        
        self.S_path = np.linspace(0,self.s_final, self.M) 
        #self.path_ref = path_ref
        
        self.x_ref = path_ref[:,0]
        self.y_ref = path_ref[:,1]
        self.psi_ref = path_ref[:,2]
        self.lut_x = ca.interpolant('LUT1','linear',[self.S_path], self.x_ref) #, X_path)
        self.lut_y = ca.interpolant('LUT2','linear',[self.S_path], self.y_ref)#, Y_path)
        self.lut_phi = ca.interpolant('LUT3','linear',[self.S_path], self.psi_ref)#, Phi_path)    
        self.w_l = w_l
        self.w_r = w_r
        self.lut_wl = ca.interpolant('LUT3','linear',[self.S_path], self.w_l)#, Phi_path)   
        self.lut_wr = ca.interpolant('LUT3','linear',[self.S_path], self.w_r)#, Phi_path)   
    
        self.min_space = 5
        self.lane_width = lane_width 
        self.t_lc = int(np.sqrt(4*self.lane_width/6)/self.dt) - 4 #int(1.2/dt)
        
        self.num_dc = None
        num_init_dc = 14 # 2 current 6 left 6 right
        self.dc = []
        for i in range(num_init_dc):
                self.dc.append(
                    DrivingCorridor(
                        lane_id=0, mode_id=int(i/num_init_dc), N=self.N, dt=self.dt, v_max=self.v_max,
                        v_min=self.v_min, a_max=self.a_max, a_min=self.a_min, f=(i+1)%2,
                    )
                )   #self.dc.append() #0i%2
                self.dc[-1].init_casadi_function(self.obs_num)
    
        self.concatenated_obs_ml = -1000*np.ones((N*obs_num,2))
        self.concatenated_obs_ll = -1000*np.ones((N*obs_num,2))
        self.concatenated_obs_rl = -1000*np.ones((N*obs_num,2))
        self.current_lane_id = None
        self.ll_existing = False
        self.rl_existing = False
        self.F_frenet = None
        self.F_rf = None
        self.init_casadi_function(num_hard_obs)
        # self.ObstacleArray = []
        # self.z0
        self.e_c = None

        s0 = self.S_path
        kappa = self.compute_curvature(self.psi_ref, self.S_path )
        s1 = self.calculate_parallel_curve(self.x_ref, self.y_ref, kappa, self.S_path, self.lane_width)
        s_1 = self.calculate_parallel_curve(self.x_ref, self.y_ref, kappa, self.S_path, -self.lane_width)
        s2 = self.calculate_parallel_curve(self.x_ref, self.y_ref, kappa, self.S_path, 2*self.lane_width)
        s_2 = self.calculate_parallel_curve(self.x_ref, self.y_ref, kappa, self.S_path, -2*self.lane_width)

        self.lut1 = ca.interpolant('LUT','linear',[s0], s1)
        self.lut_1 = ca.interpolant('LUT','linear',[s0], s_1)
        self.lut2 = ca.interpolant('LUT','linear',[s0], s2)
        self.lut_2 = ca.interpolant('LUT','linear',[s0], s_2)

        self.lut1_ = ca.interpolant('LUT','linear',[s1], s0)
        self.lut_1_ = ca.interpolant('LUT','linear',[s_1], s0)
        self.lut2_ = ca.interpolant('LUT','linear',[s2], s0)
        self.lut_2_ = ca.interpolant('LUT','linear',[s_2], s0)


    def init_casadi_function(self, num_hard_obs):
        M = self.M

        p = ca.horzcat(self.x_ref, self.y_ref, self.psi_ref)

        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        s = ca.MX.sym('s')
        d = ca.MX.sym('d')
        
        obs = ca.horzcat(x, y).T
        obs = ca.repmat(obs, 1, M).T
        distances = ca.sqrt(ca.sum2((p[:,:2]- obs)**2))
        

        closest_index = ca.find(distances <= ca.mmin(distances) ) #np.argmin(distances)
        
        #s_pred[i] = path_ref.s[closest_index][1]
        s = closest_index  #path_s[]
        d = (-ca.sin(p[:,2]) * (obs[:,0] - p[:,0]) + ca.cos(p[:,2]) * (obs[:,1] - p[:,1]))[closest_index]
        #d = distances[closest_index]
        f_frenet = ca.Function('f_frenet', [x, y], [s, d])
        self.F_frenet = f_frenet.map(self.N*num_hard_obs)

        x3 = ca.MX.sym('x')
        y3 = ca.MX.sym('y')
        
        s1 = ca.MX.sym('s1')
        s2 = ca.MX.sym('s2')
        s = ca.MX.sym('s')
        
        
        x1,y1 = self.lut_x(s1), self.lut_y(s1)
        x2,y2 = self.lut_x(s2), self.lut_y(s2)
        fac = ((x3 - x1) * ( x2 - x1) + (y3 - y1) * (y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
        s = s1 + fac*(s2 - s1)
        
        f_rf = ca.Function('f_rf', [x3, y3,s1,s2], [s])
        self.F_rf = f_rf.map(self.N*num_hard_obs)


    def get_occupancy_per_lane(self, x0, m, ObstacleArray=[]):
        #self.w_l, self.w_r, self.path_ref
        # get obstacle predictions
        # transform to frenet
        # check for each timestep on which lane obstacle is
        # get road boundaries width  paramterized on s (if lane closes add as obstacle)

        def get_lane_occ(s, orientation, lane_id_obs, lane_id, ObstacleArray=[], obs_num=0):
            N, ego_len = self.N, self.ego_len
            current_lane_id = lane_id # current_lane_id+-1
            mask = (current_lane_id==lane_id_obs) | \
                   ((current_lane_id==lane_id_obs-1) & (orientation==1 )) | \
                   ((current_lane_id==lane_id_obs+1) & (orientation==1 )) # current lane or orientation
            obs = -1000*np.ones((N*obs_num, 1)).T  
            obs[mask] = s[mask]                                                                                                                     
            obs = obs.reshape((obs_num, N)).T   
                
            
            mask = ~np.all(obs == -1000, axis=0) # # Select columns where no element is equal to -1000
            obj_id = np.flatnonzero(mask)
            obs = obs[:, mask].T.reshape(-1)
            obs = np.array(self.correct_arclength_parallel_lane(obs, self.current_lane_id)).reshape(-1)
        
            orientation_reshape = orientation.reshape((obs_num,N))
            mink_sum = (
                (orientation_reshape==0) * np.array([obj.l/2 for obj in ObstacleArray])[:, np.newaxis] + \
                (orientation_reshape==1) * np.array([obj.w/2 for obj in ObstacleArray])[:, np.newaxis] )
            
            mink_sum = mink_sum[mask].reshape(-1) 
            concatenated_obs = np.vstack((obs -mink_sum  - 3*ego_len/4 , obs + mink_sum + 1*ego_len/4)).T
            
            # if lane_id==-1:
            #     print(obs)
            return concatenated_obs

        def get_endlane(x0, concatenated_obs, w, lane_id, last_obj): 
            endlane_idcs = np.where(w<=(lane_id)*(self.lane_width-0.8))
            #print("endlane_idcs", endlane_idcs, lane_id)
            endlane_idx = endlane_idcs[0]
            endlane_idx_last = endlane_idcs[0]
            
            # #solve problem that path_ref length != w length with assumption of equidistant sampling
            # endlane_idx = (endlane_idx*(self.S_path.shape[0]/w.shape[0])).astype(int)
            # endlane_idx_last = (endlane_idx_last*(self.S_path.shape[0]/w.shape[0])).astype(int)
            if endlane_idx.size !=0:
                endlane_x = self.S_path[endlane_idx[0]]
                #print("sdfsds", endlane_x)
                endlane_x_last = self.S_path[endlane_idx_last[-1]]
                if 1: #endlane_x > x0:#-2 or endlane_x_last > x0 :
                    idx = min(last_obj,concatenated_obs.shape[0]-self.N)
                    #if x0>self.correct_arclength_parallel_lane(endlane_x, self.current_lane_id):
                    #print("!!!!!!!",x0,self.correct_arclength_parallel_lane(endlane_x, self.current_lane_id))
                    concatenated_obs[idx:,0] = self.correct_arclength_parallel_lane(endlane_x, self.current_lane_id)  #,x0+0.25)#, x0+0.1)  #endlane_x #
                    
                    #print("endlane", concatenated_obs[idx,0] )
                    concatenated_obs[idx:,1] = 1000000 #self.correct_arclength_parallel_lane(endlane_x_last, self.current_lane_id) #here only ending lane considered not opening     
            return concatenated_obs
        
        #t_ = time.time()
        obs_num = len(ObstacleArray)
        #x0 = z0[-1] #"""!!!!!"""
        N, lane_width, w_l, w_r, current_lane_id = self.N, self.lane_width, self.w_l, self.w_r, self.current_lane_id
        # idcs, d =self.F_frenet(np.concatenate([obj.x_preds[m][:N] for obj in ObstacleArray]),np.concatenate([obj.y_preds[m][:N] for obj in ObstacleArray]))#, path_ref.s[:,1])
        # s = self.path_ref.s[np.array(idcs).astype(int), 1] #0->N, ->2N->O*N

        x3, y3 = np.concatenate([obj.x_preds[min(m, len(obj.x_preds)-1)][:N] for obj in ObstacleArray]),np.concatenate([obj.y_preds[min(m, len(obj.x_preds)-1)][:N] for obj in ObstacleArray])
        idcs, d = self.F_frenet(x3,y3)
        #print("idcs frenet",idcs)
        s1 = self.S_path[np.array(idcs).astype(int)-1]
        s2 = self.S_path[np.minimum(np.array(idcs).astype(int)+1,self.S_path.shape[0]-1)]
        s = self.F_rf(x3, y3, s1, s2)
        s= np.array(s)
        diff_phi = self.psi_ref[np.array(idcs).astype(int)] - np.concatenate([obj.psi_preds[min(m, len(obj.x_preds)-1)][:N] for obj in ObstacleArray])
        orientation = np.round((diff_phi)/(np.pi/2))%(2)  # 0: parallel 1: perpendicular
        #print(orientation)
        lane_id_obs = np.round(d/ lane_width)
        #print(lane_id_obs)
        #print("t0",time.time()-t_)
        #t_ = time.time()
        #figure out ll_existing, rl_existing
        if isinstance(w_l, (int, float)):
            single_w_l = True
            if w_l > lane_width*(current_lane_id+1):
                self.ll_existing = True
                
        elif any(w_l > lane_width*(current_lane_id+1)):
            single_w_l = False
            self.ll_existing = True
        else:
            single_w_l = False
            self.ll_existing = False
            
        if isinstance(w_r, (int, float)):
            single_w_r = True
            if w_r < lane_width*(current_lane_id-1):
                self.rl_existing = True
                
        elif any(w_r < lane_width*(current_lane_id-1)):
            self.rl_existing = True
            single_w_r = False
        else:
            single_w_r = False
            self.rl_existing = False
        concatenated_obs_ml1 = get_lane_occ(s, orientation, lane_id_obs, current_lane_id, ObstacleArray, obs_num)
        self.concatenated_obs_ml[:concatenated_obs_ml1.shape[0],:] = concatenated_obs_ml1[:self.concatenated_obs_ml.shape[0],:]
        if not single_w_l:
            self.concatenated_obs_ml = get_endlane(x0, self.concatenated_obs_ml, w_l, current_lane_id, concatenated_obs_ml1.shape[0])
            self.concatenated_obs_ml = get_endlane(x0, self.concatenated_obs_ml, -w_r, -current_lane_id, concatenated_obs_ml1.shape[0])
        """s= s + 2*np.pi*np.abs(current_lane_id*lane_width)/(s[:,-1]/(2*np.pi)) # check again"""

        if self.ll_existing:
            concatenated_obs_ll1 = get_lane_occ(s, orientation, lane_id_obs, current_lane_id+1, ObstacleArray, obs_num)
            self.concatenated_obs_ll[:concatenated_obs_ll1.shape[0],:] = concatenated_obs_ll1[:self.concatenated_obs_ll.shape[0],:]
            if not single_w_l:
                self.concatenated_obs_ll = get_endlane(x0, self.concatenated_obs_ll, w_l, current_lane_id+1, concatenated_obs_ml1.shape[0])
    
        if self.rl_existing:
            concatenated_obs_rl1 = get_lane_occ(s, orientation, lane_id_obs, current_lane_id-1, ObstacleArray, obs_num)
            self.concatenated_obs_rl[:concatenated_obs_rl1.shape[0],:] = concatenated_obs_rl1[:self.concatenated_obs_rl.shape[0],:]
            if not single_w_l:
                self.concatenated_obs_rl = get_endlane(x0, self.concatenated_obs_rl, -w_r, -current_lane_id+1,concatenated_obs_ml1.shape[0])
        return self.concatenated_obs_ml, self.concatenated_obs_ll, self.concatenated_obs_rl, self.ll_existing, self.rl_existing


    def get_all_driving_corridors(
            self, x0, v0, concatenated_obs_ml, concatenated_obs_ll=None, concatenated_obs_rl=None,
            ll_existing=False, rl_existing=False, current_lane_id=0
            ):
       # dcs on current lane
        dc = self.dc
        num_dc = 0
        dc[0].f, dc[0].lane_id = 1, 0+current_lane_id
        dc[0].dc = dc[0].f_dc(x0,v0,concatenated_obs_ml, self.v_max)
        #print("concatenated_obs_ll", concatenated_obs_ll)
        #print("concatenated_obs_ml", concatenated_obs_ml)
        if self.is_valid_dc(dc, 0, v0): #not self.faulty_dc(dc, 0, v0):
            num_dc += 1
            dc[0].get_cost()
            #print("dc[0].dc",dc[0].cost)    
        dc[num_dc].f, dc[num_dc].lane_id = 0, 0+current_lane_id # was dc[0].lane_id
        dc[num_dc].dc = dc[num_dc].f_dc(x0,v0,concatenated_obs_ml, self.v_max) 
        if self.is_valid_dc(dc, num_dc, v0) and np.sum(np.abs(dc[0].dc[:,:2]-dc[num_dc].dc[:,:2])) > 0.1: # was self.faulty_dc(dc, 1, v0)
            dc[num_dc].get_cost()
            num_dc += 1
        #print("num_dc", num_dc, dc[0].cost)   
        #dcs on left lane
        if ll_existing:
            lane_id = 1 + current_lane_id
            self.t_lc = max(int(np.sqrt(max(0,4*(self.lane_width/2-self.e_c+current_lane_id*self.lane_width)/6))/self.dt) - 3 + 2 - 1, 3) #5
            dc, num_dc = self.get_driving_corridors_neighborlane(x0, v0, dc,num_dc, concatenated_obs_ml, concatenated_obs_ll, lane_id)
            #print("ll", num_dc, dc[num_dc-1].cost)
        
        #dcs on right lane
        if rl_existing:
            lane_id = -1 + current_lane_id
            self.t_lc = max(int(np.sqrt(max(0,4*(self.lane_width/2-self.e_c+current_lane_id*self.lane_width)/6))/self.dt)-3+2-1,3) #int(np.sqrt(4*(self.lane_width+self.e_c-current_lane_id*self.lane_width)/6)/self.dt)-3#5
            dc, num_dc = self.get_driving_corridors_neighborlane(x0, v0, dc,num_dc, concatenated_obs_ml, concatenated_obs_rl,  lane_id)
            #print("rl", num_dc)
            #print("rl", num_dc, dc[num_dc-1].cost)

        #print("t_lc", self.t_lc)
        return dc, num_dc
    
    def get_driving_corridors_neighborlane(self, x0, v0, dc,num_dc, concatenated_obs_l1, concatenated_obs_l2, lane_id):
        filled = num_dc
        t_overtake_start = []
        t_overtake_end = []
        lc_start = False

        self.min_space = 4  # float(0.5*v0*3/2.75)
        #t_ = time.time()
        for k in range(self.N): # overtake in the beginning
            if k > 0 and t_overtake_start == []:
                break
            if (np.all(
                (float(dc[0].dc[k,1]) + self.min_space < concatenated_obs_l2[k::self.N,0]) | \
                (float(dc[0].dc[k,0]) - self.min_space > concatenated_obs_l2[k::self.N,1]))) \
                    and dc[0].dc[k,1] not in concatenated_obs_l1[k::self.N,0]:
                if k==0:
                    t_overtake_start.append(0)
                    t_overtake_end.append(0)
                t_overtake_end[-1] = k
            else:
                break
        
        condition1 = dc[0].dc[:, 1] > (concatenated_obs_l2[:, 1].reshape(-1,self.N).T + self.min_space ) # remove -1000#2D boolean array with shape (N, M) where M is the number of elements in the slice
        idxs = np.where(condition1)
        #print(lane_id, idxs)

        idxs= [np.array(idxs[0]), np.array(idxs[1])]
        current_o = 0
        i = 0
        while len(idxs[0]) >= 1:
            #print(i,current_o, idxs)
            if current_o not in idxs[1]:
                i = 0
                current_o += 1
            if idxs[1][i] != current_o:
                i += 1
                continue
        
            k = idxs[0][i]
            o = idxs[1][i]
            all_conditions_met = True
            if np.any(
                ((concatenated_obs_l2[k::self.N, 1]-concatenated_obs_l2[k::self.N, 0])>100.0) & \
                (np.array(dc[0].dc[k, 1])>concatenated_obs_l2[k::self.N, 0]) & \
                (np.array(dc[0].dc[k, 1])<concatenated_obs_l2[k::self.N, 1])):
                #print("new logic to avoid endlane")
                    all_conditions_met = False #return  dc, num_dc
            
            if np.any(
                (concatenated_obs_l2[k + o * self.N, 1] + self.min_space * 2 >= concatenated_obs_l2[k::self.N, 0]) & \
                (concatenated_obs_l2[k + o * self.N, 1] < concatenated_obs_l2[k::self.N, 1])):
                    all_conditions_met = False
    
            #print(k,o)
            if  all_conditions_met:
                if not lc_start:
                    t_overtake_start.append(k)
                    t_overtake_end.append(k)
                    lc_start= True
                    #print(k,o)
                    #break
                elif t_overtake_end[-1] == k-1 :
                    if dc[0].dc[k,1] not in concatenated_obs_l1[k::self.N,0]:  # if ego muss nicht bremsen
                        t_overtake_end[-1] =k
                    else:
                        lc_start = False
            else:
                lc_start = False
                
            idxs[0] = np.delete(idxs[0], i) 
            idxs[1] = np.delete(idxs[1], i) 

        #print("t1",time.time()-t_)
        t_overtake_start, idx = np.unique(t_overtake_start, return_index=True)
        t_overtake_end = np.array(t_overtake_end)[idx]
        #print(t_overtake_start,t_overtake_end, lane_id )
        lead_obj = [int(self.obs_num-1)]* len(t_overtake_start)
        concatenated_obs_l12 = [concatenated_obs_l1.copy() for _ in range(len(t_overtake_start))] 
        for i, t in enumerate(t_overtake_start):
            if  self.t_lc > t_overtake_end[i]-t and t_overtake_end[i]<self.N: 
                # if t==0:
                #     print("not enoguh time", t, t_overtake_end[i], self.t_lc)
                filled -= 2
                continue # if not enough time for lane change
            t_lc1 = max(self.t_lc, t_overtake_end[i]-t)
            #print("klc",i,t,t_overtake_end[i])
            dc[2*i+filled].k_lc1 = t
            dc[2*i+filled+1].k_lc1 = t
            dc[2*i+filled].k_lc2 = t_overtake_end[i]
            dc[2*i+filled+1].k_lc2 = t_overtake_end[i]

            #get min distance to l1 lead obst at k=t
            obs_values = concatenated_obs_l1[t + np.arange(self.obs_num) * self.N, 0]
            mask = obs_values > float(dc[0].dc[t, 1])
            for o in range(self.obs_num):
                if mask[o] and obs_values[o]<obs_values[lead_obj[i]]:
                    lead_obj[i] = o

            #concatenated_obs_l12 = concatenated_obs_l1*1
            for o in range(self.obs_num):
                concatenated_obs_l12[i][dc[2*i+filled].k_lc1+o*self.N:(o+1)*self.N ,:] = \
                    concatenated_obs_l2[dc[2*i+filled].k_lc1+o*self.N:(o+1)*self.N ,:]
                #print("sdsd", i)
                #print(dc[2*i+filled].k_lc1+o*N)
                if concatenated_obs_l12[i][t+o*self.N,1] < dc[0].dc[t,1]:
                    concatenated_obs_l12[i][dc[2*i+filled].k_lc1+o*self.N:(o+1)*self.N ,0] = \
                        concatenated_obs_l2[dc[2*i+filled].k_lc1+o*self.N:(o+1)*self.N ,0]*0

            lhs = concatenated_obs_l12[i][dc[2*i+filled].k_lc1+(self.obs_num-1)*self.N:dc[2*i+filled].k_lc1+(self.obs_num-1)*self.N + t_lc1 ,:] 
            rhs = concatenated_obs_l1[dc[2*i+filled].k_lc1+lead_obj[i]*self.N:dc[2*i+filled].k_lc1+lead_obj[i]*self.N + t_lc1 ,:]
            concatenated_obs_l12[i][dc[2*i+filled].k_lc1+(self.obs_num-1)*self.N:dc[2*i+filled].k_lc1+(self.obs_num-1)*self.N + t_lc1 ,:] = rhs[:lhs.shape[0],:]
            
            dc[2*i+filled].f = 1
            dc[2*i+filled].dc = dc[2*i+filled].f_dc(x0,v0,concatenated_obs_l12[i], self.v_max)
            if not self.is_valid_dc(dc, 2*i+filled, v0): #self.faulty_dc(dc, 2*i+filled, v0):
                filled -= 1
            else:
                #print("adsad",t)
                num_dc += 1 
                dc[2*i+filled].lane_id = lane_id
                dc[2*i+filled].get_cost()
                #print("adsad1")
                #print("adsad1",dc[2*i+filled].get_cost())
                
            dc[2*i+filled+1].f = 0
            dc[2*i+filled+1].dc = dc[2*i+filled+1].f_dc(x0, v0, concatenated_obs_l12[i], self.v_max)
            if not self.is_valid_dc(dc, 2*i+filled+1, v0) or np.sum(np.abs(dc[2*i+filled+1].dc[:,:2]-dc[2*i+filled].dc[:,:2])) < 0.1: #self.faulty_dc(dc, 2*i+filled+1, v0) or np.sum(np.abs(dc[2*i+filled+1].dc[:,:2]-dc[2*i+filled].dc[:,:2])) < 0.1:
                #print("faulty or double")
                filled -= 1
            else:
                num_dc += 1
                dc[2*i+filled+1].lane_id = lane_id
                dc[2*i+filled+1].get_cost()
                #print("adsad2",dc[2*i+filled+1].get_cost())
        
        return dc, num_dc

    def is_valid_dc(self, dc, i, v0):
        corridor = dc[i].dc
        # CHECK 0 if the object computed a dc
        if corridor is None:
            return False
        
        v0 = float(v0)
        C = np.array(corridor, dtype=float)
        x_lower = C[:, 0]  # "min" bound
        x_upper = C[:, 1]  # "max" bound
        risk = C[:, 2] if C.shape[1] > 2 else None
        width = np.abs(x_lower - x_upper)

        # CHECK ASSUMPTIONS
        assumed_max_decel = 2.4  # based on nuplan constraints
        starting_frame = 3 # first 3 frames are too constraint by default

        # CHECK 1 if the objects dc contains a feasible lane-change:
        # If, at the moment ego begins changing lanes, 
        # the forward motion during the lane change would 
        # already overshoot the tightest longitudinal limit in the target lane, 
        # then this lane change is infeasible.
        if dc[i].k_lc1 != -1:
            k1 = int(dc[i].k_lc1) # before lange change
            k2 = int(dc[i].k_lc2) # after lane change
            if 0 <= k1 < k2 <= len(x_lower) - 1:
                lane_change_steps = (k2 - k1)
                lane_change_time = self.dt * lane_change_steps

                minimum_forward_progress = v0 * lane_change_time - 0.5 * assumed_max_decel * (lane_change_time ** 2)

                start_pos_est = x_lower[k1] + minimum_forward_progress
                future_upper_min = np.min(x_upper[k2:])
                if start_pos_est > future_upper_min:
                    return False
                
        # CHECK 2 if the objects dc frames are too narrow near risk
        if risk is not None and len(width) >= 4:
            mid_width = width[starting_frame:-1]
            mid_risk = risk[starting_frame:-1]  # aligned (less accidental mismatch)

            minimum_forward_progress = (v0 * self.dt - 0.5 * assumed_max_decel * ((self.dt) ** 2))
            risk_threshold = 0.2
            too_narrow_where_risky = (mid_width < minimum_forward_progress / 5.0) & (mid_risk > risk_threshold)
            if np.sum(too_narrow_where_risky) > 1:
                return False

        # CHECK 3 if the objects dc frames are pinching 
        if len(width) >= 4:
            w_prev = width[1:-2]
            w_mid  = width[2:-1]
            w_next = width[3:]
            # frame is narrower then minimum forward progress in 1 step
            minimum_forward_progress = (v0 * self.dt - 0.5 * assumed_max_decel * ((self.dt) ** 2))
            absolute_narrow = (w_mid < minimum_forward_progress * 2.0)
            # (wide |-> narrow |-> wider)
            # midpoint is less than half the previous width and
            # midpoint is at least ~33% narrower than the next width
            sharp_pinch = (2.0 * w_mid < w_prev) & (1.5 * w_mid < w_next) 

            if np.sum(absolute_narrow & sharp_pinch)>1:
                return False

        return True

    def faulty_dc(self, dc, i, v0):

        if dc[i].dc is None:
            return True

        if dc[i].k_lc1!=-1:
            lc_dist = v0*self.dt*(dc[i].k_lc2-dc[i].k_lc1) - 0.5*2.5*(self.dt*(dc[i].k_lc2-dc[i].k_lc1))**2
            if dc[i].dc[dc[i].k_lc1, 0] + lc_dist > np.min(dc[i].dc[dc[i].k_lc2:,1]):
                logger.info("INVALID LANE CHANGE IN DC")
                return True
            

        diff_k = np.abs(dc[i].dc[2:-1,0]-dc[i].dc[2:-1,1])
        diff_km1 = np.abs(dc[i].dc[1:-2,0]-dc[i].dc[1:-2,1])
        diff_kp1 = np.abs(dc[i].dc[3:,0]-dc[i].dc[3:,1])
        breakpoint()
        if np.sum((diff_k<float(v0)*self.dt/5 ) & (np.array(dc[i].dc[:-2-1,2]) > 0.2)) > 1: #any((diff_k<float(v0)*self.dt/5 ) & (np.array(dc[i].dc[:-2-1,2]) > 0.2)):
            logger.info("TOO NARROW FRAME IN DC")
            return True
        
        if any((diff_k<float(v0)*self.dt*2) & (2*diff_k<diff_km1) & (1.5*diff_k<diff_kp1)):
            logger.info("PINCH IN DC")
            return True  #float(v0)*self.& (np.array(dc[i].dc[:-2-1,2]) > 0.2)dt/8

        return False


    def select_best_driving_corridor(self, dc, num_dc):
        def get_cost(dc_item):
            return dc_item.cost
        # Sort the list of driving corridors based on cost
        sorted_dc = sorted(dc[:num_dc], key=get_cost)
        
        # Return both the best (first) and second best (second) corridors
        best_corridor = sorted_dc[0]
        second_best_corridor = sorted_dc[1] if len(sorted_dc) > 1 else sorted_dc[0]  # Handle case where there's only one corridor

        
        return best_corridor, second_best_corridor
        # # Replace lambda with named function
        # return min(dc[:num_dc], key=get_cost) #dc[0]#


    def get_current_lane_id(self, z0):
        # z0[0]+=np.cos(z0[2])*self.ego_len/2
        # z0[1]+=np.sin(z0[2])*self.ego_len/2
        x_ref = self.lut_x(z0[-1]) 
        y_ref = self.lut_y(z0[-1]) 
        phi_ref = self.lut_phi(z0[-1]) 
        self.e_c = float(-np.sin(phi_ref) * (z0[0] - x_ref) + np.cos(phi_ref) * (z0[1]- y_ref))

        self.current_lane_id = round(max(min(self.e_c,float(self.lut_wl(z0[-1]).full())-0.1), float(self.lut_wr(z0[-1]).full())+0.1  )/ self.lane_width)
        #self.current_lane_id = round(self.e_c/self.lane_width)
        
        #print(self.e_c,float(self.lut_wl(z0[-1]).full())-0.1,  float(self.lut_wr(z0[-1]).full())+0.1)
        #print(self.current_lane_id)
        self.e_c = float(-np.sin(phi_ref) * (z0[0]+np.cos(z0[2])*3*self.ego_len/4 - x_ref) + np.cos(phi_ref) * (z0[1]+np.sin(z0[2])*3*self.ego_len/4 - y_ref))
        return self.current_lane_id
    
    def extract_dc_m(self,  z0, m, ObstacleArray=[], v_max=10):
        self.v_max = v_max
        x0, v0 = z0[-1], z0[3] 
        current_lane_id = self.get_current_lane_id(z0)
        #print("current_lane_id ", current_lane_id)
        """achtung s_ref-> s_lane: lieber alles in s_ref lassen"""
        x0 = self.correct_arclength_parallel_lane(x0, current_lane_id)
        concatenated_obs_ml, concatenated_obs_ll, concatenated_obs_rl, ll_existing, rl_existing = self.get_occupancy_per_lane(x0, m, ObstacleArray)
        #print(concatenated_obs_rl)
        
        #t_ = time.time()
        dc, num_dc = self.get_all_driving_corridors(
            x0, v0, concatenated_obs_ml, concatenated_obs_ll, concatenated_obs_rl,  ll_existing, rl_existing, current_lane_id
        )
        
        #print("t1",time.time()-t_)
        self.dc = dc
        self.num_dc = num_dc
        '''instead returning best also second best if existing (if not best twice)'''
        dc_best, dc_2ndbest = self.select_best_driving_corridor(dc, num_dc)

        
        out = (dc_best.dc, dc_best.cost, dc_best.k_lc1, dc_best.k_lc2, dc_best.lane_id, current_lane_id, m, dc_2ndbest.dc, \
            dc_2ndbest.cost, dc_2ndbest.k_lc1, dc_2ndbest.k_lc2, dc_2ndbest.lane_id) # also can output all dcs etc.
        self.reset()    
        return out
    
    def reset(self):
        self.concatenated_obs_ml = -1000*np.ones((self.N*self.obs_num,2))
        self.concatenated_obs_ll = -1000*np.ones((self.N*self.obs_num,2))
        self.concatenated_obs_rl = -1000*np.ones((self.N*self.obs_num,2))
        
        for i in range(self.num_dc):
            self.dc[i].k_lc1 = -1#self.x_center_lc = []
            self.dc[i].k_lc2 = -1#
        
    def calculate_parallel_curve(self, x, y, k, s, d):
        """
        Calculate the length of a parallel curve at a distance `d` from the original curve.
    
        Parameters:
        - x: array of x-coordinates of the curve points.
        - y: array of y-coordinates of the curve points.
        - k: array of curvature values at each point.
        - s: array of arc length values at each point.
        - d: distance from the original curve to the parallel curve.
    
        Returns:
        - L_parallel: length of the parallel curve.
        """
        s_parallel = np.zeros(s.shape[0],)
        # Calculate the length of the parallel curve
        delta_s = np.diff(s)  # Differences in arc length
        avg_k = (k[:-1] + k[1:]) / 2  # Average curvature between points
        # Calculate length increments using the formula L(C_d) = (1 + d * k(s)) * ds
        for i in range(s.shape[0]-1):
            s_parallel[i+1] = s_parallel[i] + np.sum((1 - d * avg_k[i]) * delta_s[i])
            if s_parallel[i+1] <= s_parallel[i]:
                s_parallel[i+1] = s_parallel[i]+0.01
    
        return s_parallel


    def correct_arclength_parallel_lane(self, s, lane_id):
        if lane_id == 1:
            s = self.lut1(s)
        elif lane_id == -1:
            s = self.lut_1(s)
        elif lane_id == 2:
            s = self.lut2(s)
        elif lane_id == -2:
            s = self.lut_2(s)
        return s


    def correct_arclength_2_ref_lane(self, s, lane_id):
        if lane_id == 1:
            s = self.lut1_(s)
        elif lane_id == -1:
            s = self.lut_1_(s)
        elif lane_id == 2:
            s = self.lut2_(s)
        elif lane_id == -2:
            s = self.lut_2_(s)
        return s
    
    
    def compute_curvature(self, psi, s):
        """
        Compute curvature (kappa) from heading angle (psi) and arc length (s).

        Parameters:
            psi (numpy array): Heading angle at each waypoint (radians).
            s (numpy array): Arc length at each waypoint.

        Returns:
            kappa (numpy array): Curvature values at each waypoint.
        """
        # Compute dpsi/ds using finite differences
        dpsi = np.gradient(psi, s)  # Equivalent to dpsi/ds

        return dpsi
 