"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import numpy as np

import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.tracked_objects_types import STATIC_OBJECT_TYPES, AGENT_TYPES
from nuplan.common.actor_state.tracked_objects import TrackedObjectType

class Obstacle():
    def __init__(self, x, y,psi = 0, l = 6, w = 2.5,  v = 0.0, obj_num = None, obj_type = TrackedObjectType.VEHICLE, S=1, is_dummy = False):
        self.x = x
        self.y = y
        self.psi = psi
        self.l = l
        self.w = w
        self.v = v
        self.x_preds = []
        self.y_preds = []
        self.psi_preds = []
        self.v_preds = []
        self.stdx_preds = []
        self.stdy_preds = []
        self.rho_preds = []
        
        self.x0 = x
        self.y0 = y
        self.psi0 = psi
        self.v0 = v
    
        self.gt = False
        self.pred_available = False
        self.x_true =[]
        self.y_true =[]
        self.psi_true=[]
        self.v_true =[]
        self.k = 0
        self.obj_num = obj_num 
        self.obj_type =  obj_type#"TYPE_VEHICLE"
        self.trajs_w_modes = None
        
        self.x_hist = np.zeros(21,) # TODO: 21 steps comes from Nuplan history buffer minus 1
        self.y_hist = np.zeros(21,) 
        self.psi_hist = np.zeros(21,) 
        self.v_hist = np.zeros(21,)  

        self.S = S
        self.probs = np.ones(S)/S
        self.is_dummy = is_dummy
        self.is_static = True if obj_type in STATIC_OBJECT_TYPES else False
        
    def restart(self ):
        self.x = self.x0
        self.y = self.y0
        self.psi = self.psi0
        self.v = self.v0
        
    def groundtruth_traj(self, x, y, psi=None, v=None):
        self.x_true = x
        self.y_true = y
        self.psi_true = psi
        self.v_true = v
        self.gt = True
        
    def get_prediction(self, x, y, psi, N, v=None, stdx=None, stdy=None, probs=None, rho=None):
        """this is used if predictor is available"""
        padding_mode = "edge" # "constant"
        self.x_preds, self.y_preds, self.psi_preds,self.v_preds,self.stdx_preds,self.stdy_preds, self.rho_preds = [],[],[],[],[],[],[]
        if probs is None:
            probs = np.ones(self.S)/self.S
        self.probs = probs
        for s in range(self.S):
            self.x_preds.append(np.pad(x[s][:N+1], (0, max(N+1 -x[s].shape[0], 0)), mode=padding_mode)) #
            #self.x_pred =self.x_true[self.k:self.k+N+1] #
            self.y_preds.append(np.pad(y[s][:N+1], (0, max(N+1 -y[s].shape[0], 0)), mode=padding_mode)) #  
            self.psi_preds.append(np.pad(psi[s][:N+1], (0, max(N+1 -psi[s].shape[0], 0)), mode=padding_mode))
            #self.y_pred = self.y_true[self.k:self.k+N+1] #
            if v is not None and v!=[]:
                self.v_preds.append(np.pad(v[s][:N+1], (0, max(N+1 -v[s].shape[0], 0)), mode=padding_mode))# 
                self.stdx_preds.append(np.pad(stdx[s][:N+1], (0, max(N+1 -stdx[s].shape[0], 0)), mode=padding_mode))
                self.stdy_preds.append(np.pad(stdy[s][:N+1], (0, max(N+1 -stdy[s].shape[0], 0)), mode=padding_mode))
            if rho is not None and rho!=[]:   
                self.rho_preds.append(np.pad(rho[s][:N+1], (0, max(N+1 -rho[s].shape[0], 0)), mode=padding_mode))
        self.pred_available = True
        self.k = self.k +1
        
    def prediction(self, N, Ts):
        is_stationary = all(abs(v) < 0.3 for v in self.v_hist)
        if 0:#self.gt:
            self.x_pred = self.x_true[self.k:self.k+N+1]
            self.y_pred = self.y_true[self.k:self.k+N+1]
            self.psi_pred =  self.psi_true[self.k:self.k+N+1]
        if self.pred_available:
            self.x_preds = self.x_preds
            self.y_preds = self.y_preds
            self.psi_preds =  self.psi_preds
        elif is_stationary:
            self.x_preds = []
            self.y_preds = []
            self.psi_preds = []
            self.v_preds = []
            self.stdx_preds = []
            self.stdy_preds = []
            self.rho_preds = []

            # Single stationary scenario; horizon N+1
            x_traj   = np.full(N+1, self.x)
            y_traj   = np.full(N+1, self.y)
            psi_traj = np.full(N+1, self.psi)
            v_traj   = np.full(N+1, self.v)  # or 0.0 if you want "completely stopped"
            stdx     = np.zeros(N+1)
            stdy     = np.zeros(N+1)
            rho      = np.zeros(N+1)

            for _ in range(self.S):
                self.x_preds.append(x_traj.copy())
                self.y_preds.append(y_traj.copy())
                self.psi_preds.append(psi_traj.copy())
                self.v_preds.append(v_traj.copy())
                self.stdx_preds.append(stdx.copy())
                self.stdy_preds.append(stdy.copy())
                self.rho_preds.append(rho.copy())

            # Single mode: "stationary"
            self.probs = np.ones(self.S) / self.S
        else: #constant velocity model
            self.x_preds = []
            self.y_preds = []
            self.psi_preds = []
            self.v_preds  = []
            x_local = np.linspace(0,self.v*Ts*(N+1), N+1, endpoint=False )
            y_local = np.linspace(0,0, N+1, endpoint=True )   
            psi_preds = np.linspace(self.psi,self.psi, N+1, endpoint=True )  
            v_preds = np.linspace(self.v,self.v, N+1, endpoint=True )  
            stdx_preds = np.linspace(0,0, N+1, endpoint=True )  
            stdy_preds = np.linspace(0,0, N+1, endpoint=True )  
            rho_preds = np.linspace(0,0, N+1, endpoint=True )  
            x_preds = np.zeros((N+1))
            y_preds = np.zeros((N+1))
            for k in range((N+1)):
                x_preds[k] = self.x + x_local[k] * np.cos(psi_preds[k]) - y_local[k] * np.sin(psi_preds[k])
                y_preds[k] = self.y + x_local[k] * np.sin(psi_preds[k]) + y_local[k] * np.cos(psi_preds[k])
            self.psi_preds.append(psi_preds)
            self.x_preds.append(x_preds)
            self.y_preds.append(y_preds)
            self.v_preds.append(v_preds)
            self.stdx_preds.append(stdx_preds)
            self.stdy_preds.append(stdy_preds)
            self.rho_preds.append(rho_preds)
            if self.S>1:
                self.probs = np.ones(6)/6
                v_delta = [1.5,-1.5,0,0,-2.5]
                psi_delta = [0, 0, np.deg2rad(4), np.deg2rad(-4), 0]
                for m in range(6-1):
                    v_preds = np.linspace(max(self.v+v_delta[m],0),max(self.v+v_delta[m],0), N+1, endpoint=False ) 
                    v_preds[:15] = np.linspace(self.v, max(self.v+v_delta[m],0), 15, endpoint=False ) 
                    psi_preds = np.linspace(self.psi,self.psi+psi_delta[m], N+1, endpoint=False )  
                    x_local =  np.concatenate(([0.0], np.cumsum(v_preds[:-1] * Ts))) #np.linspace(0,self.v*Ts*(N+1), N+1, endpoint=False )
                    x_preds = np.zeros((N+1))
                    y_preds = np.zeros((N+1))
                    for k in range((N+1)):
                        x_preds[k] = self.x + x_local[k] * np.cos(psi_preds[k]) - y_local[k] * np.sin(psi_preds[k])
                        y_preds[k] = self.y + x_local[k] * np.sin(psi_preds[k]) + y_local[k] * np.cos(psi_preds[k])
                    self.psi_preds.append(psi_preds)
                    self.x_preds.append(x_preds)
                    self.y_preds.append(y_preds)
                    self.v_preds.append(v_preds)
                    self.stdx_preds.append(stdx_preds)
                    self.stdy_preds.append(stdy_preds)
                    self.rho_preds.append(rho_preds)
                
        return self.x_preds, self.y_preds, self.psi_preds, self.probs 
    
    def update_states(self,N, Ts,  psi0=None, v=None, k=0):
        self.k = k
        if self.gt:
            self.x = self.x_true[k]
            self.y = self.y_true[k]
            self.psi = self.psi_true[k]
            self.v = self.v_true[k]
        else:
            if psi0 is not None:
                self.psi = psi0
            if v is not None:
                self.v = v
            self.x = self.x + self.v * Ts *np.cos(self.psi)
            self.y = self.y + self.v * Ts *np.sin(self.psi)

        return self.x, self.y,  self.psi, self.v
    
    def populate_history_from_buffer(self, history: list[DetectionsTracks]):
        token = self.obj_num
        max_fill = min(len(self.x_hist), max(0, len(history) - 1))
        for t in range(1, max_fill+1):
            obj = history[t].get(token)
            if obj is None:
                continue
            self.x_hist[t-1]   = obj.center.x
            self.y_hist[t-1]   = obj.center.y
            self.psi_hist[t-1] = obj.center.heading
            self.v_hist[t-1]   = obj.velocity.magnitude()

    def populate_future_from_groundtruth(self, ground_truth: list[DetectionsTracks], sentinel=-1000):
        N = len(ground_truth)
        token = self.obj_num
        self.x_true   = np.full(N + 1, sentinel, dtype=float)
        self.y_true   = np.full(N + 1, sentinel, dtype=float)
        self.psi_true = np.full(N + 1, sentinel, dtype=float)
        self.v_true   = np.full(N + 1, sentinel, dtype=float)

        # t=0 current state
        self.x_true[0] = self.x
        self.y_true[0] = self.y
        self.psi_true[0] = self.psi
        self.v_true[0] = self.v

        # future states
        for t in range(0, N):
            obj = ground_truth[t].get(token)
            if obj is None:
                continue
            self.x_true[t+1]   = obj.center.x
            self.y_true[t+1]   = obj.center.y
            self.psi_true[t+1] = obj.center.heading
            self.v_true[t+1]   = obj.velocity.magnitude()
