'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

from scipy.linalg import block_diag
import numpy as np
from copy import deepcopy
import torch

from predictor.utils.ep_utils.basis_function import *
import predictor.utils.ep_utils.preprocess_utils.tracking_utils as tracking_utils 
from predictor.utils.ep_utils.preprocess_utils import Trajectory 
from predictor.utils.ep_utils.preprocess_utils import BernsteinPolynomials
from predictor.utils.ep_utils.preprocess_utils import TrajectoryMotion
from predictor.utils.ep_utils.preprocess_utils import TrajectoryObservation


class Polynomial_Tracker():
    '''
    This is a tracker with modified Kalman Filter. The state variables in the Kalman Filter are polynomial parameters (control points). 
    Details can be found in paper: https://www.uni-das.de/images/pdf/fas-workshop/2022/FAS2022-03-Reichardt.pdf
    '''
    def __init__(self, timescale, degree, space_dim, hist_len):
        self.timescale = timescale
        self.degree = degree
        self.space_dim = space_dim
        self.hist_len = hist_len
        
        self.BASIS = BernsteinPolynomials(self.degree)
        self.TRAJ = Trajectory(basisfunctions=self.BASIS, spacedim=self.space_dim, timescale=self.timescale)
        self.motion_prior = np.eye(self.BASIS.size * self.space_dim) * 200000 # uninformative prior for motion 
        self.Q = np.kron(np.diag([0, .2, .1])**2, np.eye(self.space_dim)) # process noise
        self.MM = TrajectoryMotion(self.TRAJ, self.Q, Prior=self.motion_prior) # motion model
        self.OM = TrajectoryObservation(self.TRAJ, t=self.timescale, derivatives=[[0,1]], R=[block_diag(np.diag([1, 1]), np.diag([1, 1]))]) # observation model, R is dummy
        self.H = self.OM._H(None) # observation matrix
        
        
    def track(self, x, timestep_mask, timestamps, time_window, priors, hist_priors, av_index, agent_index, x_mean, x_cov):
        '''
        x: [x, y, heading, vx, vy] should be in the original coordinate in dataset
        '''
        num_actors = x.shape[0]
        agent_appearance_mask = np.zeros((num_actors), dtype=bool)
        agent_last_appearance_time = deepcopy(time_window[:, 0]) # [A], we start the tracking for at agents' first appearance time (time_window[:, 0])
        for t in range(self.hist_len):
            x_t = deepcopy(x[:, t, :])
            x_mask = timestep_mask[:, t] #[A]
            new_agent_mask = x_mask & (~agent_appearance_mask)
            old_agent_mask = x_mask & (~new_agent_mask)
            
            # initialize new agents
            for i, m in enumerate(new_agent_mask):
                if m:
                    if i == av_index:
                        new_state = tracking_utils.get_initial_state_ego(x_t[i, :2], x_t[i, -2:], hist_priors[i], self.TRAJ, self.BASIS, self.timescale)
                    else:
                        new_state = tracking_utils.get_initial_state_agt(x_t[i, :2], x_t[i, -2:], hist_priors[i], self.TRAJ, self.BASIS, self.timescale)
                    
                    x_mean[i] = new_state.x
                    x_cov[i] = new_state.P
            
            
            delta_ts = np.array([timestamps[t]]) - agent_last_appearance_time
            delta_ts = np.clip(delta_ts, 0., self.timescale)


            self.predict(x_mean=x_mean,
                         x_cov=x_cov,
                         delta_ts=delta_ts,
                         mask_to_predict=agent_appearance_mask if t==self.hist_len-1 else old_agent_mask) # predict for existing agents
            
            self.update(x_t=x_t,
                        priors= priors,
                        current_timestep=t,
                        av_index=av_index,
                        mask_to_update=old_agent_mask,
                        x_mean=x_mean,
                        x_cov=x_cov)
            
            
            agent_appearance_mask = agent_appearance_mask | x_mask
            agent_last_appearance_time[x_mask] = timestamps[t]
        return
    
    
    
    def predict(self, delta_ts, mask_to_predict, x_mean, x_cov):
        t1 = np.linspace(0, self.timescale - delta_ts, self.degree+1) / self.timescale
        t2 = np.linspace(delta_ts, self.timescale, self.degree+1) / self.timescale

        B1 = torch.tensor(np.kron(phi_b(t1, self.degree), np.eye(self.space_dim)))
        B2 = torch.tensor(np.kron(phi_b(t2, self.degree), np.eye(self.space_dim)))

        Fs = torch.linalg.solve(B1.mT @ B1 + self.MM._invPrior, B1.mT @ B2).numpy()[mask_to_predict] # details here: https://www.uni-das.de/images/pdf/fas-workshop/2022/FAS2022-03-Reichardt.pdf
        Qs = Fs @ self.MM._Qmat @ np.transpose(Fs, (0,2,1))
        

        # Predict
        x0 = (np.array([np.kron(
                self.TRAJ.basisfunctions.get(dt / self.timescale),
                np.eye(self.space_dim)
            )  for dt in delta_ts]) @ x_mean[:, :, None])[:,:,0]


        x0 = np.kron(np.ones(self.TRAJ.basisfunctions.size), x0)

        x_mean[mask_to_predict] = (Fs @ ((x_mean - x0)[mask_to_predict, :, None]) + x0[mask_to_predict, :, None]).squeeze(-1)
        x_cov[mask_to_predict] = (Fs @ x_cov[mask_to_predict] @ np.transpose(Fs, (0,2,1))) + Qs
        
        return
    
    def update(self, x_t, priors, current_timestep, av_index, mask_to_update, x_mean, x_cov,):
        if np.any(mask_to_update):
            R = self.build_R(x_t=x_t,
                             priors=priors,
                             current_timestep=current_timestep,
                             av_index=av_index,
                             mask_to_update=mask_to_update)
            
            z = np.concatenate([x_t[:, :2], x_t[:, -2:]], axis=-1)[mask_to_update]
            
            S = self.H @ x_cov[mask_to_update] @ self.H.T + R
            S = (S + np.transpose(S, (0,2,1))) / 2

            K = torch.linalg.lstsq(torch.tensor(S), torch.tensor(self.H @ x_cov[mask_to_update]), rcond=None)[0].mT.numpy() # Kalman Filter

            x_mean[mask_to_update] = x_mean[mask_to_update] + (K @ (z - (self.H@x_mean[mask_to_update, :, None]).squeeze(-1))[:, :, None]).squeeze(-1)

            P_new = (np.eye(x_mean[mask_to_update].shape[1]) - K @ self.H) @ x_cov[mask_to_update]
            P_new = (P_new + np.transpose(P_new, (0,2,1))) / 2

            x_cov[mask_to_update] = P_new
            
        else:
            return 
        
        
    def build_R(self, x_t, priors, current_timestep, av_index, mask_to_update):
        R_pos = tracking_utils.build_obj_observation_noise_covariance_scene(x_t=x_t,
                                                                            av_idx=av_index,
                                                                            degree = self.degree,
                                                                            prior_data_list = priors)
                
        if current_timestep == self.hist_len-1: # activate this if you want the last control points more close to the last measurement.
            R_pos = R_pos/9.

        R_vel = np.repeat((np.diag([3., 3.])**2)[None, :, :], axis=0, repeats=R_pos.shape[0])
        R_vel[av_index] = np.diag([0.5, 0.1])**2 # ego has smaller noise for observed vel

        agent_heading = x_t[:, 2]
        agent_ro_mat = np.zeros((R_pos.shape[0], self.space_dim, self.space_dim))
        agent_ro_mat[:, 0, 0] = np.cos(agent_heading)
        agent_ro_mat[:, 0, 1] = -np.sin(agent_heading)
        agent_ro_mat[:, 1, 0] = np.sin(agent_heading)
        agent_ro_mat[:, 1, 1] = np.cos(agent_heading)

        R_vel = agent_ro_mat @ R_vel @ np.transpose(agent_ro_mat, (0,2,1))
        
        R = np.stack([block_diag(r_pos, r_vel) for r_pos, r_vel in zip(R_pos[mask_to_update], R_vel[mask_to_update])])
        
        return R