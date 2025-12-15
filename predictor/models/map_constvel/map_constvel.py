"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import json
from scipy.linalg import block_diag
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, OneCycleLR
from torch_geometric.data import Batch, HeteroData

from predictor.models.base_model.base_model import BaseModel
from predictor.models.map_constvel.map_utils import convert_map, associate_vehicle_to_lanes, search_end_states, associate_vehicle_to_lanes_incrementally

from predictor.utils.ep_utils.preprocess_utils import tracking_utils
from predictor.utils.ep_utils.preprocess_utils.tracker import Polynomial_Tracker


from predictor.utils.ep_utils.preprocess_utils.cmotap.trajectory import Trajectory 
from predictor.utils.ep_utils.preprocess_utils.cmotap.basisfunctions.bernsteinpolynomials import BernsteinPolynomials
from predictor.utils.ep_utils.preprocess_utils.cmotap.statedensities.gaussian_control_point_density import GaussianControlPointDensity
from predictor.utils.ep_utils.preprocess_utils.cmotap.motionmodels.trajectory_motion import TrajectoryMotion
from predictor.utils.ep_utils.preprocess_utils.cmotap.observationmodels.trajectory_observation import TrajectoryObservation

import matplotlib.pyplot as plt

class MapConstVel(BaseModel):
    '''
    Map Const Vel Class.
    '''

    def __init__(self, config):
        super().__init__(config)
        self.num_modes=6
        self.N = 60
        self.dt = 0.1
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(10, 100),
            torch.nn.Linear(100,50),
            torch.nn.Linear(50, 1)
        )
        self.learning_rate = 5
        self.past_len = config["data"]["past_len"]
        #self.fut_len = config["data"]["future_len"]
        self.multimodality = config["model"]["multimodality"]
        self.only_target = config["model"]["only_target"]
        
        self.hist_timescale = (self.past_len-1)/10
        self.fut_timescale = (self.N)/10 # include current timestep
        
        self.track_deg = 2 # TODO: make this configurable
        self.pred_deg = 4 # TODO: make this configurable
        
        self.hist_basis = BernsteinPolynomials(self.track_deg)
        self.fut_basis = BernsteinPolynomials(self.pred_deg)
        self.tracker = Polynomial_Tracker(timescale = self.hist_timescale, 
                                          degree = self.track_deg,
                                          space_dim = 2, 
                                          hist_len = self.past_len)
        self.load_priors()
        
        # super(MapConstVel, self).__init__()

    def _metric_kwargs(self, batch, out):
        # Map your batch/out to the metric input names you implemented
        targets = batch["agent"]["train_mask"].nonzero(as_tuple=True)[0]

        # convert groundtruth trajectories to local frames
        agt_origin, agt_theta = batch['agent']['position'][targets, self.past_len-1, :2], batch['agent']['heading'][targets, self.past_len-1]
        cos, sin = agt_theta.cos(), agt_theta.sin()
        rot_mat = agt_theta.new_zeros(agt_origin.shape[0],2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        gt_traj = (batch["agent"]["position"][targets][:,self.config.data.past_len:-20,:2] - agt_origin.unsqueeze(1))@rot_mat
        gt_mask = batch["agent"]["valid_mask"][targets][:,self.config.data.past_len:-20]
        gt_curr_vel = batch["agent"]["velocity"][targets][:, self.config.data.past_len-1]

        out["predicted_trajectory"][...,4] = torch.clip(out["predicted_trajectory"][..., 4], -0.5, 0.5)

        return dict(
            gt_traj=gt_traj,       # [B,T,D]
            gt_mask=gt_mask,       # [B,T]
            pred_traj=out["predicted_trajectory"],     # [B,M,T,D]
            pred_prob=out["predicted_probability"],     # [B,M]
            trajs=out["predicted_trajectory"], # [B,M,T,D]
            probs=out["predicted_probability"],
            mask=torch.ones(gt_traj.shape[0]).bool(),
            curr_vel = gt_curr_vel.norm(dim=-1),
            heading_idx = 5
            # optionally:
            # gt_position_idx=slice(0,2),
            # pred_param_idx=slice(0,5),
            # curr_vel=batch.get("curr_vel"),
        )

    def forward(self, input_batch, output_in_batch=False):
        output = []
        
        if isinstance(input_batch, Batch):
            #input_batch = input_batch.to(deivce='cpu')
            data_list = input_batch.to_data_list()
            output = [self.predict_scene(data) for data in data_list]
        else:
            output = [self.predict_scene(input_batch)]
        loss = torch.tensor(0.0, requires_grad=True) # a dummy loss, required by Unitraj Framework
        
        if output_in_batch: # output torch geometric Batch format
            output_batch = Batch.from_data_list(output)
            for k, v in input_batch.items():
                if k not in ['agent', 'map']:
                    output_batch[k] = v
            return output_batch, loss
        else: # output tensor
            return {'predicted_trajectory': torch.concat([output_data['prediction']['predicted_trajectory'] for output_data in output], axis=0),
                    'predicted_probability': torch.concat([output_data['prediction']['predicted_probability'] for output_data in output], axis=0),
                    'predicted_log_probability': torch.concat([output_data['prediction']['predicted_log_probability'] for output_data in output], axis=0),}, loss
                    
    
    def predict_scene(self, scene):
        device = scene['agent']['position'].device
        if self.only_target:
            target_mask = scene['agent']['train_mask']
        else:
            target_mask = scene['agent']['train_mask'].new_ones((scene['agent']['train_mask'].shape[0]), dtype=bool)
        
        A, T, K = scene['agent']['position'][target_mask].shape[0], scene['agent']['position'][target_mask].shape[1], self.num_modes

        trajectories = torch.zeros((A, self.num_modes, self.N, 5), device=device)
        probabilities = torch.zeros((A, self.num_modes), device=device)
        
        
        converted_map = convert_map(scene['map'], 
                                    update_adjacent=True) # update the annotated adjacent lanes based on geometry, set to True for nuplan
        
        
        ### track agent history ###
        x_origin = torch.stack([scene['agent']['position'][target_mask, :, 0], # x
                                scene['agent']['position'][target_mask, :, 1], # y
                                scene['agent']['heading'][target_mask, :], # heading
                                scene['agent']['velocity'][target_mask, :, 0], # vx
                                scene['agent']['velocity'][target_mask, :, 1],], dim=-1).cpu() # vy  # [A, T, 5]
        x_mask = scene['agent']['valid_mask'][target_mask].cpu() # [A, T]
        x_mean = np.zeros((A, (self.track_deg+1) * 2), dtype=float) 
        x_cov = np.zeros((A, (self.track_deg+1) * 2, (self.track_deg+1) * 2), dtype=float)
        priors, history_priors = [None] * A, np.zeros((A, (self.track_deg+1)*2, (self.track_deg+1)*2))
        timestamps = np.linspace(0, T, T+1) * self.dt 
        time_window = np.zeros((A, 2), dtype=float)
        av_index = torch.where(scene['agent']['role'][target_mask][:, 0])[0][0].cpu()
        
        for i_a in range(A): # loop through all targets
            hist_mask = x_mask[i_a, :self.past_len]
            hist_steps = np.where(hist_mask[:self.past_len]==True)[0]
            T_hist = timestamps[hist_steps]
            time_window[i_a] = np.array([np.min(T_hist), np.max(T_hist)])
            agent_type = scene['agent']['type'][target_mask][i_a]
            
            if i_a == av_index: # is AV
                priors[i_a] = self.prior_ego
                history_priors[i_a] = self.hist_prior_ego
            else:
                prior_data, hist_prior_data = None, None
                if agent_type == 1: # vehicle
                    prior_data, hist_prior_data = self.prior_vehicle, self.hist_prior_vehicle
                elif agent_type == 2: # pedestrian
                    prior_data, hist_prior_data = self.prior_pedestrian, self.hist_prior_pedestrian
                elif agent_type == 3: # cyclist
                    prior_data, hist_prior_data = self.prior_cyclist, self.hist_prior_cyclist
                else: # TODO: what is the prior for unknown objects?
                    prior_data, hist_prior_data = self.prior_vehicle, self.hist_prior_vehicle

                priors[i_a] = prior_data
                history_priors[i_a] = hist_prior_data
  
        
        self.tracker.track(x=x_origin.cpu().numpy(), 
                          timestep_mask=x_mask.cpu().numpy(), 
                          timestamps=timestamps, 
                          time_window=time_window, 
                          priors=priors, 
                          hist_priors=history_priors, 
                          av_index=av_index, 
                          agent_index = None,
                          x_mean=x_mean, 
                          x_cov=x_cov)
        
        for i_a in range(A):            
            if scene['agent']['valid_mask'][target_mask][i_a, self.past_len-1]: # check if last observation is valid
                agent_xy, agent_heading = scene['agent']['position'][target_mask][i_a, self.past_len-1, :2].contiguous().cpu(), scene['agent']['heading'][target_mask][i_a, self.past_len-1].contiguous().cpu()
                agent_type = scene['agent']['type'][target_mask][i_a].contiguous().cpu()                
                
                # the agent_data might include future observations
                agent_data = torch.stack([scene['agent']['position'][target_mask][i_a, :, 0], # x
                                          scene['agent']['position'][target_mask][i_a, :, 1], # y
                                          scene['agent']['heading'][target_mask][i_a, :], # heading
                                          scene['agent']['velocity'][target_mask][i_a, :, 0], # vx
                                          scene['agent']['velocity'][target_mask][i_a, :, 1],], dim=-1) # vy  
                
                association_results = associate_vehicle_to_lanes_incrementally(vehicle_xy = agent_xy,
                                                                               vehicle_heading = agent_heading,
                                                                               map_data = converted_map)
                end_states = self.get_end_states(agent_data.cpu(), 
                                                 converted_map,
                                                 association_results)

                if len(end_states) > 0 and agent_type==1: # agent can be associated to at least one lane & agent is vehicle type => use map based CV model
                    predicted_trajectory, predicted_probability = self.predict_agent_with_map_constvel(agent_data.cpu(), 
                                                                                                       x_mean[i_a],
                                                                                                       x_cov[i_a],
                                                                                                       end_states, 
                                                                                                       converted_map)
                    trajectories[i_a] = predicted_trajectory
                    probabilities[i_a] = predicted_probability
                else: # use standard CV model
                    predicted_trajectory, predicted_probability = self.predict_agent_with_constvel(agent_data.cpu())
                    trajectories[i_a] = predicted_trajectory
                    probabilities[i_a] = predicted_probability
            else:
                continue # if last observation is invalid, the predictions/probabilities will be zeros
        
        return HeteroData({'prediction':{'predicted_trajectory': trajectories, # [A, K, T, 5]
                                         'predicted_probability': probabilities, # [A, K]
                                         'predicted_log_probability': torch.log(probabilities), # [A, K]
                                         'num_nodes': scene['agent']['num_nodes']}})
    
    
    def predict_agent_with_map_constvel(self, 
                                        agent_data, 
                                        hist_state_mean, 
                                        hist_state_cov,
                                        end_states, 
                                        map_data):
        hypotheses_list = []
        predictions_list = []
        
        p_0 = np.array([agent_data[self.past_len-1, 0], agent_data[self.past_len-1, 1]], dtype = np.float32)
        v_0 = np.array([agent_data[self.past_len-1, -2], agent_data[self.past_len-1, -1]], dtype = np.float32)
        v_norm = np.linalg.norm(v_0)
        travel_dist = v_norm*self.fut_timescale
        
        t_pred = np.linspace(0, self.fut_timescale, self.N+1)
        predicted_trajectory = torch.zeros((self.num_modes, self.N, 5), device=agent_data.device)
        predicted_probability =  torch.zeros((self.num_modes), device=agent_data.device)
        
        TRAJ_PRED = Trajectory(basisfunctions=self.fut_basis, spacedim=2, timescale=self.fut_timescale)
        TRAJ_HIST = Trajectory(basisfunctions=self.hist_basis, spacedim=2, timescale=self.hist_timescale)
        
        OM_0 = TrajectoryObservation(TRAJ_HIST, t=self.hist_timescale, derivatives=[0, 1, 2], R=np.eye(6))

        x_0 = OM_0.h(hist_state_mean) # estimated states at current timestep
        R_0 = OM_0.H() @ hist_state_cov @ OM_0.H().T
        
        for end_state in end_states:
            if end_state['distance'] < np.max((travel_dist - 5), 0): # reach map boundary, ignore
                continue
            
            pos_T, phi_T, kappa_T = np.array(end_state['xy']), end_state['heading'], end_state['curvature']
            elon_T = np.array([ np.cos(phi_T), np.sin(phi_T)])
            elat_T = np.array([-np.sin(phi_T), np.cos(phi_T)])

            v_T = elon_T * v_norm
            a_T = elat_T * np.abs(kappa_T) * v_norm**2 
            Rot = np.row_stack([elon_T, elat_T])

            Rx = Rot.T @ np.diag([  5, 1])**2 @ Rot
            Rv = Rot.T @ np.diag([  2, 1])**2 @ Rot
            Ra = Rot.T @ np.diag([1, 0.5])**2 @ Rot
            
            R_T= block_diag(Rx, Rv, Ra)
            
            OM_T = TrajectoryObservation(TRAJ_PRED, t=np.array([0, self.fut_timescale]), derivatives=[[0, 1, 2], [0, 1, 2]], R=[R_0, R_T])
            
            hyp = GaussianControlPointDensity(
                    x=np.kron(np.ones(self.fut_basis.size), p_0),
                    P=np.eye(self.fut_basis.size * 2) * 10000
                    ).update(np.block([x_0, pos_T, v_T, a_T]), OM_T)
            
            hypotheses_list.append(
                    hyp
                )
            
            prediction, prediction_cov = TRAJ_PRED.estimate(t = t_pred[1:], density = hyp)
            prediction = prediction.reshape(self.N, 2) # [60,2]
            diag_blocks = prediction_cov.reshape(self.N, 2, self.N, 2)[np.arange(self.N), :, np.arange(self.N), :] # [60,2,2]
            prediction = np.concatenate([prediction, 
                                         diag_blocks[:, [0], [0]], 
                                         diag_blocks[:, [1], [1]], 
                                         diag_blocks[:, [0], [1]]], axis=-1)

            predictions_list.append(prediction)
        
        
        ### calculate probabilities ###
        log_prob = np.log(np.ones(len(hypotheses_list)) / len(hypotheses_list))
        for hyp_index, hyp in enumerate(hypotheses_list):
            delta_T = 0.5
            OM_0_delta = TrajectoryObservation(TRAJ_HIST, t=self.hist_timescale - delta_T, derivatives=[0, 1, 2], R=np.eye(6))
            x_0_delta = OM_0_delta.h(hist_state_mean)
            R_0_delta = OM_0_delta.H() @ hist_state_cov @ OM_0_delta.H().T
            
            OM_T_delta = TrajectoryObservation(TRAJ_PRED, t=self.fut_timescale - delta_T, derivatives=[0, 1, 2], R=np.eye(6))
            x_T_delta = OM_T_delta.h(hyp.x)
            R_T_delta = OM_T_delta.H() @ hyp.P @ OM_T_delta.H().T

            OM_0T = TrajectoryObservation(TRAJ_PRED, t=np.array([0, self.fut_timescale]), derivatives=[[0, 1, 2], [0, 1, 2]], R=[R_0_delta, R_T_delta])

            transition = GaussianControlPointDensity(
                x=np.kron(np.ones(self.fut_basis.size), p_0),
                P=np.eye(self.fut_basis.size * 2) * 10000
            ).update(np.block([x_0_delta, x_T_delta]), OM_0T)


            OM = TrajectoryObservation(TRAJ_PRED, t=delta_T, derivatives=[0, 1, 2], R=R_0)

            log_prob[hyp_index] += transition.observationLogLikelihood(x_0, OM)


        log_prob = log_prob - np.max(log_prob)
        probs = np.exp(log_prob) / np.sum(np.exp(log_prob))
        predictions = np.array(predictions_list)
        
        num_predictions = len(predictions_list)
        considered_num_predictions = num_predictions if num_predictions < self.num_modes else self.num_modes # dummy implementation
        idx = np.argpartition(probs, -considered_num_predictions)[-considered_num_predictions:]          # unsorted top considered_num_predictions
        idx = idx[np.argsort(probs[idx])[::-1]]        # sort by prob desc
        
        top_preds = predictions[idx] 
        top_probs = probs[idx]
        top_probs = top_probs / top_probs.sum()
        
        predicted_trajectory[:considered_num_predictions, :, :] = torch.Tensor(top_preds, device=agent_data.device)
        predicted_probability[:considered_num_predictions] = torch.Tensor(top_probs, device=agent_data.device)
        return predicted_trajectory, predicted_probability
    
    def predict_agent_with_constvel(self, agent_data):
        last_x = agent_data[self.past_len-1, 0]
        last_y = agent_data[self.past_len-1, 1]
        last_vel_x = agent_data[self.past_len-1, -2]
        last_vel_y = agent_data[self.past_len-1, -1]

        std_x = torch.linspace(0.1, 2, steps=self.N, device=last_vel_x.device)
        std_y = torch.linspace(0.1, 2, steps=self.N, device=last_vel_x.device)
        rho = torch.linspace(0.0, 0.0, steps=self.N, device=last_vel_x.device)
        
        time_stamps = torch.linspace(1, ( self.N), steps= self.N, device=last_vel_x.device) * self.dt
        predicted_probability =  torch.ones((self.num_modes))/self.num_modes # euqal probability for all modes
        if not self.multimodality:
            predicted_trajectory = torch.zeros((self.num_modes, self.N, 5), device=last_vel_x.device)
            predicted_trajectory[:, :, 0] =  last_x + last_vel_x * time_stamps
            predicted_trajectory[:, :, 1] =  last_y + last_vel_y * time_stamps
            predicted_trajectory[:, :, 2] =  std_x[None, None, :]
            predicted_trajectory[:, :, 3] =  std_y[None, None, :]
            predicted_trajectory[:, :, 4] =  rho[None, None, :]                      
        else:
            # compute base speed and heading
            base_speed = torch.sqrt(last_vel_x**2 + last_vel_y**2)
            base_heading = torch.atan2(last_vel_y, last_vel_x)
            v_deltas = torch.tensor([1.5,-1.5,0,0,-2.5,0], device=last_vel_x.device)
            heading_deltas = torch.pi*torch.tensor([0, 0, 4, -4, 0,0])/180

            # prepare outputs
            predicted_trajectory = torch.zeros((self.num_modes, self.N, 5), device=last_vel_x.device)
            # iterate over modes
            for m in range(self.num_modes):
                # mode-specific speed and heading
                mode_speed = base_speed + v_deltas[m]
                mode_heading = base_heading + heading_deltas[m]
                # displacements
                dx = (mode_speed * torch.cos(mode_heading)).view(-1,1) * time_stamps.view(1,-1)
                dy = (mode_speed * torch.sin(mode_heading)).view(-1,1) * time_stamps.view(1,-1)
                # assign
                predicted_trajectory[m, :, 0] = last_x+dx
                predicted_trajectory[m, :, 1] = last_y+dy
                predicted_trajectory[m, :, 2] = std_x
                predicted_trajectory[m, :, 3] = std_y
                predicted_trajectory[m, :, 4] = rho

        return predicted_trajectory, predicted_probability
    
    
    def get_end_states(self, 
                       agent_data,
                       map_data,
                       association_results):
        if len(association_results) == 0:
            return []
        
        v_0 = np.array([agent_data[self.past_len-1, -2], agent_data[self.past_len-1, -1]], dtype = np.float32)
        v_norm = np.linalg.norm(v_0)
        travel_dist = v_norm*self.fut_timescale
        
        end_states = search_end_states(association_results,
                                       map_data=map_data,
                                       max_distance=travel_dist,
                                       include_start_neighbour=True if travel_dist > 5 else False, # too less travel distance, exclude lane change possibilities
                                       include_end_neighbour=True if travel_dist > 5 else False) # too less travel distance, exclude lane change possibilities
        
        end_states = [end_state for end_state in end_states if end_state['distance'] >= np.max((travel_dist - 5), 0)]
        
        return end_states
    
    
    def predict_step(self, batch, batch_idx):
        # used in simulation
        prediction = self.forward(batch)[0]
        target_mask =  batch['agent']['train_mask']
        gt = batch['agent']['position'][target_mask, self.past_len:self.past_len+self.N, :2]
        gt_mask = batch['agent']['valid_mask'][target_mask, self.past_len:self.past_len+self.N]
        
        
        predicted_trajectory = prediction['predicted_trajectory'].cpu().numpy()
        predicted_probability = prediction['predicted_probability'].cpu().numpy()
        
        agt_origin, agt_theta = batch['agent']['position'][target_mask, self.past_len-1, :2], batch['agent']['heading'][target_mask, self.past_len-1]
        cos, sin = agt_theta.cos(), agt_theta.sin()

        rot_mat = agt_theta.new_zeros(agt_origin.shape[0],2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        
        
        for k,v in prediction.items():
            if not self.only_target:
                prediction[k] = v[target_mask]
                
            if k == 'predicted_trajectory':
                # transform to agent individual frame
                prediction[k][:,:,:, :2] =  (prediction[k][:,:,:, :2]-agt_origin.unsqueeze(1).unsqueeze(2))@rot_mat.unsqueeze(1)
 

  
        return prediction
        
    def log_info(self, batch, batch_idx, prediction, status='train'):
        ## logging
        inputs = {'center_gt_trajs': None,
                  'center_gt_trajs_mask': None,
                  'center_gt_final_valid_idx': None}
        target_mask =  batch['agent']['train_mask']
        # sdc_mask = batch['agent']['role'][:, 0]
        # target_mask = torch.logical_and(target_mask, ~sdc_mask)
        gt = batch['agent']['position'][target_mask, self.past_len:self.past_len+60, :2]
        gt_mask = batch['agent']['valid_mask'][target_mask, self.past_len:self.past_len+60]
        inputs['center_gt_trajs'] = gt
        inputs['center_gt_trajs_mask'] = gt_mask
        
        ### computing final valid idx ###
        idx = torch.arange(1, gt_mask.shape[1] + 1).expand(gt_mask.shape[0], -1).to(gt_mask.device)
        gt_mask_int = gt_mask*idx
        gt_final_valid_idx = torch.argmax(gt_mask_int, dim=1)

        
        inputs['center_gt_final_valid_idx'] = gt_final_valid_idx
        
        for k,v in prediction.items():
            if not self.only_target:
                prediction[k] = v[target_mask]
        
        loss_dict = self.compute_custom_metrics(inputs, prediction)
        
        size_dict = {key: value.numel() for key, value in loss_dict.items()}
        loss_dict = {key: value.mean().item() for key, value in loss_dict.items()}

        for k, v in loss_dict.items():
            self.log(status + "_" + k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=size_dict[k])
            
        return
    
    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, eps=self.epsilon)

        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0002, steps_per_epoch=1, epochs=self.config.training["max_epochs"],
        #                                                 pct_start=0.02, div_factor=100.0, final_div_factor=10)
        optimizer = AdamW(
            self.parameters(), 
            eps=self.config.training["epsilon"],
            weight_decay=self.config.training["weight_decay"]
        )
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=self.learning_rate, 
            total_steps=self.trainer.estimated_stepping_batches,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1,
            }
        }
    
    
    def load_priors(self):
        dataset = 'argoverse2' # since no prior for nuplan available, we use argoverse 2 priors
        current_file = Path(__file__).resolve().parents[0].as_posix()
        with open(current_file + '/priors/' + dataset + '/vehicle/vehicle_1s.json', "r") as read_file:
            self.prior_vehicle = json.load(read_file)
            self.hist_prior_vehicle = tracking_utils.get_bernstein_prior(degree=self.track_deg, timescale=self.hist_timescale, prior_data=self.prior_vehicle)[1]
            
        with open(current_file + '/priors/' + dataset + '/cyclist/cyclist_1s.json', "r") as read_file:
            self.prior_cyclist = json.load(read_file)
            self.hist_prior_cyclist = tracking_utils.get_bernstein_prior(degree=self.track_deg, timescale=self.hist_timescale, prior_data=self.prior_cyclist)[1]

        with open(current_file + '/priors/' + dataset + '/pedestrian/pedestrian_1s.json', "r") as read_file:
            self.prior_pedestrian = json.load(read_file)
            self.hist_prior_pedestrian = tracking_utils.get_bernstein_prior(degree=self.track_deg, timescale=self.hist_timescale, prior_data=self.prior_pedestrian)[1]

        with open(current_file + '/priors/' + dataset + '/ego/ego_1s.json', "r") as read_file:
            self.prior_ego = json.load(read_file)
            self.hist_prior_ego = tracking_utils.get_bernstein_prior(degree=self.track_deg, timescale=self.hist_timescale, prior_data=self.prior_ego)[1]