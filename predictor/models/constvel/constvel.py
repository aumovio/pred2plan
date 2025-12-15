"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, OneCycleLR

from predictor.models.base_model.base_model import BaseModel

class ConstVel(BaseModel):
    '''
    Const Vel Class.
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
        self.multimodality = config["model"]["multimodality"]

    def forward(self, batch):
        inputs = batch['input_dict']
        #shape inputs['obj_trajs'] num_targets, num_vx, 39 # v1 = inputs['obj_trajs'] [0,inputs["track_index_to_predict"][0], :2]
        agents_in, agents_mask= inputs['obj_trajs'], inputs['obj_trajs_mask']

        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1,
                                                                                                      *agents_in.shape[
                                                                                                       -2:])).squeeze(1)
        
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,
                                                                                                       agents_mask.shape[
                                                                                                           -1])).squeeze(
            1)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)
        #:param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        #breakpoint()
        # last_vel_x = (ego_in[:, -3, 25] + ego_in[:, -2, 25] + ego_in[:, -1, 25])/3
        # last_vel_y = (ego_in[:, -3, 26] + ego_in[:, -2, 26] + ego_in[:, -1, 26])/3
        last_vel_x = (ego_in[:, -1, 25])/1
        last_vel_y = (ego_in[:, -1, 26])/1
 
        base_speed = torch.sqrt(last_vel_x**2 + last_vel_y**2)
        std_x = torch.linspace(0.1, 2, steps=self.N, device=base_speed.device).unsqueeze(0)
        std_y = torch.linspace(0.1, 2, steps=self.N, device=base_speed.device).unsqueeze(0)
        rho = torch.linspace(1.0, 1.0, steps=self.N, device=base_speed.device).unsqueeze(0)

        if not self.multimodality:
            time_steps = torch.linspace(1, ( self.N), steps= self.N, device=last_vel_x.device) * self.dt
            time_steps_x = time_steps * last_vel_x.view(-1, 1) # * self.dt
            time_steps_y = time_steps * last_vel_y.view(-1, 1) # * self.dt
            # breakpoint()
            output = {}
            output['predicted_trajectory'] = torch.zeros((ego_in.shape[0], self.num_modes, self.N, 5), device=last_vel_x.device)
            output['predicted_trajectory'][:, :, :, 0] = time_steps_x.unsqueeze(1).expand(-1,  self.num_modes, -1)
            output['predicted_trajectory'][:, :, :, 1] = time_steps_y.unsqueeze(1).expand(-1,  self.num_modes, -1)
            output['predicted_trajectory'][:, :, :, 2] = torch.log(std_x.unsqueeze(1).expand(-1,  self.num_modes, -1))
            output['predicted_trajectory'][:, :, :, 3] = torch.log(std_y.unsqueeze(1).expand(-1,  self.num_modes, -1))
            output['predicted_trajectory'][:, :, :, 4] = rho.unsqueeze(1).expand(-1,  self.num_modes, -1)

            output['predicted_probability'] =  torch.ones((ego_in.shape[0], self.num_modes), device=last_vel_x.device)/self.num_modes
            output["predicted_log_probability"] = torch.log(torch.ones((ego_in.shape[0], self.num_modes), device=last_vel_x.device)/self.num_modes)                                       
            loss = torch.tensor(0.0, device=base_speed.device, requires_grad=True)
            output['dataset_name'] = inputs['dataset_name']
            return output, loss
        else:
            # compute base speed and heading
            base_heading = torch.atan2(last_vel_y, last_vel_x)
            v_deltas = torch.tensor([1.5,-1.5,0,0,-2.5,0], device=ego_in.device)
            #breakpoint()
            heading_deltas = torch.pi*torch.tensor([0, 0, -4, 4, 0,0])/180
            # time vector
            time_steps = torch.linspace(1, (self.N) , steps=self.N, device=base_speed.device) * self.dt
            # prepare outputs
            B = ego_in.shape[0]
            output = {}
            traj = torch.zeros((B, self.num_modes, self.N, 5), device=base_speed.device)
            # iterate over modes
            for m in range(self.num_modes):
                # mode-specific speed and heading
                mode_speed = base_speed + v_deltas[m]
                mode_heading = base_heading + heading_deltas[m]
                # displacements
                dx = (mode_speed * torch.cos(mode_heading)).view(-1,1) * time_steps.view(1,-1)
                dy = (mode_speed * torch.sin(mode_heading)).view(-1,1) * time_steps.view(1,-1)
                # assign
                traj[:, m, :, 0] = dx
                traj[:, m, :, 1] = dy
                traj[:, m, :, 2] = std_x
                traj[:, m, :, 3] = std_y
                traj[:, m, :, 4] = rho
            
            output['predicted_trajectory'] = traj
            output['predicted_probability'] = torch.ones((B, self.num_modes), device=base_speed.device) / self.num_modes
            output['predicted_log_probability'] = torch.log(output['predicted_probability'])
            output['dataset_name'] = inputs['dataset_name']                                 
            loss = torch.tensor(0.0, requires_grad=True)
            return output, loss
    
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




