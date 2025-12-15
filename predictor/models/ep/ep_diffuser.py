"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import numpy as np
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from predictor.models.ep.metrics import minADE
from predictor.models.ep.metrics import minFDE
from predictor.models.ep.modules import EPEncoder
from predictor.models.ep.modules import EPDenoiser
from predictor.utils.ep_utils.optim import WarmupCosineLR
from predictor.utils.ep_utils.basis_function import basis_function_b, basis_function_m, transform_m_to_b, transform_b_to_m
from predictor.utils.ep_utils.geometry import get_angle_from_2d_rotation_matrix

from predictor.utils.utils.wosac_utils import get_scenario_rollouts, make_id_mapping
from predictor.models.smart.metrics import (
    WOSACMetrics,
    DiversityMetric,
)



class EPDiffuser(pl.LightningModule):
    def __init__(self, config)  -> None:
                 # pred_deg: int,
                 # hidden_dim: int,
                 # num_future_steps: int,
                 # num_encoder_layers: int,
                 # num_denoiser_layers: int,
                 # num_freq_bands: int,
                 # num_heads: int,
                 # head_dim: int,
                 # space_dim: int,
                 # dropout: float,
                 # pl2pl_radius: float,
                 # pl2a_radius: float,
                 # a2a_radius: float,
                 # lr: float,
                 # weight_decay: float,
                 # T_max: int,
                 # homogenizing: bool,
                 # **kwargs)
        super(EPDiffuser, self).__init__()
        self.save_hyperparameters()
        model_config = config["model"]

        self.wosac_data_path = model_config.wosac_data_path
        
        self.hidden_dim = model_config.hidden_dim
        self.space_dim = model_config.space_dim
        self.num_future_steps = model_config.num_future_steps
        self.num_encoder_layers = model_config.num_encoder_layers
        self.num_denoiser_layers = model_config.num_denoiser_layers
        self.num_heads = model_config.num_heads
        self.head_dim = model_config.head_dim
        self.num_freq_bands = model_config.num_freq_bands
        self.dropout = model_config.dropout
        self.pl2pl_radius = model_config.pl2pl_radius
        self.pl2a_radius = model_config.pl2a_radius
        self.a2a_radius = model_config.a2a_radius
        self.lr = model_config.lr
        self.weight_decay = model_config.weight_decay
        self.T_max = model_config.lr_total_steps
        self.T_warm_up = model_config.lr_warmup_steps
        self.pred_deg = model_config.pred_deg
        
        self.tau_pred = np.linspace(0.0, 1.0, self.num_future_steps+1)
        self.Phi_B_pred = torch.Tensor(basis_function_b(tau=self.tau_pred,
                                                        n=self.pred_deg,
                                                        delta_t = self.num_future_steps/10)) # note this include the current timestep
        
        self.encoder = EPEncoder(
                                hidden_dim=self.hidden_dim,
                                pl2pl_radius=self.pl2pl_radius,
                                pl2a_radius=self.pl2a_radius,
                                a2a_radius=self.a2a_radius,
                                num_layers=self.num_encoder_layers,
                                num_heads=self.num_heads,
                                head_dim=self.head_dim,
                                dropout=self.dropout,
                                num_freq_bands = self.num_freq_bands,
                                #homogenizing=homogenizing,
                                )
        
        self.denoiser = EPDenoiser(timesteps=1000,
                                   hidden_dim=self.hidden_dim,
                                   num_layers=self.num_denoiser_layers,
                                   num_heads=self.num_heads,
                                   head_dim=self.head_dim,
                                   dropout=self.dropout,
                                   pred_deg= self.pred_deg,
                                   space_dim = self.space_dim,
                                   num_freq_bands = self.num_freq_bands,) 
        

        self.minADE_1 = minADE(max_guesses=1)
        self.minFDE_1 = minFDE(max_guesses=1)
        
        self.diversity_metrics = DiversityMetric(measure="heading", threshold=0, inflection=0.5, pooling="max")
        self.wosac_metrics = WOSACMetrics("val_closed")
        
    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.denoiser(data=data,
                             scene_enc=scene_enc)

        return pred
                             
                             
    def sample(self, 
               data: HeteroData,
               num_samples: int,
               num_denoising_steps=10,
               method: str = 'ddim',
               post_process = True):
        
        device = data['agent']['x'].device
        scene_enc =self.encoder(data)
        denoised_cp_samples = self.denoiser.sample(data, 
                                                   scene_enc,
                                                   num_samples=num_samples,
                                                   num_denoising_steps=num_denoising_steps,
                                                   method = method) # [num_samples, A, (pred_deg+1)*spacedim]
                  
        Phi_B_pred_kron = torch.kron(self.Phi_B_pred[1:].contiguous().to(self.device), torch.eye(self.space_dim, device=self.device))   
        traj_samples = (denoised_cp_samples @ Phi_B_pred_kron.mT).view(num_samples, data['agent']['num_nodes'], -1, self.space_dim)
        
        
        if post_process:
            agt_origin = data['agent']['agent_origin']
            agt_R = data['agent']['agent_R']
            agt_theta = get_angle_from_2d_rotation_matrix(agt_R)
            gl_origin = data['global_origin'][0]
            gl_R = data['global_R'][0]
            
            gl_origin = torch.concat([torch.tensor(data['global_origin'][i_b], device=device).unsqueeze(dim=0).repeat_interleave(repeats=data['agent']['ptr'][i_b+1] - data['agent']['ptr'][i_b], dim=0) 
                                      for i_b in range(data.batch_size)], dim=0)
            gl_R = torch.concat([torch.tensor(data['global_R'][i_b], device=device).unsqueeze(dim=0).repeat_interleave(repeats=data['agent']['ptr'][i_b+1] - data['agent']['ptr'][i_b], dim=0) 
                                      for i_b in range(data.batch_size)], dim=0)
            
            gl_theta = get_angle_from_2d_rotation_matrix(gl_R)    

            phi_b_p = torch.Tensor(basis_function_b(tau=self.tau_pred[1:],
                                                    n=self.pred_deg,
                                                    k=1,
                                                    return_kron=True,
                                                    delta_t = self.num_future_steps/10)).to(device=device)
            
            pred_v_vec= (denoised_cp_samples@phi_b_p.mT).reshape(denoised_cp_samples.shape[0], denoised_cp_samples.shape[1], -1, 2) # [K, A, T, 2]
            pred_v_norm= torch.norm(pred_v_vec, p=2, dim=-1) # [K, A, T]
            pred_heading = torch.atan2(pred_v_vec[..., 1], pred_v_vec[..., 0]) # [K, A, T]

            dist_agent = torch.norm(denoised_cp_samples[:, :, :2] - denoised_cp_samples[:, :, -2:], p=2, dim=-1) # [A], distance between start and end position
            mask_stop = torch.where(dist_agent <=1.0)
            traj_samples[mask_stop[0], mask_stop[1]] = traj_samples.new_zeros(traj_samples[mask_stop[0], mask_stop[1]].shape)
        
        
            ### coordinate transformation ###
            
            traj_samples = ((traj_samples@ agt_R[None, :].mT) + agt_origin[None, :, None])@gl_R[None, ...].mT + gl_origin[None, :, None] # coordinate transformation
            agt_gl_heading = traj_samples.new_zeros(traj_samples.shape[0], traj_samples.shape[1], traj_samples.shape[2]+1)
            agt_gl_heading[:, :, 1:] = pred_heading + agt_theta[None, :, None] + gl_theta[None, :, None]
            agt_gl_heading[:, :, 0] = data['agent']['x'][:, -1, 2] + gl_theta[None, :] # last measured heading

            x = traj_samples[..., 0]
            y = traj_samples[..., 1]
            z = data['agent']['x'][:, -1, -1][None, :, None].repeat(x.shape[0], 1, x.shape[2]) # last measured z-value
            heading = agt_gl_heading
        
        
            for t in range(traj_samples.shape[2]):
                mask_t_stop = torch.where(pred_v_norm[:, :, t] < 1.0)
                heading[mask_t_stop[0], mask_t_stop[1], t+1] = heading[mask_t_stop[0], mask_t_stop[1], t]

            heading = heading[:, :, 1:]
            dummy_heading = data['agent']['x'][mask_stop[1], -1, 2]
            heading[mask_stop[0], mask_stop[1], :] = dummy_heading[:, None].repeat(1, self.num_future_steps) + gl_theta[mask_stop[1], None] # assign all stoped agents with last measured heading
            
            
            sim_mask = data['agent']['timestep_x_mask'][:, -1]
            traj_samples = torch.concat([x[:, :, :, None], # num_samples, num_agents, num_timesteps, dimension
                                           y[:, :, :, None], 
                                           z[:, :, :, None], 
                                           heading[:, :, :, None]], axis=-1)
            
        return traj_samples, denoised_cp_samples
    
    
    def training_step(self,
                      data,
                      batch_idx):
        mask = data['agent']['timestep_x_mask'][:, :].contiguous() #[A,50]
        mask_a = torch.logical_and(~torch.any(~data['agent']['timestep_y_mask'], dim=-1), mask[:, -1]) # consider all current observed and future fully observed agents
        reg_mask = data['agent']['timestep_y_mask'][mask_a] #[A, 60]
        
        pred = self(data) #[S,A,60,2]
        loss_dn = F.mse_loss(pred['pred_noise_cum'][0, mask_a], pred['target_noise_cum'][0, mask_a])
        
        pred_x0 = pred['pred_x0'] #[A, 14] 

        Phi_B_pred_kron = torch.kron(self.Phi_B_pred[1:].contiguous().to(self.device), torch.eye(self.space_dim, device=self.device))   
        pred_trajs = (pred_x0 @ Phi_B_pred_kron.mT).view(data['agent']['num_nodes'], -1, self.space_dim)[mask_a] #[A,T,2]
        gt = data['agent']['target'][mask_a, :, :self.space_dim] #[A,T,2]
        
        loss_reg = (torch.norm(gt - pred_trajs, p=2, dim=-1) * reg_mask).sum()
        
        loss_reg = loss_reg / reg_mask.sum().clamp_(min=1)

        self.log('train_dn_loss', loss_dn, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self.log('train_reg_loss', loss_reg, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        loss = loss_dn #+ 0.2*loss_reg
        return loss
    
    @torch.no_grad()
    def validation_step(self,
                        data,
                        batch_idx,
                        eval_len = 80,
                        eval_sim_agent = True):       
        if eval_sim_agent:
            trajs, _ = self.sample(data=data.to('cuda'),
                                    num_samples=32,
                                    num_denoising_steps = 10,
                                    method='ddim',
                                    post_process = True,
                                    )
            self.evaluate_rollouts(data, 
                                   trajs.transpose(0,1))
            
        else:
            gt = data['agent']['target'][:, :eval_len, :self.space_dim]
            reg_mask = data['agent']['timestep_y_mask'][:, :eval_len]

            scene_enc = self.encoder(data)
            pred = self.denoiser(data=data,
                                 scene_enc=scene_enc,
                                 timesteps = self.denoiser.timesteps)

            num_samples, A, _ = pred['pred_x0'].shape        
            eval_mask = data['agent']['track_category_origin'] == 3

            gt_eval = gt[eval_mask]
            Phi_B_pred_kron = torch.kron(self.Phi_B_pred[1:].contiguous().to(self.device), torch.eye(self.space_dim, device=self.device))   
            pred_trajs = (pred['pred_x0'] @ Phi_B_pred_kron.mT).view(num_samples, A, self.num_future_steps, self.space_dim).transpose(0,1)

            traj_eval = pred_trajs[eval_mask]
            valid_mask_eval = reg_mask[eval_mask]
            pi_eval = torch.ones((traj_eval.shape[0], 1)).to(gt.device) # probabilities are not considered and are set to 1.

            self.minADE_1.update(pred=traj_eval[..., :self.space_dim], target=gt_eval[..., :self.space_dim], prob=pi_eval,
                                valid_mask=valid_mask_eval)

            self.minFDE_1.update(pred=traj_eval[..., :self.space_dim], target=gt_eval[..., :self.space_dim], prob=pi_eval,
                               valid_mask=valid_mask_eval)


            self.log('val_minADE_1', self.minADE_1, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minFDE_1', self.minFDE_1, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        
        
        
    
    def evaluate_rollouts(self, data, pred_trajs):
        '''
        data: Heterodata
        
        '''

        target_agent_mask = data["agent"]["timestep_y_mask"].new_zeros(data['agent']['num_nodes'])
        data['agent']['av_index'] += data['agent']['ptr'][:-1]
        eval_mask = data['agent']['track_category_origin'] == 3
        eval_mask[data['agent']['av_index']] = True

        self.diversity_metrics.update(pred=pred_trajs[:, :, :, -1],
                                      targets= eval_mask
                                     )
        
        agent_ids_int = [list(map(int, sublist)) for sublist in data["agent"]["agent_ids"]]
        scenario_rollouts = get_scenario_rollouts(scenario_id = data['scenario_id'],
                                                  agent_id = agent_ids_int,
                                                  agent_batch = data['agent']['batch'],
                                                  pred_traj = pred_trajs[..., :2],
                                                  pred_z = pred_trajs[..., 2],
                                                  pred_head = pred_trajs[..., 3])
        
        
        scenarios = []
        for scenario_id in data['scenario_id']:
            with open(self.wosac_data_path + str(scenario_id) + ".pkl", "rb") as f:
                scenarios.append(pickle.load(f))
        self.wosac_metrics.update(scenarios, scenario_rollouts)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        scheduler = WarmupCosineLR(optimizer=optimizer, min_lr=1e-8, max_lr=self.lr, warmup_epochs=self.T_warm_up, total_epochs=self.T_max)
        return [optimizer], [scheduler]
    
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = parent_parser.add_argument_group('EPDiffuser')
#         parser.add_argument('--pred_deg', type=int, default=6)
#         parser.add_argument('--input_dim', type=int, default=2)
#         parser.add_argument('--hidden_dim', type=int, default=128)
#         parser.add_argument('--num_freq_bands', type=int, default=64)
#         parser.add_argument('--space_dim', type=int, default=2)
#         parser.add_argument('--num_future_steps', type=int, default=80)
#         parser.add_argument('--num_encoder_layers', type=int, default=1)
#         parser.add_argument('--num_denoiser_layers', type=int, default=2)
#         parser.add_argument('--num_heads', type=int, default=8)
#         parser.add_argument('--head_dim', type=int, default=16)
#         parser.add_argument('--dropout', type=float, default=0.1)
#         parser.add_argument('--pl2pl_radius', type=float, default = 150)
#         parser.add_argument('--pl2a_radius', type=float, default=150)
#         parser.add_argument('--a2a_radius', type=float, default=150)
#         parser.add_argument('--lr', type=float, default=5e-4)
#         parser.add_argument('--weight_decay', type=float, default=0.)
#         parser.add_argument('--T_max', type=int, default=64)
#         parser.add_argument('--homogenizing', type=bool, default=True)

#         return parent_parser