'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

from typing import Dict, List, Mapping, Optional
import json
import warnings
import diffusers

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph

from predictor.models.ep.layers.attention_layer import AttentionLayer
from predictor.models.ep.layers.fourier_embedding import FourierEmbedding
from predictor.models.ep.layers.embedding_block import EmbeddingBlock
from predictor.models.ep.layers.mlp_layer import MLPLayer
from predictor.utils.ep_utils.weight_init import weight_init


class EPDenoiser(nn.Module):
    def __init__(self,
                 timesteps: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 num_freq_bands: int,
                 pred_deg: int = 6,
                 space_dim: int = 2,) -> None:
        super(EPDenoiser, self).__init__()
        self.timesteps = timesteps
        self.noise_schedule = diffusers.DDPMScheduler(num_train_timesteps=timesteps+1, beta_schedule="scaled_linear")
        self.b_t = self.noise_schedule.betas
        self.a_t = self.noise_schedule.alphas
        self.ab_t = self.noise_schedule.alphas_cumprod

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        
        self.pred_deg=pred_deg
        self.space_dim=space_dim
    
        input_dim_y_a = (self.pred_deg)*self.space_dim
        
        ###################### trainable parts ############################
        self.t_embed = MLPLayer(input_dim = 1,
                                hidden_dim = hidden_dim,
                                output_dim = hidden_dim)

        
        self.sample_embed = FourierEmbedding(input_dim=input_dim_y_a, hidden_dim=hidden_dim,
                                             num_freq_bands=num_freq_bands)
        
        self.pl2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        
        self.output_layer = MLPLayer(input_dim = hidden_dim,
                                    hidden_dim = hidden_dim,
                                    output_dim = (self.pred_deg)*self.space_dim)
        
        
        self.apply(weight_init)

        
        
    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                timesteps: int = None) -> Dict[str, torch.Tensor]:
        '''
        if t is not set, then generate t randomly
        '''
        A = data['agent']['num_nodes']
        y = data['agent']['cps_mean_fut_transform'].contiguous()
        device=data['agent']['cps_mean'].device
        x_gt = (y[:, 1:] - y[:, :-1]).view(A, -1)
        
        ### diffusion ###
        noise = self.sample_noise(num_samples=1, num_agents=A, device=device)
        
        if timesteps is None:
            t = torch.randint(1, self.timesteps, (1,A,1)).to(device) 
        else:
            t = torch.full((1,A,1), timesteps, dtype=torch.long).to(device)
            
        x_pert = self.perturb_input(x0=x_gt, t=t, noise = noise)
        
        ### noise prediction ###
        pred_noise = self.pred_noise(data=data,
                                     scene_enc=scene_enc,
                                     samples=x_pert,
                                     t=t)
        
        noise_cum = torch.cumsum(noise.reshape(noise.shape[0], noise.shape[1], -1, 2), dim=-2).reshape(noise.shape[0], noise.shape[1], -1)
        pred_noise_cum = torch.cumsum(pred_noise.reshape(pred_noise.shape[0], pred_noise.shape[1], -1, 2), dim=-2).reshape(pred_noise.shape[0], pred_noise.shape[1], -1)
        
        ### one step denoising ###
        x0 = self.get_x0(x_pert,
                         pred_noise, 
                         t).view(1, A, self.pred_deg, self.space_dim)
        
        x0_start = x0.new_zeros((x0.shape[0], x0.shape[1], 1, 2))
        x0 = torch.concat([x0_start, x0], dim=-2)
        x0 = torch.cumsum(x0, dim=-2).view(x0.shape[0], x0.shape[1], -1)

        return {'target_noise': noise,
                'pred_noise': pred_noise,
                'target_noise_cum': noise_cum,
                'pred_noise_cum': pred_noise_cum,
                'pred_x0': x0,
               }
    
    
    def pred_noise(self,
                  data: HeteroData,
                  scene_enc: Mapping[str, torch.Tensor],
                  samples, 
                  t):      
        '''
        data: data
        scene_enc: condition tokens from encoder
        samples: [num_samples, A, 12]
        t: [num_samples, A, 1]
        '''
        
        num_samples, A, _ = samples.shape
        t_embed = self.t_embed(t/self.timesteps)
        
        mask = data['agent']['timestep_x_mask'][:, :].contiguous() #[A,50]
        mask_a = mask[:, -1] #[A]
        
        x_a = scene_enc['x_a']
        r_pl2a = scene_enc['r_pl2a']
        r_a2a = scene_enc['r_a2a']
        edge_index_pl2a = scene_enc['edge_index_pl2a']
        edge_index_a2a = scene_enc['edge_index_a2a']
        
        y_a = self.sample_embed(continuous_inputs=samples, 
                                categorical_embs=[t_embed]) # [num_samples, A, d_hidden]
        y_a =  y_a + x_a.unsqueeze(0).repeat(num_samples, 1, 1)
        
        r_pl2a = r_pl2a.repeat(num_samples, 1)
        r_a2a = r_a2a.repeat(num_samples, 1) # [num_samples*A, d_hidden]
        y_a = y_a.view(num_samples*A, -1) #[num_samples*A, d_hidden]

        
        edge_index_pl2a = torch.cat(
            [edge_index_pl2a + i * edge_index_pl2a.new_tensor([[0], [data['agent']['num_nodes']]]) for i in
             range(num_samples)], dim=1)

        edge_index_a2a = torch.cat(
            [edge_index_a2a + i * edge_index_a2a.new_tensor([data['agent']['num_nodes']]) for i in
             range(num_samples)], dim=1)

        for i in range(self.num_layers):
            y_a = self.pl2a_attn_layers[i]((scene_enc['x_pl'], y_a), r = r_pl2a, edge_index=edge_index_pl2a)
            y_a = self.a2a_attn_layers[i](y_a, r = r_a2a, edge_index=edge_index_a2a)

            
        y_a = y_a.reshape(num_samples, A, -1)  
        pred_noise = self.output_layer(y_a)
        
        return pred_noise
    
    
    def perturb_input(self, x0, t, noise):
        '''
        x0: clean data (vectors between control points) # [A, degree*space_dim]
        t: timestep, int
        noise: same dim to x
        '''
        ab=self.ab_t.to(x0.device)[t]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    
    
    def get_x0(self, x_pert, pred_noise, t):
        ab=self.ab_t.to(x_pert.device)[t]
        return (x_pert - (1 - ab).sqrt() * pred_noise) / ab.sqrt()
        
        
    @torch.no_grad()
    def sample(self, 
               data: HeteroData, 
               scene_enc: Mapping[str, torch.Tensor],
               num_samples: int,
               num_denoising_steps: int = None, 
               temperature = 1.0,
               method: str = 'ddim'):
        A = data['agent']['num_nodes']
        device=data['agent']['cps_mean'].device
        samples = self.sample_noise(num_samples = num_samples, num_agents=A, device=device)  * temperature

        if method == 'ddpm': # ddpm
            if num_denoising_steps is not None:
                warnings.warn('Warning: ddpm do not consider num_denoising_steps.')

            for i in range(self.timesteps, 0, -1):
                # reshape time tensor
                t = torch.full((num_samples,data['agent']['num_nodes'],1), i, dtype=torch.long).to(device) 
  
                # sample some random noise to inject back in. For i = 1, don't add back in noise
                if i > 1:
                    z = self.sample_noise(num_samples = num_samples, num_agents=A, device=device)
                else:
                    z=0
                eps = self.pred_noise(data=data,
                                     scene_enc=scene_enc,
                                     samples = samples, 
                                     t=t,)    # predict noise e_(x_t,t)
                              
                samples = self.denoise_ddpm(samples, i, eps, z)
                samples_out = samples.view(samples.shape[0], samples.shape[1], -1, 2)
                samples_start = samples.new_zeros((samples.shape[0], samples.shape[1], 1, 2))
                samples_out = torch.concat([samples_start, samples_out], dim=-2)
                samples_out = torch.cumsum(samples_out, dim=-2)
                              
        elif method == 'ddim': # ddim
            if num_denoising_steps is None:
                Exception('Error: ddim requires num_denoising_steps.')
                              
            step_size = self.timesteps // num_denoising_steps
            for i in range(self.timesteps, 0, -step_size):
                t = torch.full((num_samples,data['agent']['num_nodes'],1), i, dtype=torch.long, device=device)
                eps = self.pred_noise(data=data,
                                     scene_enc=scene_enc,
                                     samples = samples, 
                                     t=t)   # predict noise e_(x_t,t)

                samples = self.denoise_ddim(samples, i, i - step_size, eps)
                samples_out = samples.view(samples.shape[0], samples.shape[1], -1, 2)
                samples_start = samples.new_zeros((samples.shape[0], samples.shape[1], 1, 2))
                
                samples_out = torch.concat([samples_start, samples_out], dim=-2)
                samples_out = torch.cumsum(samples_out, dim=-2)
                
        else:
            raise Exception("Error: unknown sampling method")
        
        samples_out = samples_out.view(samples.shape[0], samples.shape[1], -1)
        return samples_out
    
    
    def sample_noise(self, num_samples, num_agents, device):
        return torch.randn(num_samples, num_agents, self.pred_deg*self.space_dim).to(device)
        
    
    def denoise_ddim(self, x, t, t_prev, pred_noise):
        ab = self.ab_t.to(x.device)[t]
        ab_prev = self.ab_t.to(x.device)[t_prev]

        x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise

        return x0_pred + dir_xt
    
    
    def denoise_ddpm(self, x, t, pred_noise, z):
        noise = self.b_t.sqrt().to(x.device)[t] * z
        mean = (x - pred_noise * ((1 - self.a_t.to(x.device)[t]) / (1 - self.ab_t.to(x.device)[t]).sqrt())) / self.a_t.to(x.device)[t].sqrt()
        return mean + noise