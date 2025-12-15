"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import torch
from torch_geometric.data import HeteroData

from predictor.datasets.transforms.base_target_builder import BaseTargetBuilder
from predictor.utils.ep_utils.basis_function import basis_function_b
from predictor.utils.ep_utils.geometry import wrap_angle
import numpy as np

NUM_SAMPLE_POINTS = 9
    
Phi_B = torch.tensor(basis_function_b(tau= np.linspace(0,1,NUM_SAMPLE_POINTS), # Bernstein basis function for map elements
                                      n=3,
                                      d=2, 
                                      k=0,
                                      delta_t=1.))

Phi_Bp = torch.tensor(basis_function_b(tau= np.linspace(0,1,NUM_SAMPLE_POINTS), # Bernstein basis function (prime) for map elements
                                      n=3,
                                      d=2, 
                                      k=1,
                                      delta_t=1.))

class EPTargetBuilder(BaseTargetBuilder):

    def __init__(self,
                 config) -> None:
        self.num_historical_steps = config.data.past_len
        self.num_future_steps = config.data.future_len

    def __call__(self, data: HeteroData) -> HeteroData:
        origin = data['agent']['x'][:, -1, :2]
        theta = data['agent']['x'][:, -1, 2]
        cos, sin = theta.cos(), theta.sin()

        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        
        if 'y' in data['agent'].keys():
            data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_future_steps, 3)
            data['agent']['target'][..., :2] = torch.bmm(data['agent']['y'][:, :, :2] -
                                                         origin[:, :2].unsqueeze(1), rot_mat)
            data['agent']['target'][..., 2] = wrap_angle(data['agent']['y'][:, :, 2] - theta.unsqueeze(-1))
        
        data['agent']['agent_R'] = rot_mat
        data['agent']['agent_origin'] = origin
        
        data['agent']['cps_mean_transform'] = origin.new_zeros(data['agent']['cps_mean'].shape)
        data['agent']['cps_mean_transform'] = torch.bmm(data['agent']['cps_mean'][:, :, :2] -
                                                     origin[:, :2].unsqueeze(1), rot_mat)
        
        if 'cps_mean_fut' in data['agent'].keys():
            data['agent']['cps_mean_fut_transform'] = origin.new_zeros(data['agent']['cps_mean_fut'].shape)
            data['agent']['cps_mean_fut_transform'] = torch.bmm(data['agent']['cps_mean_fut'][:, :, :2] -
                                                         origin[:, :2].unsqueeze(1), rot_mat)

        lane_cl_cps = data['map']['mapel_cps'] #[M,4,2] (-1) means the lane center not boundaries

        reference_pos = Phi_B[None, :, :].to(lane_cl_cps.device) @ lane_cl_cps
        reference_head_vec = Phi_Bp[None, :, :].to(lane_cl_cps.device) @ lane_cl_cps
        reference_heading = torch.atan2(reference_head_vec[..., 1], reference_head_vec[..., 0])
        data['map']['reference_pos'] = reference_pos[:, 4] # middle point as reference
        data['map']['reference_heading'] = reference_heading[:, 4] # middle point as reference
        return data