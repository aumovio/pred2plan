'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

from typing import Dict

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from predictor.models.ep.layers.attention_layer import AttentionLayer
from predictor.models.ep.layers.fourier_embedding import FourierEmbedding
from predictor.models.ep.layers.embedding_block import EmbeddingBlock
from predictor.models.ep.layers.mlp_layer import MLPLayer
from predictor.utils.ep_utils.geometry import angle_between_2d_vectors
from predictor.utils.ep_utils.weight_init import weight_init
from predictor.utils.ep_utils.geometry import wrap_angle

from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph

import numpy as np


class EPEncoder(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 pl2pl_radius: float,
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 num_freq_bands: int,
                 ) -> None:
        super(EPEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pl2pl_radius = pl2pl_radius
        self.pl2a_radius=pl2a_radius
        self.a2a_radius=a2a_radius
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        #self.homogenizing = homogenizing

        input_dim_x_pl = 6
        input_dim_x_a = 4
        input_dim_r_pl2pl = 5
        input_dim_r_pl2a= 5
        input_dim_r_a2a = 5

        self.type_pl_emb = nn.Embedding(20, hidden_dim)

        # if not self.homogenizing: # using all info available in argoverse 2, NOT the case by default
        #     self.bound_type_pl_emb = nn.Embedding(17, hidden_dim)
        #     self.int_pl_emb = nn.Embedding(3, hidden_dim)

        self.x_pl_emb = FourierEmbedding(input_dim=input_dim_x_pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands) 
        self.r_pl2pl_emb = FourierEmbedding(input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        
        
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.type_a_emb = nn.Embedding(5, hidden_dim)
        self.x_a_time_emb = MLPLayer(input_dim=2, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        
        self.pl2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
 
        self.apply(weight_init)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        ### encode map ###
        cps_pl = data['map']['mapel_cps'].contiguous()
        pos_pl = data['map']['reference_pos'].contiguous()
        orient_pl = data['map']['reference_heading'].contiguous()
        orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)
        
        origin = pos_pl
        cos, sin = orient_pl.cos(), orient_pl.sin()

        rot_mat = orient_pl.new_zeros(data['map']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        x_pl = torch.bmm(cps_pl.view(data['map']['num_nodes'], -1, 2) - origin[:, :2].unsqueeze(1), rot_mat).view(cps_pl.shape) # normalized control points [A,12,2]
        x_pl = (x_pl[:, 1:, :] - x_pl[:, :-1, :]).view(data['map']['num_nodes'], -1) # vectors between control points
        
        x_pl_categorical_embs = [self.type_pl_emb(data['map']['mapel_types'].long()),]
        
        # if self.homogenizing:
        #     x_pl_categorical_embs = [self.type_pl_emb(data['map']['mapel_type'].long()),
        #                              ]
        # else:
        #     x_pl_categorical_embs = [self.bound_type_pl_emb(data['map']['mapel_boundary_type'][:, 0].long()), # left bound
        #                              self.bound_type_pl_emb(data['map']['mapel_boundary_type'][:, 1].long()), # right bound
        #                              self.type_pl_emb(data['map']['mapel_type'].long()),
        #                              self.int_pl_emb(data['map']['mapel_is_intersection'].long())]
            
        x_pl = self.x_pl_emb(continuous_inputs=x_pl, categorical_embs=x_pl_categorical_embs)
        
        
        ### encode agents ### 
        x_a = data['agent']['cps_mean_transform'].contiguous() # [A, 6, 2]
        x_a = (x_a[:, 1:] - x_a[:, :-1]).view(data['agent']['num_nodes'], -1)
        x_a = self.x_a_emb(continuous_inputs=x_a, 
                           categorical_embs=[self.x_a_time_emb(data['agent']['time_window'].contiguous()), 
                                             self.type_a_emb(data['agent']['object_type'].long())])
        
        pos_a = data['agent']['x'][:, -1, :2].contiguous() # [A, 2]
        head_a = data['agent']['x'][:, -1, 2].contiguous() # [A]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        mask = data['agent']['timestep_x_mask'][:, :].contiguous() #[A,50]
        mask_a = mask[:, -1] #[A]
        
        ### relative position pl2a ###
        pos_pl = data['map']['reference_pos'].contiguous() # [M, 2]
        orient_pl = data['map']['reference_heading'].contiguous() # [M]

        edge_index_pl2a = radius(x=pos_a[:, :2], 
                                 y=pos_pl[:, :2], 
                                 r=self.pl2a_radius, 
                                 batch_x=data['agent']['batch'], 
                                 batch_y=data['map']['batch'], 
                                 max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_a[edge_index_pl2a[1]]]
        
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_a[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_a[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             torch.cos(angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2])),
             torch.sin(angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2])),
             torch.cos(rel_orient_pl2a),
             torch.sin(rel_orient_pl2a)], dim=-1)
        r_pl2a = self.r_pl2a_emb(r_pl2a) #[A, d_hidden]
        
        
        ### relative position a2a ###               
        edge_index_a2a = radius_graph(x=pos_a[:, :2], 
                                      r=self.a2a_radius, 
                                      batch=data['agent']['batch'], 
                                      loop=False,
                                      max_num_neighbors=300)
        
        edge_index_a2a = subgraph(subset=mask_a, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_a[edge_index_a2a[0]] - pos_a[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_a[edge_index_a2a[0]] - head_a[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             torch.cos(angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2])),
             torch.sin(angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2])),
             torch.cos(rel_head_a2a),
             torch.sin(rel_head_a2a),], dim=-1)
        r_a2a = self.r_a2a_emb(r_a2a) #[A, d_hidden]
        
        
        ### pl2pl ###
        edge_index_pl2pl = radius_graph(x=pos_pl[:, :2], 
                                        r=self.pl2pl_radius,
                                        batch=data['map']['batch'] if isinstance(data, Batch) else None,
                                        loop=False, 
                                        max_num_neighbors=150)

        rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]]
        rel_orient_pl2pl = wrap_angle(orient_pl[edge_index_pl2pl[0]] - orient_pl[edge_index_pl2pl[1]])

        r_pl2pl = torch.stack(
            [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
             torch.cos(angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]], nbr_vector=rel_pos_pl2pl[:, :2])),
             torch.sin(angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]], nbr_vector=rel_pos_pl2pl[:, :2])),
             torch.cos(rel_orient_pl2pl), 
             torch.sin(rel_orient_pl2pl)], dim=-1)
        
        r_pl2pl = self.r_pl2pl_emb(r_pl2pl)

        for i in range(self.num_layers):
            x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl)

        return {'x_pl': x_pl, 
                'x_a': x_a,
                'r_pl2a': r_pl2a,
                'edge_index_pl2a': edge_index_pl2a,
                'r_a2a': r_a2a,
                'edge_index_a2a': edge_index_a2a}
    
    
