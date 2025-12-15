'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import math
from typing import List, Optional

import torch
import torch.nn as nn

from predictor.utils.ep_utils import weight_init
from predictor.models.ep.layers.mlp_layer import MLPLayer

class EmbeddingBlock(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 output_dim,
                 concat = False):
        super(EmbeddingBlock,self).__init__()
        self.concat = concat
        self.dense_layers = nn.Sequential(
                                        nn.Linear(input_dim, hidden_dim),
                                        nn.LayerNorm(hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        )
        
        if self.concat:
            self.output_layer = nn.Sequential(
                                            nn.Linear(2*hidden_dim, hidden_dim),
                                            nn.LayerNorm(hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, output_dim),
                                            )
            self.norm = nn.LayerNorm(2*hidden_dim)
        else:
            self.output_layer = nn.Sequential(
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.LayerNorm(hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, output_dim),
                                            )
            self.norm = nn.LayerNorm(hidden_dim)
            
        self.apply(weight_init)
        
        
    def forward(self, 
             continuous_inputs: Optional[torch.Tensor] = None, 
             categorical_embs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        x = self.dense_layers(continuous_inputs)
        categorical_embs_sum = None
        if categorical_embs is None:
            return self.output_layer(self.norm(x))
        else:
            categorical_embs_sum = torch.stack(categorical_embs).sum(dim=0)
            if self.concat:
                return self.output_layer(self.norm(torch.concat([x, categorical_embs_sum], axis=-1)))
            else:
                return self.output_layer(self.norm(x + categorical_embs_sum))