# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
from torchmetrics import Metric

from .utils import topk_joint
from .utils import valid_filter
from torch_scatter import segment_csr

class avgminADE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(avgminADE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'ADE',
               ptr: Optional[torch.Tensor] = None,) -> None:
        
        '''
        pred: [K, A, 60, 2]
        target: [A, 60 ,2]
        prob: [B, K]
        valid_mask: [A, 60]
        '''
        assert ptr != None
        assert pred.shape[0] == prob.shape[1]
        assert prob.shape[0] == (ptr.shape[0] -1)
        
        #pred, target, prob, valid_mask, ptr = valid_filter(pred, target, prob, valid_mask, keep_invalid_final_step = True, ptr=ptr)
    
        pred_topk, _ = topk_joint(self.max_guesses, pred, prob, ptr)

        l2_norm = torch.norm(pred_topk -
                             target.unsqueeze(0),
                               p=2, dim=-1) *  valid_mask.unsqueeze(0) # [K,A,60]

        l2_norm = segment_csr(src=l2_norm.transpose(0,1), indptr=ptr, reduce='sum').sum(dim=-1) # [B, K]
        num_agents_batch = segment_csr(src=torch.where(valid_mask, 1.0, 0.0), indptr=ptr, reduce='sum').sum(dim=-1) # [B]

        best_mode = l2_norm.argmin(dim=-1) # [B]

        self.sum += (l2_norm[torch.arange(l2_norm.shape[0]), best_mode] / num_agents_batch).sum()
        self.count += l2_norm.size(0)
        


    def compute(self) -> torch.Tensor:
        return self.sum / self.count