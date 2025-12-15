"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
from torch_geometric.transforms import BaseTransform

class BaseTargetBuilder(BaseTransform):
    def __init__(self,
                 config) -> None:
        self.config = config
        
    def forward(self, data):
        return data