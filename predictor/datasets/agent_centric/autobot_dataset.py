"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

from .agent_centric_dataset import AgentCentricBaseDataset


class AutoBotEgoDataset(AgentCentricBaseDataset):

    def __init__(self, config, data_splits):
        super().__init__(config, data_splits)
