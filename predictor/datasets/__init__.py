from .agent_centric.MTR_dataset import MTRDataset
from .agent_centric.autobot_dataset import AutoBotEgoDataset
from .agent_centric.wayformer_dataset import WayformerDataset
from .query_centric.smart_dataset import SmartDataset
from .query_centric.map_constvel_dataset import MapConstVelDataset
from .query_centric.ep_dataset import EPDataset
from .wosac_transform_dataset import WOSACDataset
from typing import List, Dict
from torch.utils.data import Dataset

# for target builder
from .transforms.base_target_builder import BaseTargetBuilder
from .transforms.ep_target_builder import EPTargetBuilder

__all__ = ["MTRDataset", "AutoBotDataset", "WayformerDataset", "SmartDataset", "EPDataset",]

name_constructor_map = {
    'autobotEgo': AutoBotEgoDataset,
    'wayformer': WayformerDataset,
    'MTR': MTRDataset,
    "smart": SmartDataset,
    'constvel': AutoBotEgoDataset,
    'mapconstvel': MapConstVelDataset,
    'ep_diffuser': EPDataset,
    'ep': EPDataset,
    'wosac': WOSACDataset,
}

name_transform_map = {
    'autobotEgo': BaseTargetBuilder,
    'wayformer': BaseTargetBuilder,
    'MTR': BaseTargetBuilder,
    "smart": BaseTargetBuilder,
    'constvel': BaseTargetBuilder,
    'ep_diffuser': EPTargetBuilder,
    'ep': EPTargetBuilder,
}

def build_dataset(config, data_splits: List[Dict]) -> Dataset:
    dataset = name_constructor_map[config.model.model_name](
        config, data_splits
    )
    dataset.load_from_cache()
    return dataset

def build_transform(config) -> Dataset:
    target_builder = name_transform_map[config.model.model_name](
        config
    )
    return target_builder

def cache_data_splits(config, data_splits: List[Dict]) -> None:
    name_constructor_map[config.model.model_name](config, data_splits).cache_data()
