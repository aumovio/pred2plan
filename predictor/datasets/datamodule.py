"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

from typing import Optional, List, Dict, Any
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
#from torch_geometric.loader import DataLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pathlib import Path
from datasets import build_dataset, cache_data_splits
from omegaconf import DictConfig
from predictor.datasets.types import object_type, polyline_type, track_type

split_defaults = {
    "ego_track_difficulty": [0.0, np.inf],
    "center_track_difficulty": [0.0, np.inf],
    "ego_track_types": set(track_type.keys()),
    "center_track_types": set(track_type.keys()),
    "center_object_types": ['VEHICLE', 'PEDESTRIAN', 'CYCLIST'],
    "starting_frames": 0,
    "sample_num": 1.0
}

class DataModule(LightningDataModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.drop_last = config.data.drop_last
        self.num_workers = config.data.load_num_workers // len(config.trainer.devices)
        self.prefetch_factor = config.data.prefetch_factor
        self.pin_memory = config.data.pin_memory
        # collect split-specific configuration
        self.data_root = self.config.data.data_root
        self.cache_root = self.config.data.cache_root
        self.batch_size = max(self.config.trainer["batch_size"] // len(self.config.trainer.devices) // self.config.data.bucket_size, 1)
        self.train_splits = self.collect_splits(**self.config.data.train)
        self.val_splits = self.collect_splits(**self.config.data.val)
        self.test_splits = self.collect_splits(**self.config.data.test)
        self.predict_splits = self.collect_splits(**self.config.data.predict)

    def collect_splits(self, splits: List[str], starting_frames: List[int], tracks_to_predict: Dict[List, Any], filter: Dict[List, Any]) -> List:
        num_splits = len(splits)
        collected_splits = []
        for i in range(num_splits):
            split = splits[i]
            collected_split = {"filter":{}}
            # properties
            collected_split["split"] = split
            collected_split["starting_frame"] = starting_frames[i] if starting_frames else split_defaults["starting_frames"]
            for k, v in filter.items():
                collected_split["filter"][k] = v[i] if v else split_defaults[k]
            # paths and cache_names
            collected_split["tracks_to_predict"] = tracks_to_predict.selection[i]
            collected_split["data_path"] = Path(self.data_root) / split.lstrip("/")
            cache_name =  f"s{starting_frames[i]}l{self.config.data.past_len}f{self.config.data.future_len}_{tracks_to_predict.selection[i]}"
            collected_split["cache_name"] = cache_name
            collected_split["cache_path"] = Path(self.cache_root) / split.lstrip("/") / cache_name
            collected_splits.append(collected_split)
        return collected_splits

    def prepare_data(self):
        """
        Before parallelization prepare data by preprocessing and caching it (model-specific)
        """
        unique_splits = {split["cache_path"]: split for split in [*self.train_splits, *self.val_splits, *self.test_splits, *self.predict_splits]}
        unique_splits = list(unique_splits.values())
        cache_data_splits(self.config, unique_splits)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        During parallelization load data as needed from available caches
        """
        if stage == "fit":
            self.train_dataset = build_dataset(self.config, self.train_splits)
            self.val_dataset = build_dataset(self.config, self.val_splits)
        if stage == "test":
            self.test_dataset = build_dataset(self.config, self.test_splits)
        if stage == "predict":
            self.predict_dataset = build_dataset(self.config, self.predict_splits)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, prefetch_factor=self.prefetch_factor,
                          num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.train_dataset.collate_fn,
                          pin_memory=self.pin_memory
                          )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, prefetch_factor=self.prefetch_factor,
                          num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.val_dataset.collate_fn,
                          pin_memory=self.pin_memory
                          )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, prefetch_factor=self.prefetch_factor,
                          num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.test_dataset.collate_fn,
                          pin_memory=self.pin_memory
                          )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, prefetch_factor=self.prefetch_factor,
                          num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.predict_dataset.collate_fn,
                          pin_memory=self.pin_memory
                          )
