"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import sys
import os
# %load_ext autoreload
# %autoreload 2
basepath = os.path.dirname(os.path.dirname(__file__))
if not basepath in sys.path:
    sys.path.insert(0, basepath)
    print(f"sys PATH now includes: '{basepath}'")

import hydra
import json
from omegaconf import DictConfig

import logging 
logger = logging.getLogger(__name__)


from nuplan.planning.script.run_simulation import run_simulation as main_simulation


@hydra.main(version_base=None, config_path="../configs", config_name="simulation")
def run(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.planner.modular_planner.predictor = cfg.predictor
    cfg.simulator.planner = cfg.planner
    main_simulation(cfg.simulator)

    
if __name__ == '__main__':
    run()


