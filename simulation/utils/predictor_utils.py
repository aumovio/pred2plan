"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict, OmegaConf, DictConfig
from os import getcwd, cpu_count
from predictor.utils.seed import set_seed
import numpy as np
from copy import deepcopy
from torch import tensor, Tensor
from typing import Dict, List, Mapping, Optional
from torch_geometric.data import Batch, HeteroData

def load_config(cfg_path:str, predictor_cfg_path:str, method:str, data_path:str="data_samples/nuscenes") -> DictConfig:
    """ Load general config and config for specified predictor (method)

    :param cfg_path     : path to general config file (usually "unitraj/configs/")
    :param predictor_cfg_path   : path to config file for specified predictor (usually "unitraj/configs/method/")
    :param method       : name of specified predictor (e.g. "autobot")
    :param data_path    : specifies the path where the scenario is located (default="data_samples/nuscenes/")

    :return cfg: loaded configuration as DictConfig
    """
    def _load_cfg_file(cfg_path, job_name):
        """ open and read config file """
        with initialize(version_base=None, config_path=cfg_path, job_name=job_name):
            cfg = compose(config_name=job_name)
        return cfg
    
    GlobalHydra.instance().clear()
    # load config.yaml
    cfg = _load_cfg_file(cfg_path, 'config')
    # load method.yaml
    predictor_cfg = _load_cfg_file(predictor_cfg_path, method)

    # overwrite cfg with predictor_cfg
    cfg.model = predictor_cfg.model
    with open_dict(cfg):
        for key in predictor_cfg.training.keys():
            cfg['training'][key] = predictor_cfg.training[key]
        for key in predictor_cfg.data.keys():
            cfg['data'][key] = predictor_cfg.data[key]

    # settings to load only one scenario
    this_dir = getcwd()
    cfg.paths.logs_path = f"{this_dir}/results"
    cfg.paths.artifacts_path = f"{this_dir}/results"
    cfg.paths.data_root = f"{this_dir}/{data_path.split('/')[0]}"
    cfg.paths.train_data_path = None
    cfg.paths.val_data_path = [f"{this_dir}/{data_path}"]
    cfg.paths.test_data_path = None
    cfg.paths.cache_path = f"{this_dir}/{data_path}/cache_{method}"
    cfg.data.load_num_workers = cpu_count() -1

    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # open the config struct
    cfg['eval'] = True
    return cfg


def rot_mat(theta):
    theta = theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return rot


def transform_batch_to_current_ego_obs(
        batch, idx, map_og, map_center_og, md, t, N, vx_0, vy_0, history_len=21,
        ):
    """
    Transform map and obstacles from global coords to local coords of current ego-obstacle
    :param batch: batch_dict, holding all information about the current scenario states
    :param idx  : index in batch_dict of current obstacle to predict
    :param map_og       : original map, not transformed in any way
    :param map_center_og: original map center, not transformed in any way
    :param md   : original motion data of all objects with shape [15, 81, 4]
    :param t    : current timestep
    :param N    : prediction horizon
    :param vx_0 : vx of first timestep of original ego
    :param vy_0 : vy of first timestep of original ego
    :param hist_len : length of history (default 2.1s)
    
    :return R    : rotation matrix R
    :return batch: updated batch_dict
    """
    # deepcopy original to avoid unintended changes
    motion_data = deepcopy(md)  # shape [15, 81, 4] (x, y, vx, vy)
    
    # compute rotation matrix R
    if t < N:
        vx_1 = motion_data[idx, t+history_len-1, 2].cpu().numpy()   # -1 because hist has max_index 20
        vy_1 = motion_data[idx, t+history_len-1, 3].cpu().numpy()
    else:  # vx and vy of very last point in gt are zero, so we use values of point before
        vx_1 = motion_data[idx, t+history_len-1, 2].cpu().numpy()
        vy_1 = motion_data[idx, t+history_len-1, 3].cpu().numpy()
    phi_1 = np.arctan2(vy_1, vx_1)
    phi_0 = np.arctan2(vy_0, vx_0)
    R = rot_mat(phi_1 - phi_0)

    # move and rotate
    batch['input_dict']['map_polylines'][0, :, :, :2] = \
        (map_og[0, :, :, :2] - motion_data[idx, t+history_len-1, :2]) @ R
    batch['input_dict']['map_polylines_center'][0, :, :2] = \
        (map_center_og[0, :, :2] - motion_data[idx, t+history_len-1, :2]) @ R
    motion_data[:, :, :2] = tensor(
        (motion_data[:, :, :2] - motion_data[idx, t+history_len-1, :2]).cpu().numpy() @ R       # x, y
    )
    motion_data[:, :, 2:] = tensor(
        motion_data[:, :, 2:].cpu().numpy() @ R     # vx, vy
    )
    
    # update history of all obstacles in batch dict
    batch['input_dict']['obj_trajs'][0, :, :, 0:2] = motion_data[:,t:t+history_len, :2]   # x,y
    batch['input_dict']['obj_trajs'][0, :, :, 35:37] = motion_data[:,t:t+history_len, 2:] # vx,vy

    return R, batch


def batch_dict_tensors_to_device(batch_dict:dict, device:str='cuda') -> dict:
    """ Moves all tensors in batch_dict to given device
        uses and returns a copy of the batch_dict, so the original remains unchanged
    """
    bd = deepcopy(batch_dict)
    if "input_dict" in bd and isinstance(bd["input_dict"], dict):
        for key, item in bd["input_dict"].items():
            if isinstance(item, Tensor):
                bd["input_dict"][key] = item.clone().detach().to(device)
    else:
        bd = bd.to(device)
    return bd


def get_simulation_input(predictor_input):
    if isinstance(predictor_input, Dict): # agent-centric data
        return {"oids": predictor_input["input_dict"]["center_objects_id"],
                "predicted_track_difficulty": predictor_input["input_dict"]["center_track_difficulty"].cpu().numpy(),
                "predicted_track_type": predictor_input["input_dict"]["center_track_type"].cpu().numpy(),
                "ego_track_difficulty": predictor_input["input_dict"]["ego_track_difficulty"].cpu().numpy(),}
    else: # query-centric data
        target_mask = predictor_input['agent']['train_mask']
        oids = (np.array(predictor_input['agent']['id'])[target_mask]).tolist()
        
        return {"oids": oids,
                "predicted_track_difficulty": None,
                "predicted_track_type": None,
                "ego_track_difficulty": None,}