"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

from collections import defaultdict, OrderedDict
from omegaconf import DictConfig


from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
import torch
import json
import pandas as pd
import hashlib
from copy import deepcopy
import compress_pickle as pickle
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
from torch_geometric.data import HeteroData, Batch
import time
from tqdm import tqdm
from pathlib import Path
from functools import partial, partialmethod

from metadrive.scenario import utils as sd_utils

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos.scenario_pb2 import Scenario, Track, ObjectState, RequiredPrediction
from waymo_open_dataset.protos.map_pb2 import Map, MapFeature, LaneCenter, Crosswalk, RoadEdge, RoadLine, MapPoint 

from predictor.datasets.base_dataset import BaseDataset, plot_processed_data, plot_tokenized_data, plot_multiple_cur_and_split
from predictor.datasets.types import object_type, polyline_type, track_type
from predictor.utils.visualization import check_loaded_data
from predictor.datasets.transforms.ep_target_builder import EPTargetBuilder
from predictor.datasets.data_utils import read_scenario

from predictor.utils.ep_utils.preprocess_utils import map_utils, tracking_utils
from predictor.utils.ep_utils.preprocess_utils.tracker import Polynomial_Tracker


def safe_int_strict(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class WOSACDataset(BaseDataset):

    def __init__(self, config: DictConfig, data_splits: List[Dict]):
        self.num_historical_steps = config.data.past_len
        self.num_future_steps = config.data.future_len
        self.num_steps = self.num_historical_steps + self.num_future_steps
        print('using wosac dataset')
        super().__init__(config, data_splits)
  
    
    def _cache_datasplit(self, split_name: str, data_path: Path, cache_path: Path, starting_frame: int, tracks_to_predict: str) -> None:
        """
        Process and cache data from data path
        """
        rank_zero_info(f"Loading, processing and caching data from {data_path} into cache {cache_path}...")
        _, summary_list, mapping = sd_utils.read_dataset_summary(data_path, check_file_existence=False)

        # for debugging purposes:
        whitelist = self.config.whitelist #[scenario + self.config.compression_source for scenario in self.config.whitelist]
        if whitelist:
            summary_list = [scenario for scenario in summary_list if any(allowed in scenario for allowed in whitelist)]

        if cache_path.exists():
            rank_zero_info(f'Warning: cache at {cache_path} already exists and will be overwritten')
            shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)

        if self.config["load_num_workers"]<0:
            process_num = os.cpu_count() -1 
        else:
            process_num = self.config['load_num_workers'] #os.cpu_count() - 1
        rank_zero_info(f'Using {process_num} processes to load data from {len(summary_list)} files...')


        max_chunk_size=self.config["max_processing_chunk_size"] 
        files_per_worker = 0
        if len(summary_list) // process_num <= max_chunk_size:
            chunk_size = max(1,len(summary_list) // process_num)
        else:
            chunk_size = max_chunk_size
        data_splits = []
        for idx, i in enumerate(range(0, len(summary_list), chunk_size)):
            chunk_index = idx
            file_list = summary_list[i: i + chunk_size]
            subset_paths = [mapping[k] for k in file_list]
            data_splits.append((chunk_index, file_list, subset_paths))
            if idx % process_num == 0:
                files_per_worker += len(file_list)

        # data_splits = np.array_split(summary_list, process_num)
        # data_splits = [(i, list(data_splits[i])) for i in range(process_num)]
        process_fn = partial(self.process_data_chunk,
                        split_name = split_name,
                        data_path = data_path,
                        cache_path = cache_path,
                        starting_frame = starting_frame,
                        tracks_to_predict = tracks_to_predict,
                        total_workers = process_num,
                        total_files = files_per_worker)

        if process_num > 1: # with multiprocessing
            with Pool(processes=process_num) as pool:
                results = pool.starmap(process_fn, data_splits)
        else: # without multiprocessing 
            results = []
            for i, file_list, subset_list in data_splits:
                results.append(process_fn(i,file_list, subset_list))
                
        return
        
    
    def process_data_chunk(self, 
                           chunk_index: int, 
                           file_list: List, 
                           subset_list: List, 
                           split_name: str, 
                           data_path: Path, 
                           cache_path: Path, 
                           starting_frame: int, 
                           tracks_to_predict: str, 
                           total_workers: int, 
                           total_files: int) -> Dict:
        file_map = {}
        
        save_cnt = 0
        worker_index = chunk_index % total_workers
        processed = chunk_index // total_workers
        max_chunk_size = self.config["max_processing_chunk_size"]
        
        for cnt, file_name in enumerate(tqdm(file_list)):
            # if chunk_index == 0 and cnt % max(int(len(file_list) / 10), 1) == 0:
            #     rank_zero_info(f'{cnt}/{len(file_list)} data chunks processed', flush=True)
            if worker_index == 0 and cnt % max(len(file_list) // 5, 1) == 0:
                print(f"{cnt + processed*max_chunk_size}/{total_files} files processed", flush=True)

            scenario = read_scenario(data_path, subset_list[cnt], file_name, self.compression_source)
            scenario_id = scenario['metadata']['scenario_id']
            #try:
            output = self.preprocess(scenario, split_name, starting_frame, tracks_to_predict)
            output = self.process(output)
                #output = self.postprocess(output)
            # except Exception as e:
            #     print('Error: {} in {}'.format(e, file_name), flush=True)
            #     output = None

            if output is None: continue
            
            relative_save_path = Path(f"{scenario_id}").with_suffix(self.compression_cache)
            with open(cache_path /relative_save_path , 'wb') as f:
                pickle.dump(output, f)

        return file_map
    
    
        
    def process(self, internal_format: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Transform data with internal_format in to waymo protobuf format
        '''
        wo_scenario = self.transform_track(internal_format)
        wo_scenario = self.transform_map(wo_scenario, internal_format)
        
        return wo_scenario
    
    
    def postprocess(self, output: Dict[str, Any]) -> Dict[str,Any]:
        return output

    
    def transform_track(self, internal_format):
        wo_scenario = Scenario()
        wo_scenario.scenario_id  = internal_format['scenario_id']
        wo_scenario.current_time_index = self.num_historical_steps - 1
        wo_scenario.sdc_track_index = internal_format['sdc_track_index']
        
        # tracks to predict
        for i_pred, track_index in enumerate(internal_format['tracks_to_predict']['track_index']):
            to_pred = RequiredPrediction()
            to_pred.track_index = track_index 
            #to_pred.difficulty = internal_format['tracks_to_predict']['track_difficulties'][i_pred]
            wo_scenario.tracks_to_predict.append(to_pred)
        
        # timestamps
        timestamps = np.array(internal_format['timestamps_seconds'])
        for t in (timestamps - timestamps[0]):
            wo_scenario.timestamps_seconds.append(t) # shape (T)
        
        tracks_to_predict = []
        wo_tracks=[]
        
        track_infos = internal_format['track_infos']
        
        for i_track, actor_id in enumerate(track_infos['object_id']):
            #actor_id = track_infos['actor_ids'][i_track]
            actor_type = track_infos["object_type"][i_track]
            #actor_difficulty = track_infos["track_difficulties"][i_track]
            actor_traj = track_infos["trajs"][i_track]
            actor_steps =  np.where(np.array([actor_traj[t, 9] for t in range(self.num_steps)]))[0] 
            
            wo_track = Track()
            
            
            wo_track.id = safe_int_strict(actor_id)
            
            wo_track.object_type = self.agent_type_converter(actor_type)
        
            for state in actor_traj:
                wo_state = ObjectState()
                wo_state.center_x = state[0]
                wo_state.center_y = state[1]
                wo_state.center_z = state[2]
                
                wo_state.length = state[3]
                wo_state.width =  state[4]
                wo_state.height =  state[5]
                
                wo_state.heading = state[6]

                wo_state.velocity_x = state[7]
                wo_state.velocity_y = state[8]

                wo_state.valid = state[9].astype(bool)
                
                wo_track.states.append(wo_state)
            
            wo_scenario.tracks.append(wo_track)
        
        return wo_scenario


    
    def transform_map(self, wo_scenario, internal_format):
        '''
        Only consider the road edges (boundaries) and road lines for Sim Agents
        '''
        map_infos = internal_format['map_infos']
        
        ### road edges ###
        
        for road_edge in map_infos['road_edge']:
            wo_road_edge = RoadEdge()
            polyline = map_infos['all_polylines'][road_edge['polyline_index'][0] : road_edge['polyline_index'][1]] # [x,y,z, dx, dy, dz, type]
            pts = polyline[:, :3] # [num_points, 2]
            if (pts.shape[0] < 2): # too less valid points
                continue
            
            polyline_type_ = polyline[0, -1]

            for point in pts:
                wo_b_pt = MapPoint()
                wo_b_pt.x = point[0]
                wo_b_pt.y = point[1]
                wo_b_pt.z = point[2]
                wo_road_edge.polyline.append(wo_b_pt)
                
            wo_map_feature_road_edge = MapFeature(id= int(road_edge['id']), road_edge= wo_road_edge)
            wo_scenario.map_features.append(wo_map_feature_road_edge)
        
        
        for road_line in map_infos['road_line']:
            wo_road_line = RoadLine()
            polyline = map_infos['all_polylines'][road_line['polyline_index'][0] : road_line['polyline_index'][1]] # [x,y,z, dx, dy, dz, type]
            
            pts = polyline[:, :3] # [num_points, 2]
            if (pts.shape[0] < 2): # too less valid points
                continue
            
            polyline_type_ = polyline[0, -1]
            
            
            for point in pts:
                wo_b_pt = MapPoint()
                wo_b_pt.x = point[0]
                wo_b_pt.y = point[1]
                wo_b_pt.z = point[2]
                wo_road_line.polyline.append(wo_b_pt)
                
            wo_map_feature_road_line = MapFeature(id= int(road_line['id']), road_line= wo_road_line)
            wo_scenario.map_features.append(wo_map_feature_road_line)
        
        
        return wo_scenario
            
        
    def agent_type_converter(self, scenarionet_type):
        if scenarionet_type == 0: # UNSET
            return Track.ObjectType.TYPE_UNSET
        elif scenarionet_type == 1: # VEHICLE
            return Track.ObjectType.TYPE_VEHICLE 
        elif scenarionet_type == 2: # PEDESTRIAN
            return Track.ObjectType.TYPE_PEDESTRIAN
        elif scenarionet_type == 3: # CYCLIST
            return Track.ObjectType.TYPE_CYCLIST
        elif scenarionet_type == 4: # OTHER
            return Track.ObjectType.TYPE_OTHER 
    
    ### the following functions should not be called ########
    def __getitem__(self, idx):
        pass
    
    
    def collate_fn(self, data_list: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        pass
