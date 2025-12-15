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
from pathlib import Path

from predictor.datasets.base_dataset import BaseDataset, plot_processed_data, plot_tokenized_data, plot_multiple_cur_and_split
from predictor.datasets.types import object_type, polyline_type, track_type
from predictor.utils.visualization import check_loaded_data
#from predictor.datasets import build_transform
from predictor.datasets.transforms.ep_target_builder import EPTargetBuilder

from predictor.utils.ep_utils.preprocess_utils import map_utils, tracking_utils
from predictor.utils.ep_utils.preprocess_utils.tracker import Polynomial_Tracker


default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)

_polygon_light_type = [
    "NO_LANE_STATE",
    "LANE_STATE_UNKNOWN",
    "LANE_STATE_STOP",
    "LANE_STATE_GO",
    "LANE_STATE_CAUTION",
]

def to_float(inputs):
    for i, v in enumerate(inputs):
        inputs[i] = v.astype(np.float32)
        
    return inputs

class EPDataset(BaseDataset):

    def __init__(self, config: DictConfig, data_splits: List[Dict]):
        self.track_deg = config.data.track_deg
        self.fut_deg = config.data.fut_deg
        self.mapel_deg = config.data.mapel_deg
        self.num_historical_steps = config.data.past_len
        self.num_future_steps = config.data.future_len
        self.num_steps = self.num_historical_steps + self.num_future_steps
        self.hist_timescale = (self.num_historical_steps-1) / 10
        self.prior_loaded = False
        self.loaded_dataset_name= None
        self.global_map = {}
        # with open("/home/ec2-user/works/data/caches/cache_smart/waymo/global_map.pkl", "rb") as f:
        #     self.global_map = pickle.load(f)
        
        self.tracker = Polynomial_Tracker(timescale=self.hist_timescale, 
                                  degree=self.track_deg, 
                                  space_dim=2, 
                                  hist_len = self.num_historical_steps)
        self.transform = EPTargetBuilder(config)
        super().__init__(config, data_splits)

    def _build_sample_idx(self, file_map: Dict, filter: Dict) -> List:
        selected_ego_track_types = set(track_type[t] for t in filter["ego_track_types"])

        def is_whitelisted(sid, whitelist): return (not whitelist) or any(w in sid for w in whitelist)
        # filter samples
        sample_index = [
            (file, idx) 
            for file, file_info in file_map.items() 
            for idx, sid in enumerate(file_info["scenario_id"])
            if (sid not in self.config.blacklist)
                and is_whitelisted(sid, self.config.whitelist)
                and (filter["ego_track_difficulty"][0] <= file_info["ego_track_difficulty"][idx] < filter["ego_track_difficulty"][1])
                and (file_info["ego_track_type"][idx] in selected_ego_track_types)
        ]

        # subsample data
        num_samples = len(sample_index)
        data_usage = filter["sample_num"]
        if data_usage >= num_samples: #as number
            data_usage = num_samples
        elif data_usage > 1.0: # as number
            data_usage = data_usage
        elif 1.0 >= data_usage > 0.0: # as percentage
            data_usage = data_usage * num_samples  # guarantees multiples of 8
        #sample_index = random.sample(sample_index, int(data_usage))
        sample_index = sample_index[:int(data_usage)]
        rank_zero_info(f"Subsampled {len(sample_index)} from {num_samples} selected samples")
        return sample_index
    
    
    def cache_data(self) -> None:
        """
        Cache all data splits
        """
        for split in self.data_splits:
            if split["cache_path"].exists() and not self.config.get("overwrite_cache", False):
                rank_zero_info(f"Already existing cache {split['cache_path']} will be used")
                continue
            else:
                self._cache_datasplit(split["split"], split["data_path"], split["cache_path"], split["starting_frame"], split["tracks_to_predict"])
        
        # print('saving map')
        # map_saving_path = Path(split["cache_path"]).parents[1]
        # with Path(map_saving_path, "global_map.pkl").with_suffix(self.compression_cache).open("wb") as f:
        #     pickle.dump(self.global_map, f)
        
        
    def process(self, internal_format: Dict[str, Any]) -> Dict[str, Any]:
        dataset_split = internal_format["dataset_split"]
        map_infos = internal_format["map_infos"]
        track_infos = internal_format["track_infos"]
        scenario_id = internal_format["scenario_id"]
        dynamic_map_infos = internal_format["dynamic_map_infos"]
        sdc_track_index = internal_format["sdc_track_index"]
        current_time_index = internal_format["current_time_index"]
        tracks_to_predict = internal_format["tracks_to_predict"]
        dataset_name = internal_format["dataset_name"]
        meta_data = internal_format["metadata"]
        
        if not (self.prior_loaded and dataset_name == self.loaded_dataset_name):
            self.load_priors(dataset_name)
        
        # initialize
        output_data = {'scenario_id': scenario_id,
                       'track_deg': self.track_deg,
                       'mapel_deg': self.mapel_deg,
                       'fut_deg': self.fut_deg}
        
        # unitraj related info
        output_data["dataset_split"] = dataset_split
        output_data["ego_track_difficulty"] = internal_format["tracks_to_predict"]["ego_track_difficulty"][0]
        output_data["ego_track_type"] = internal_format["tracks_to_predict"]["ego_track_type"][0]
        
        # agent processed info
        output_data['agent'], global_origin, global_R = self.get_agent_data(track_infos, tracks_to_predict, sdc_track_index, meta_data)
        output_data['global_origin'] = global_origin
        output_data['global_R'] = global_R
        
        # map processed info
        output_data['map'] = self.get_map_data(map_infos, origin=global_origin, R=global_R)
        return [output_data]
    
    def get_agent_data(self, track_infos, tracks_to_predict, sdc_track_index, meta_data):
        data_list = [{
                     'track_id': track_id,
                     'object_type': object_type,
                     'length': track_value[timestep, 3] if timestep < self.num_steps else 0.,
                     'width': track_value[timestep, 4] if timestep < self.num_steps else 0.,
                     'valid': track_value[timestep, 9] if timestep < self.num_steps else 0.,
                     'position_x': track_value[timestep, 0] if timestep < self.num_steps else 0.,
                     'position_y': track_value[timestep, 1] if timestep < self.num_steps else 0.,
                     'position_z': track_value[timestep, 2] if timestep < self.num_steps else 0.,
                     'heading': track_value[timestep, 6] if timestep < self.num_steps else 0.,
                     'velocity_x': track_value[timestep, 7] if timestep < self.num_steps else 0.,
                     'velocity_y': track_value[timestep, 8] if timestep < self.num_steps else 0.,
                     'timestep': timestep,
                     }
                    for (track_id, object_type, track_value) in zip(track_infos['object_id'], track_infos['object_type'], track_infos['trajs']) for timestep in range(self.num_steps)]
        
        df = pd.DataFrame(data_list)
        
        historical_df = df[df['timestep'] < self.num_historical_steps]
        timesteps = list(np.sort(df['timestep'].unique()))
        timestamps = meta_data['timestamps_seconds']
        
        actor_ids = list(historical_df['track_id'].unique())

        #actor_ids = list(filter(lambda actor_id: np.sum(historical_df[historical_df['track_id'] == actor_id]['valid'])>=1, actor_ids))

        historical_df = historical_df[historical_df['track_id'].isin(actor_ids)]
        df = df[df['track_id'].isin(actor_ids)]
        
        av_id = track_infos['object_id'][sdc_track_index]
        to_predict_agt_id = [track_infos['object_id'][target_track_index] for target_track_index in tracks_to_predict['track_index']]
        
        # save the orginal designed target agent
        to_predict_agt_id_origin = []
        if meta_data.get('tracks_to_predict', None) is None:
            to_predict_agt_id_origin=[av_id]
        else:
            to_predict_agt_id_origin = list(meta_data['tracks_to_predict'].keys())
            if len(to_predict_agt_id_origin) == 0:
                to_predict_agt_id_origin=[av_id]
        
        # DataFrame for AV and Agent
        av_df = df[df['track_id'] == av_id].iloc
        av_index = actor_ids.index(av_df[0]['track_id'])
        agent_index = [actor_ids.index(target_id) for target_id in to_predict_agt_id]
        agent_index_origin = [actor_ids.index(target_id) for target_id in to_predict_agt_id_origin]
        
        num_actors = len(actor_ids)
        timestep_mask = np.zeros((num_actors, self.num_steps), dtype=bool) # booleans indicate if object is observed at each timestamp
        time_window = np.zeros((num_actors, 2), dtype=float) # start and end timestamps for the control points
        objects_size = np.zeros((num_actors, 2), dtype=float) # agent length and width
        objects_type = np.zeros((num_actors), dtype=int)
        tracks_category = np.zeros((num_actors), dtype=int)
        tracks_category_origin = np.zeros((num_actors), dtype=int)
        x = np.zeros((num_actors, self.num_steps, 6), dtype=float) # [x, y, heading, vx, vy, z]
        x_origin = np.zeros((num_actors, self.num_steps, 5), dtype=float) # [x, y, heading, vx, vy]
        x_mean = np.zeros((num_actors, (self.track_deg+1) * 2), dtype=float) 
        x_mean_fut = np.zeros((num_actors, (self.fut_deg+1)* 2), dtype=float) 
        x_cov = np.zeros((num_actors, (self.track_deg+1) * 2, (self.track_deg+1) * 2), dtype=float)
        agent_id = [None] * num_actors
        
        # make the scene centered at AV
        origin = np.array([av_df[self.num_historical_steps-1]['position_x'], av_df[self.num_historical_steps-1]['position_y']])
        theta = np.array(av_df[self.num_historical_steps-1]['heading'])
        rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
        R_mat = np.kron(np.eye(self.track_deg+1), rotate_mat)
        
        ego_positions = np.array([av_df[:]['position_x'].values, av_df[:]['position_y'].values]).T
        ego_headings = np.array(av_df[:]['heading'].values)
        ego_velocities = np.array([av_df[:]['velocity_x'].values, av_df[:]['velocity_y'].values]).T

        ego_traj = np.concatenate([ego_positions, ego_headings[:, None], ego_velocities], axis=1) # This is raw data

        obj_trajs = [None] * num_actors
        priors, history_priors = [None] * num_actors, np.zeros((num_actors, (self.track_deg+1)*2, (self.track_deg+1)*2))
        
        for actor_id, actor_df in df.groupby('track_id'):
            actor_idx = actor_ids.index(actor_id)
            agent_id[actor_idx] = actor_id
            actor_time_mask = np.where(actor_df['valid'].to_numpy(), True, False)  # [num_steps]

            actor_steps = np.where(actor_time_mask==True)[0]
            actor_hist_steps = np.where(actor_time_mask[:self.num_historical_steps]==True)[0]
            actor_fut_steps = np.where(actor_time_mask[(self.num_historical_steps-1):]==True)[0] + self.num_historical_steps -1 
            timestep_mask[actor_idx, actor_time_mask] = True
            fut_timestep_mask = timestep_mask[actor_idx, self.num_historical_steps-1:]

            objects_type[actor_idx] = actor_df['object_type'].unique()[0]
            objects_size[actor_idx] = (np.vstack([actor_df[:]['length'].values, actor_df[:]['width'].values]).T)[actor_hist_steps][-1] # first valid size in history
            tracks_category[actor_idx] = 3 if actor_id in to_predict_agt_id else 1
            tracks_category_origin[actor_idx] = 3 if actor_id in to_predict_agt_id_origin else 1
            
            positions = np.array([actor_df[:]['position_x'].values, actor_df[:]['position_y'].values]).T
            headings = np.array(actor_df[:]['heading'].values)
            velocities = np.array([actor_df['velocity_x'].values, actor_df['velocity_y'].values]).T
            positions_z = np.array(actor_df[:]['position_z'].values)
            
            positions = positions[actor_steps]
            headings = headings[actor_steps]
            velocities = velocities[actor_steps]
            positions_z = positions_z[actor_steps]
            
            
            obj_traj = np.concatenate([positions, headings[:, None], velocities], axis=1) # This is raw data
            obj_trajs[actor_idx] = obj_traj

            x_origin[actor_idx, actor_steps, :2] = positions
            x_origin[actor_idx, actor_steps, 2] = headings
            x_origin[actor_idx, actor_steps, 3:5] = velocities
            
            x[actor_idx, actor_steps, :2] = (positions - origin) @ rotate_mat
            x[actor_idx, actor_steps, 2] = headings - theta
            x[actor_idx, actor_steps, 3:5] = velocities @ rotate_mat
            x[actor_idx, actor_steps, 5] = positions_z
            
            T_hist = timestamps[actor_hist_steps]
            T_fut = timestamps[actor_fut_steps]

            time_window[actor_idx] = np.array([np.min(T_hist), np.max(T_hist)])
            
            
            if actor_id == av_id:
                priors[actor_idx] = self.prior_ego
                history_priors[actor_idx] = self.hist_prior_ego         
                
                cps_mean_fut, cps_cov_fut = tracking_utils.bayesian_regression_ego(trajectory = obj_traj[np.where(np.array(actor_steps) >= self.num_historical_steps-1)], 
                                                                                   timestamps = T_fut-timestamps[self.num_historical_steps-1],
                                                                                   prior_data = self.prior_ego_8s,
                                                                                   timescale = T_fut[-1] - T_hist[-1],
                                                                                   degree = self.fut_deg)
                
                x_mean_fut[actor_idx] = np.reshape(cps_mean_fut, (-1))

            else:
                prior_data, hist_prior_data = None, None
                object_type = objects_type[actor_idx]
                if object_type == 1: # vehicle
                    prior_data, hist_prior_data = self.prior_vehicle, self.hist_prior_vehicle
                    prior_data_fut = self.prior_vehicle_8s
                elif object_type == 2: # pedestrian
                    prior_data, hist_prior_data = self.prior_pedestrian, self.hist_prior_pedestrian
                    prior_data_fut = self.prior_pedestrian_8s
                elif object_type == 3: # cyclist
                    prior_data, hist_prior_data = self.prior_cyclist, self.hist_prior_cyclist
                    prior_data_fut = self.prior_cyclist_8s
                else: # TODO: what is the prior for unknown objects?
                    prior_data, hist_prior_data = self.prior_vehicle, self.hist_prior_vehicle
                    prior_data_fut = self.prior_vehicle_8s
                
                if np.any(fut_timestep_mask):
                    cps_mean_fut, cps_cov_fut = tracking_utils.bayesian_regression_agt(obj_traj[np.where(np.array(actor_steps) >= self.num_historical_steps-1)], 
                                                                                       ego_traj[actor_fut_steps],  
                                                                                       timestamps = T_fut-timestamps[self.num_historical_steps-1], 
                                                                                       prior_data = prior_data_fut,
                                                                                       timescale = T_fut[-1] - T_hist[-1],
                                                                                       degree = self.fut_deg)
                    
                    x_mean_fut[actor_idx] = np.reshape(cps_mean_fut, (-1))
            
                priors[actor_idx] = prior_data
                history_priors[actor_idx] = hist_prior_data
            
        ## Start Tracking ##
        self.tracker.track(x=x_origin, 
                          timestep_mask=timestep_mask, 
                          timestamps=timestamps, 
                          time_window=time_window, 
                          priors=priors, 
                          hist_priors=history_priors, 
                          av_index=av_index, 
                          agent_index = None,
                          x_mean=x_mean, 
                          x_cov=x_cov)
        
        x_mean = (np.reshape(x_mean, (x_mean.shape[0], self.track_deg+1, 2)) - origin) @ rotate_mat
        x_cov = R_mat.T @ x_cov @ R_mat
        
        x_mean_fut = (np.reshape(x_mean_fut, (x_mean_fut.shape[0], self.fut_deg+1, 2)) - origin) @ rotate_mat
        
            
        track_data = {
            'object_type': torch.tensor(objects_type, dtype=torch.long), # [A]
            'object_size': torch.tensor(objects_size, dtype=torch.float32), # [A]
            'track_category': torch.tensor(tracks_category, dtype=torch.long), # [A]
            'track_category_origin': torch.tensor(tracks_category_origin, dtype=torch.long), # [A]
            'timestamps_seconds': torch.tensor(timestamps, dtype=torch.float32), # [num_steps]
            'x': torch.tensor(x[:, :self.num_historical_steps], dtype=torch.float32), # [N, 50, 6]
            'y': torch.tensor(x[:, self.num_historical_steps:], dtype=torch.float32), # [A, num_future_steps, 6]
            'cps_mean': torch.tensor(x_mean, dtype=torch.float32), # [N, 4, 2]
            'cps_mean_fut': torch.tensor(x_mean_fut, dtype=torch.float32), # [N, 4, 2]
            'cps_cov': torch.tensor(x_cov, dtype=torch.float32), # [N, 8, 8]
            'timestep_x_mask': torch.tensor(timestep_mask[:, :self.num_historical_steps], dtype=bool), #[N, 50]
            'timestep_y_mask': torch.tensor(timestep_mask[:, self.num_historical_steps:], dtype=bool), #[N, 60]
            'time_window': torch.tensor(time_window - timestamps[0], dtype=torch.float32), # [N, 2]
            'av_index': torch.tensor(av_index, dtype=torch.long),
            'agent_index': torch.tensor(agent_index, dtype=torch.long),
            'agent_index_origin': torch.tensor(agent_index_origin, dtype=torch.long),
            'agent_ids': agent_id, # [N]
            'num_nodes': len(actor_ids),
           }
        
        return track_data, origin, rotate_mat
        
    def get_map_data(self, map_infos, origin = None, R = None, radius = 300):
         # initialization
        lane_cps_list = [] #np.zeros((num_lanes, 3, 4, 2), dtype=float)
        lane_type_list = [] #np.zeros((num_lanes), dtype=np.uint8)        

        cw_cps_list = [] #np.zeros((num_cross_walks, 3, self.mapel_deg+1, 2), dtype=np.float32)
        cw_type_list = [] #np.zeros((num_cross_walks), dtype=np.uint8) 
        
        # lane
        for lane in map_infos['lane']:
            polyline = map_infos['all_polylines'][lane['polyline_index'][0] : lane['polyline_index'][1]] # [x,y,z, dx, dy, dz, type]
            polyline_type_ = polyline[0, -1]
            

            pts = polyline[:, :2] # [num_points, 2]
                
            if (pts.shape[0] < 2) or map_utils.is_single_point_lane([pts])[0]: # too less valid points
                continue
            
            if self.loaded_dataset_name == 'nuscenes':
                dist = np.linalg.norm(origin - pts[pts.shape[0]//2])
                if dist > radius:
                    continue
                
            rounded_pts = np.round(pts, decimals=2)
            hash_id = hashlib.md5(rounded_pts.tobytes()).hexdigest()
                
            try:
                cps_list = []
                if hash_id in self.global_map.keys(): # already processed
                    cps_list = self.global_map[hash_id]
                else:
                    map_utils.recurrent_fit_line(pts=pts, cps_list=cps_list, degree=self.mapel_deg)
                    self.global_map[hash_id] = deepcopy(to_float(cps_list))

                lane_cps_list = lane_cps_list + cps_list
                type_list = [polyline_type_ for _ in range(len(cps_list))]
                lane_type_list = lane_type_list + type_list
                
            except:               
                if hash_id in self.global_map.keys(): # already processed
                    cps_list = self.global_map[hash_id]
                else:
                    cps_list = [map_utils.fit_line(pts, degree = self.mapel_deg, no_clean = True, use_borgespastva=True)]
                    self.global_map[hash_id] = deepcopy(to_float(cps_list))

                lane_cps_list = lane_cps_list+ cps_list
                lane_type_list.append(polyline_type_)
        
        # crosswalk        
        for cw in map_infos['crosswalk']:
            polyline = map_infos['all_polylines'][cw['polyline_index'][0] : cw['polyline_index'][1]] # [x,y,z, dx, dy, dz, type]
            polyline_type_ = polyline[0, -1]
            edge_1, edge_2 = map_utils.find_edges(polyline[:, :2])
            center = (edge_1 + edge_2)/2.
            
            if self.loaded_dataset_name == 'nuscenes':
                dist = np.linalg.norm(origin - center[center.shape[0]//2])
                if dist > radius:
                    continue

            cw_cps = map_utils.fit_line(center, degree = self.mapel_deg, use_borgespastva = False, num_sample_point=4)
            cw_cps_reverse = cw_cps[::-1]
            cw_cps_list.append(cw_cps)
            cw_cps_list.append(cw_cps_reverse)

            cw_type_list.append(polyline_type_)
            cw_type_list.append(polyline_type_)

                          

        num_lanes = len(lane_cps_list)
        num_cws = len(cw_cps_list)

        lane_segment_ids = torch.zeros(num_lanes, dtype=torch.long)
        lane_cps_list = np.array(lane_cps_list)
        
        if origin is not None and R is not None:
            lane_cps_list = (lane_cps_list - origin) @ R 
        
        lane_cps_list = torch.tensor(lane_cps_list, dtype=torch.float32)
        lane_type_list = torch.tensor(np.array(lane_type_list), dtype=torch.long)


        cross_walk_ids = torch.zeros(num_cws, dtype=torch.long)
        cw_cps_list = np.array(cw_cps_list)
        if num_cws >0 and origin is not None and R is not None:
            cw_cps_list = (cw_cps_list-origin)@R
        
        cw_cps_list = torch.tensor(cw_cps_list, dtype=torch.float32)
        cw_type_list = torch.tensor(np.array(cw_type_list), dtype=torch.long)

        mapel_ids = torch.concat((lane_segment_ids, cross_walk_ids), dim=0)
        mapel_cps = torch.concat((lane_cps_list, cw_cps_list), dim=0)
        mapel_types = torch.concat((lane_type_list, cw_type_list), dim=0)
        num_mapels = num_lanes + num_cws

        map_data = {
                #'mapel_ids': mapel_ids,
                'mapel_cps': mapel_cps,
                'mapel_types': mapel_types,
                'num_nodes': num_mapels,
                }                

        return map_data


    
    
    def _process_dynamic_map(self, dynamic_map_infos: Dict[str,Any]) -> pd.DataFrame:
        """
        Convert dynamic map info into a DataFrame of traffic light states.
        """
        lane_ids = dynamic_map_infos["lane_id"]

        # If no lane_ids are provided, return an empty DataFrame with the expected columns
        if len(lane_ids) == 0:
            return pd.DataFrame(columns=["lane_id", "time_step", "state"])
        
        tf_lights = []
        # TODO: Bugfix caused by string lane IDs
         # For each time step, combine lane IDs, time, and state.
        for t in range(len(lane_ids)):
            lane_id = lane_ids[t]
            time = np.ones_like(lane_id) * t
            state = dynamic_map_infos["state"][t]
            tf_light = np.concatenate([lane_id, time, state], axis=0)
            tf_lights.append(tf_light)

        # Concatenate and reshape the data.
        tf_lights = np.concatenate(tf_lights, axis=1).transpose(1, 0)
        tf_lights = pd.DataFrame(data=tf_lights, columns=["lane_id", "time_step", "state"])

        # Convert columns to string and standardize state names.
        tf_lights["time_step"] = tf_lights["time_step"].astype("str")
        tf_lights["lane_id"] = tf_lights["lane_id"].astype("str")
        tf_lights["state"] = tf_lights["state"].astype("str")
        tf_lights.loc[tf_lights["state"].str.contains("STOP"), ["state"] ] = 'LANE_STATE_STOP'
        tf_lights.loc[tf_lights["state"].str.contains("GO"), ["state"] ] = 'LANE_STATE_GO'
        tf_lights.loc[tf_lights["state"].str.contains("CAUTION"), ["state"] ] = 'LANE_STATE_CAUTION'
        return tf_lights
    
    
    def postprocess(self, output: Dict[str, Any]) -> Dict[str,Any]:
        return output

    
    def __getitem__(self, idx):
        samples = []
        for file, indices in self.sample_index[idx]:
            with open(file, 'rb') as f:
                all_samples =  pickle.load(f)
            selected_samples = [HeteroData(all_samples[i]) for i in indices]
            samples.extend(selected_samples)
        return samples
    
    
    def collate_fn(self, data_list: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Collates a list of sample dictionaries into a single batch dictionary.
        """
        data_list = [sample for sublist in data_list for sample in sublist]
        if getattr(self, "transform", None) is not None:
            data_list = [self.transform(s) for s in data_list]
        return Batch.from_data_list(data_list)
    
    
    def load_priors(self, dataset):
        self.loaded_dataset_name = deepcopy(dataset)
        if dataset not in ['argoverse2', 'waymo']:
            warning_msg = 'Priors for ' + dataset + ' not available, use argoverse2 priors instead'
            warnings.warn(warning_msg)
            dataset = 'argoverse2'
            
            
        with open('models/ep/priors/' + dataset + '/vehicle/vehicle_1s.json', "r") as read_file:
            self.prior_vehicle = json.load(read_file)
            self.hist_prior_vehicle = tracking_utils.get_bernstein_prior(degree=self.track_deg, timescale=self.hist_timescale, prior_data=self.prior_vehicle)[1]

        with open('models/ep/priors/' + dataset + '/cyclist/cyclist_1s.json', "r") as read_file:
            self.prior_cyclist = json.load(read_file)
            self.hist_prior_cyclist = tracking_utils.get_bernstein_prior(degree=self.track_deg, timescale=self.hist_timescale, prior_data=self.prior_cyclist)[1]

        with open('models/ep/priors/' + dataset + '/pedestrian/pedestrian_1s.json', "r") as read_file:
            self.prior_pedestrian = json.load(read_file)
            self.hist_prior_pedestrian = tracking_utils.get_bernstein_prior(degree=self.track_deg, timescale=self.hist_timescale, prior_data=self.prior_pedestrian)[1]

        with open('models/ep/priors/' + dataset + '/ego/ego_1s.json', "r") as read_file:
            self.prior_ego = json.load(read_file)
            self.hist_prior_ego = tracking_utils.get_bernstein_prior(degree=self.track_deg, timescale=self.hist_timescale, prior_data=self.prior_ego)[1]
            
        # Load prior parameters for future trajectory
        with open('models/ep/priors/' + dataset + '/vehicle/vehicle_8s.json', "r") as read_file:
            self.prior_vehicle_8s = json.load(read_file)
 
        with open('models/ep/priors/' + dataset + '/cyclist/cyclist_8s.json', "r") as read_file:
            self.prior_cyclist_8s = json.load(read_file)

        with open('models/ep/priors/' + dataset + '/pedestrian/pedestrian_8s.json', "r") as read_file:
            self.prior_pedestrian_8s = json.load(read_file)

        with open('models/ep/priors/' + dataset + '/ego/ego_8s.json', "r") as read_file:
            self.prior_ego_8s = json.load(read_file)
            
        self.prior_loaded = True