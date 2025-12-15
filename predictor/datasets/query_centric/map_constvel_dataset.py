"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

from collections import defaultdict, OrderedDict
from omegaconf import DictConfig


from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
import torch
import pandas as pd
from shapely import LineString, Point, Polygon
import compress_pickle as pickle
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
from torch_geometric.data import HeteroData, Batch

from predictor.datasets.base_dataset import BaseDataset, plot_processed_data, plot_tokenized_data, plot_multiple_cur_and_split
from predictor.datasets.types import object_type, polyline_type, track_type
from predictor.datasets.query_centric.utils import _interpolating_polyline
from predictor.utils.visualization import check_loaded_data

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

class MapConstVelDataset(BaseDataset):

    def __init__(self, config: DictConfig, data_splits: List[Dict]):
        super().__init__(config, data_splits)
        # self.tokenizer = TokenProcessor(**config.model.token_processor)
    
    
    def process_single_scenario_file(self, file_path, starting_frame, tracks_to_predict, inserted_static_obstacles, whitelist_agent_ids: Optional[List[str]]=None):
        scenario = sd_utils.read_scenario_data(file_path)
        scenario = self.process_single_scenario(scenario, starting_frame, tracks_to_predict, inserted_static_obstacles, whitelist_agent_ids)
        return scenario
    
    def process_single_scenario(self, scenario: Dict, starting_frame: int, tracks_to_predict: str, inserted_static_obstacles: Optional[Dict[str, Any]]=None, whitelist_agent_ids: Optional[List[str]]=None) -> Dict:
        scenario = self.postprocess(self.process(self.preprocess(scenario, 
                                                                split_name="nuplan_sim", 
                                                                starting_frame=starting_frame, 
                                                                tracks_to_predict = tracks_to_predict,
                                                                obstacles = inserted_static_obstacles
                                                                )))[0]
        if whitelist_agent_ids:
            scenario["agent"] = self.filter_agents_by_id(scenario["agent"], whitelist_agent_ids)
        scenario = HeteroData(scenario)
        return scenario
    
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

    def process(self, internal_format: Dict[str, Any]) -> Dict[str, Any]:
        dataset_split = internal_format["dataset_split"]
        map_infos = internal_format["map_infos"]
        track_infos = internal_format["track_infos"]
        scenario_id = internal_format["scenario_id"]
        dynamic_map_infos = internal_format["dynamic_map_infos"]
        sdc_track_index = internal_format["sdc_track_index"]
        current_time_index = internal_format["current_time_index"]
        tracks_to_predict = internal_format["tracks_to_predict"]

        if len(tracks_to_predict["track_index"]) < 1:
            return None
        
        #tf_lights = self._process_dynamic_map(dynamic_map_infos) # for now, we ignore traffic light states
        tf_lights = pd.DataFrame(columns=["lane_id", "time_step", "state"])
        tf_current_light = tf_lights.loc[tf_lights["time_step"] == internal_format["current_time_index"]] 
        agent_data = self._get_agent_data(track_infos, tracks_to_predict, sdc_track_index, current_time_index) 
        center_position = agent_data["position"][sdc_track_index,current_time_index,:2]
        data = {}
        data["scenario_id"] = scenario_id
        data["dataset_split"] = dataset_split
        data["ego_track_difficulty"] = internal_format["tracks_to_predict"]["ego_track_difficulty"][0]
        data["ego_track_type"] = internal_format["tracks_to_predict"]["ego_track_type"][0]
        data["agent"] = agent_data
        data['map'] = self._get_map_data(map_infos, center_position, map_range=self.config["map_range"])

        return [data]
    
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


    def _get_map_data(self, map_data: Dict[str, Any], center: Tuple[float, float], map_range: float,) -> Dict[str, Any]:
        lane_center_pts = []
        lane_mask = []
        lane_relationships = []
        
        all_lane_ids = np.array([lane['id'] for lane in map_data['lane']])
        lane_relationships = torch.zeros((len(all_lane_ids), len(all_lane_ids)), dtype=torch.long)
        for i, lane in enumerate(map_data['lane']):
            polyline = map_data['all_polylines'][lane['polyline_index'][0] : lane['polyline_index'][1]] # [x,y,z, dx, dy, dz, type]
            pts = polyline[:, :2] # [num_points, 2]
            
            mapel = Point(pts) if pts.shape[0] <= 1 else LineString(pts)
            
            if (pts.shape[0] < 2): # too less valid points
                lane_mask.append(False)
                pts = np.concatenate([pts, pts], axis=0)
            elif self.is_single_point_lane([pts])[0]: # too less valid points
                lane_mask.append(False)
            elif mapel.distance(Point(center))>map_range: # too far
                lane_mask.append(False)
            else:
                lane_mask.append(True)
            #elif lane_ls
                
            entry_lanes_idx = np.where(np.isin(all_lane_ids, lane['entry_lanes']))[0]
            exit_lanes_idx = np.where(np.isin(all_lane_ids, lane['exit_lanes']))[0] 
            left_neighbors_idx = np.where(np.isin(all_lane_ids, lane['left_neighbors']))[0] 
            right_neighbors_idx = np.where(np.isin(all_lane_ids, lane['right_neighbors']))[0]  
            
            lane_relationships[i, entry_lanes_idx]= 1
            lane_relationships[i, exit_lanes_idx]= 2
            lane_relationships[i, left_neighbors_idx]= 3
            lane_relationships[i, right_neighbors_idx]= 4
            
            
            #resampled_pts = self.resample_line(self.clean_lines(pts)[0])
            lane_center_pts.append(pts)
            
            
        
        lane_center_pts = [self.resample_line(self.clean_lines(lane_center)[0]) for lane_center, valid in zip( lane_center_pts, lane_mask) if valid]
        lane_center_pts = torch.tensor( lane_center_pts, dtype=torch.float32)
        
        map_data = {
            'lane_id': all_lane_ids[lane_mask],
            'center_pts': lane_center_pts,
            'lane_relationships': lane_relationships[lane_mask, :][:, lane_mask].tolist(),
            'num_nodes': len(all_lane_ids[lane_mask]),
            }                
        
        return map_data

    def _get_agent_data(self, track_infos, tracks_to_predict, sdc_track_index, current_time_index, dim=3):
        # same implementation in query_centric_dataset.py
        """
        Process raw tracking data into per-agent tensors and lists.

        We already filter invalid agents out during preprocessing

        tracks_to_predict.keys()
        dict_keys(['track_index', 'track_difficulties', 'track_types', 'object_type', 'track_id', 'ego_track_difficulty', 'ego_track_type'])
        """
        # Unpack raw data from track_info
        trajs = track_infos["trajs"]  # shape: (num_agents, num_steps, 10)
        object_type = np.array(track_infos["object_type"])
        other_type = object_type == 4
        to_vehicle = np.prod(trajs[:,current_time_index,3:5],axis=1) > 1.5
        to_ped = ~to_vehicle
        object_type[other_type & to_ped] = 2
        object_type[other_type & to_vehicle] = 1

        num_agents = trajs.shape[0]
        num_steps = trajs.shape[1]
        out_dict = {
            "num_nodes": num_agents,
            "valid_mask": torch.zeros([num_agents, num_steps], dtype=torch.bool),
            "train_mask": torch.zeros([num_agents], dtype=torch.bool),
            "role": torch.zeros([num_agents, 3], dtype=torch.bool),
            "id": torch.zeros(num_agents, dtype=torch.int64) - 1,
            "type": torch.zeros(num_agents, dtype=torch.uint8),
            "position": torch.zeros([num_agents, num_steps, 3], dtype=torch.float32),
            "heading": torch.zeros([num_agents, num_steps], dtype=torch.float32),
            "velocity": torch.zeros([num_agents, num_steps, 2], dtype=torch.float32),
            "shape": torch.zeros([num_agents, 3], dtype=torch.float32),
            'track_difficulty': torch.zeros([num_agents], dtype=torch.float32),
            'track_type': torch.zeros([num_agents], dtype=torch.uint8),
        }
        out_dict["role"][sdc_track_index] = torch.tensor([True, False, False])
        out_dict["type"] = torch.tensor(object_type, dtype=torch.uint8)
        # Get the per-agent metadata.
        out_dict["id"] = track_infos["object_id"]
        out_dict["track_difficulty"] = torch.tensor(track_infos["track_difficulties"], dtype=torch.float)
        out_dict["track_type"] = torch.tensor(track_infos["track_types"], dtype=torch.uint8)
        # Define state vectors
        out_dict["position"] = torch.tensor(trajs[:,:,:dim], dtype=torch.float)
        out_dict["heading"] = torch.tensor(trajs[:,:,6], dtype=torch.float)
        out_dict["velocity"] =  torch.tensor(trajs[:,:,7:9], dtype=torch.float)
        out_dict["shape"] =  torch.tensor(trajs[:,current_time_index,3:6], dtype=torch.float)
        # Define masks
        out_dict["valid_mask"] = torch.tensor(trajs[:,:,9], dtype=torch.bool) # TODO: for vecto_repr
        # which trajectories to train and validate on
        out_dict["train_mask"][tracks_to_predict["track_index"]] = True
        #out_dict["train_mask"][sdc_track_index] = True # should be included in tracks_to_predict anyways
        #print("TRAIN ON ", sum(out_dict["train_mask"]), " OF ", num_agents)
        return out_dict
    
    def postprocess(self, output: Dict[str, Any]) -> Dict[str,Any]:
        #tokenized_map, tokenized_agent = self.tokenizer(Batch.from_data_list([HeteroData(output[0])]))
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
        return Batch.from_data_list(data_list)
    
    
    def resample_line(self, line: np.ndarray, num_sample_point = 20):
        '''
        resample (interpolate) line with equal distance.

        parameter:
            - line: [N, 2]

        return:
            - resampled (interpolated) line [M,2]
        '''

        ls = LineString(line)
        s0 = 0
        s1 = ls.length

        return np.array([
            ls.interpolate(s).coords.xy
            for s in np.linspace(s0, s1, num_sample_point)
        ]).squeeze()


    def clean_lines(self, lines):
        '''
        clean line points, which go backwards.

        parameter:
            - lines: list of lines with shape [N, 2]

        return:
            - cleaned list of lines with shape [M, 2]
        '''
        cleaned_lines = []
        if not isinstance(lines, list):
            lines = [lines]
        for candidate in lines:
            # remove duplicate points
            ds = np.linalg.norm(np.diff(candidate, axis=0), axis=-1) > 0.05
            keep = np.block([True, ds])

            cleaned = candidate[keep, :]

            # remove points going backward
            if cleaned.shape[0] > 1:
                dx, dy = np.diff(cleaned, axis=0).T
                dphi = np.diff(np.unwrap(np.arctan2(dy, dx)))

                keep = np.block([True, dphi < (np.pi / 2), True])

                cleaned = cleaned[keep, :]

            cleaned_lines.append(cleaned)

        return cleaned_lines
    
    def is_single_point_lane(self, lines):
        cleaned_lines = self.clean_lines(lines)
        num_points = np.array([l.shape[0] for l in cleaned_lines])
        return num_points<=1

