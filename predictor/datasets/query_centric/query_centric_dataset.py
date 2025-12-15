"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

from collections import defaultdict, OrderedDict
from omegaconf import DictConfig


from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
import torch
import pandas as pd
import compress_pickle as pickle
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
from torch_geometric.data import HeteroData, Batch
from metadrive.scenario import utils as sd_utils
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

class QueryCentricBaseDataset(BaseDataset):

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
    
    def filter_agents_by_id(self, agent_data, whitelist_agent_ids):
        whitelist_indices = [agent_data["id"].index(oid) for oid in whitelist_agent_ids]
        filtered_agent_data = {}
        for k, v in agent_data.items():
            if k == "num_nodes":
                new_v = len(whitelist_indices)
            elif torch.is_tensor(v):
                new_v = v[whitelist_indices]
            else:
                new_v = v
            filtered_agent_data[k] = new_v
        return filtered_agent_data

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
        map_data = self._get_map_data(map_infos, tf_current_light, center=center_position, map_range=self.config["map_range"], polygon_types=self.config["line_type"])
        data = self._process_map(map_data)
        data["scenario_id"] = scenario_id
        data["dataset_split"] = dataset_split
        data["ego_track_difficulty"] = internal_format["tracks_to_predict"]["ego_track_difficulty"][0]
        data["ego_track_type"] = internal_format["tracks_to_predict"]["ego_track_type"][0]
        data["agent"] = agent_data
        # self.plot_map(map_data, internal_format["dataset_name"], agent_data["position"][sdc_track_index][agent_data["valid_mask"][sdc_track_index]])
        if self.plotting: 
            plot_processed_data(data, current_idx=current_time_index)
            from predictor.models.smart.tokens.tokenizer import TokenProcessor
            from easydict import EasyDict
            graph = HeteroData(data)
            graphs = Batch.from_data_list([graph])
            sampling = EasyDict({"num_k": 1, "temp": 1.0})
            tokenizer = TokenProcessor(map_token_file="map_traj_token5.pkl", agent_token_file="agent_vocab_555_s2.pkl", map_token_sampling=sampling, agent_token_sampling=sampling)
            tokens = tokenizer(graphs)
            plot_tokenized_data(tokens[0], tokens[1])

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

    def _get_map_data(self, map_infos: Dict[str, Any], tf_current_light: pd.DataFrame, center: Tuple[float, float], map_range: float, polygon_types: List[str], dim: int = 3) -> Dict[str, Any]:
        """
        Extract cropped 2D map features and point→polygon edges.

        Args:
            map_infos: Dict with keys in _polygon_types plus "all_polylines".
            tf_current_light: DataFrame of light states by lane_id.
            center: (cx, cy) center of cropping window.
            map_range: side-length of square crop.
            polyline_types: polyline_types selected
            dim: number of spatial dims (2 or 3).

        Returns:
            Slimmed map_data dict with
            - map_polygon: num_nodes, type, light_type
            - map_point:   num_nodes, position, type
            - (map_point→map_polygon): edge_index
        """
        # 1) Unpack & build a mask to filter out too distant points
        all_polylines = map_infos["all_polylines"]
        cx, cy = np.array(center)
        half = map_range / 2
        # boolean mask per point
        mask = (
            (all_polylines[:, 0] >= cx - half) &
            (all_polylines[:, 0] <= cx + half) &
            (all_polylines[:, 1] >= cy - half) &
            (all_polylines[:, 1] <= cy + half)
        )
        # 2) filter out any segments which dont have any valid point
        def _filter(segments):
            out = []
            for seg in segments:
                start, end = seg["polyline_index"]
                if mask[start:end].any():
                    out.append(seg)
            return out

        for key in polygon_types:
            map_infos[key] = _filter(map_infos[key])

        # 3) Build polygon_ids and init
        polygon_ids = [x["id"] for k in polygon_types for x in map_infos[k]]
        #polygon_types = [x["type"] for k in polygon_types for x in map_infos[k]]
        num_polygons = len(polygon_ids)
        polygon_type       = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_light_type = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[torch.Tensor] = [None] * num_polygons
        point_type:     List[torch.Tensor] = [None] * num_polygons
        # 4) Single pass over each type, filter out invalid points of polylines
        for key in polygon_types:
            for seg in map_infos[key]:
                idx = polygon_ids.index(seg["id"])

                # apply the mask slice to the polyline
                start, end = seg["polyline_index"]
                polyline = all_polylines[start:end]
                valid = mask[start:end]
                polyline = torch.from_numpy(polyline[valid]).float()

                # record polygon type & light
                polygon_type[idx] = polygon_types.index(key) 
                if key == "lane":
                    res = tf_current_light[tf_current_light["lane_id"] == seg["id"]]
                    if len(res):
                        polygon_light_type[idx] = _polygon_light_type.index(res["state"].item())

                # point features
                point_position[idx] = polyline[:, :dim] # cl[:-1, :dim]
                deltas = polyline[1:] - polyline[:-1]
                point_type[idx] = torch.full((len(deltas)+1,), polyline_type[seg["type"]], dtype=torch.uint8)

        # 5) Build point→polygon edge index
        ## NOTE: we replaced the torch.arange functions since those do their own threading which deadlocks multiprocessing
        num_points = torch.tensor([p.size(0) for p in point_position], dtype=torch.long)
        num_points_arr = np.asarray(num_points, dtype=np.int64)
        num_polygons = num_points_arr.shape[0]
        total = num_points_arr.sum()
        edge_index = np.vstack([
            np.arange(total, dtype=np.int64),
            np.repeat(np.arange(num_polygons, dtype=np.int64), num_points_arr)
        ])
        edge_index = torch.from_numpy(edge_index)

        # edge_index = torch.stack([
        #     torch.arange(num_points.sum(), dtype=torch.long),
        #     torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)
        # ], dim=0)

        # 6) Package map_data
        map_data = {
            "map_polygon": {
                "num_nodes": num_polygons,
                "type":       polygon_type,
                "light_type": polygon_light_type,
            },
            "map_point": {
                "num_nodes": 0 if num_points.numel()==0 else num_points.sum().item(),
                "position":  torch.tensor([], dtype=torch.float) if num_points.sum()==0
                            else torch.cat(point_position, dim=0),
                "type":      torch.tensor([], dtype=torch.uint8) if num_points.sum()==0
                            else torch.cat(point_type, dim=0),
            },
            ("map_point","to","map_polygon"): {
                "edge_index": edge_index
            }
        }
        return map_data


    def _process_map(self, map_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes sure the map polylines are of uniform dimensions
        """
        pt2pl = map_data[("map_point", "to", "map_polygon")]["edge_index"]
        split_polyline_type = []
        split_polyline_pos = []
        split_polyline_theta = []
        split_polygon_type = []
        split_light_type = []
        for i in sorted(torch.unique(pt2pl[1])):
            index = pt2pl[0, pt2pl[1] == i]
            if len(index) <= 2:
                continue
            
            polygon_type = map_data["map_polygon"]["type"][i]
            light_type = map_data["map_polygon"]["light_type"][i]
            cur_type = map_data["map_point"]["type"][index]
            cur_pos = map_data["map_point"]["position"][index, :2]
            # assert len(np.unique(cur_type)) == 1
            #split_polyline = _interplating_polyline(cur_pos.numpy(), distance=0.5, split_distance=5)
            split_polyline = _interpolating_polyline(cur_pos.numpy())

            if split_polyline is None:
                continue
            split_polyline_pos.append(split_polyline[..., :2])
            split_polyline_theta.append(split_polyline[..., 2])
            split_polyline_type.append(cur_type[0].repeat(split_polyline.shape[0]))
            split_polygon_type.append(polygon_type.repeat(split_polyline.shape[0]))
            split_light_type.append(light_type.repeat(split_polyline.shape[0]))
        data = {}
        if len(split_polyline_pos) == 0:  # add dummy empty map
            data["map_save"] = {
                # 6e4 such that it's within the range of float16.
                "traj_pos": torch.zeros([1, 3, 2], dtype=torch.float32) + 6e4,
                "traj_theta": torch.zeros([1], dtype=torch.float32),
            }
            data["pt_token"] = {
                "type": torch.tensor([0], dtype=torch.uint8),
                "pl_type": torch.tensor([0], dtype=torch.uint8),
                "light_type": torch.tensor([0], dtype=torch.uint8),
                "num_nodes": 1,
            }
        else:
            data["map_save"] = {
                "traj_pos": torch.cat(split_polyline_pos, dim=0),  # [num_nodes, 3, 2]
                "traj_theta": torch.cat(split_polyline_theta, dim=0)[:, 0],  # [num_nodes]
            }
            data["pt_token"] = {
                "type": torch.cat(split_polyline_type, dim=0),  # [num_nodes], uint8
                "pl_type": torch.cat(split_polygon_type, dim=0),  # [num_nodes], uint8
                "light_type": torch.cat(split_light_type, dim=0),  # [num_nodes], uint8
                "num_nodes": data["map_save"]["traj_pos"].shape[0],
            }
        return data

    def _get_agent_data(self, track_infos, tracks_to_predict, sdc_track_index, current_time_index, dim=3):
        """
        Process raw tracking data into per-agent tensors and lists.

        We already filter invalid agents out during preprocessing

        tracks_to_predict.keys()
        dict_keys(['track_index', 'track_difficulties', 'track_types', 'object_type', 'track_id', 'ego_track_difficulty', 'ego_track_type'])
        """
        # Unpack raw data from track_info
        trajs = track_infos["trajs"]  # shape: (num_agents, num_steps, 10)
        # map "OTHER" object types to standard vehicle types for tokenization later
        # TODO 20.6.2025: generally it might be a better strategy to tokenize based on given shape not type
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
        out_dict["train_mask"][sdc_track_index] = True # should be included in tracks_to_predict anyways
        #print("TRAIN ON ", sum(out_dict["train_mask"]), " OF ", num_agents)
        return out_dict
    
    def postprocess(self, output: Dict[str, Any]) -> Dict[str,Any]:
        #tokenized_map, tokenized_agent = self.tokenizer(Batch.from_data_list([HeteroData(output[0])]))
        return output
    
    def plot_map(self, map_data, dataset_name, ego_track):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.title(dataset_name)
        plt.axis('equal')
        plt.scatter(map_data['map_point']['position'][:, 0],map_data['map_point']['position'][:, 1], s=0.2, c='black', edgecolors='none')
        plt.scatter(ego_track[:,0], ego_track[:,1], s=1, c="red", edgecolors="none")
        plt.show()

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



# def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
#     try:
#         return ls.index(elem)
#     except ValueError:
#         return None

    # def _get_map_data(self, map_infos: Dict[str, Any], tf_current_light: pd.DataFrame, center: np.ndarray, dim: int = 3) -> Dict[str, Any]:
    #     """
    #     Extract map features and connectivity from raw map_infos and current traffic light info.

    #     Args:
    #         map_infos: Dictionary with keys 'lane', 'crosswalk', 'road_edge', 'road_line', and 'all_polylines'.
    #         tf_current_light: DataFrame with current traffic light states.
    #         dim: Spatial dimension (default: 3).

    #     Returns:
    #         Dictionary containing processed map polygon and point features, along with connectivity indices.
    #     """
    #     # Ignored map elements: sidewalk
    #     # Extract raw map segments.
    #     lane_segments = map_infos['lane']
    #     all_polylines = map_infos["all_polylines"]
    #     crosswalks = map_infos['crosswalk']
    #     road_edges = map_infos['road_edge']
    #     road_lines = map_infos['road_line']
    #     obstacles = map_infos["obstacles"]

    #     cx, cy = np.array(center)
    #     half = self.config["map_range"]/2
    #     mask = (
    #         (all_polylines[:, 0] >= cx - half) &
    #         (all_polylines[:, 0] <= cx + half) &
    #         (all_polylines[:, 1] >= cy - half) &
    #         (all_polylines[:, 1] <= cy + half)
    #     )

    #     def filter_segments(segments, mask):
    #         out = []
    #         for seg in segments:
    #             start, end = seg["polyline_index"]
    #             if mask[start:end].any():
    #                 out.append(seg)
    #         return out
    
    #     lane_segments = filter_segments(lane_segments, mask)
    #     crosswalks = filter_segments(crosswalks, mask)    
    #     road_edges = filter_segments(road_edges, mask)    
    #     road_lines = filter_segments(road_lines, mask)    
    #     obstacles = filter_segments(obstacles, mask) 

    #     # Gather polygon IDs from each segment type.
    #     lane_segment_ids = [info["id"] for info in lane_segments]
    #     cross_walk_ids = [info["id"] for info in crosswalks]
    #     road_edge_ids = [info["id"] for info in road_edges]
    #     road_line_ids = [info["id"] for info in road_lines]
    #     obstacle_ids = [info["id"] for info in obstacles]
    #     polygon_ids = lane_segment_ids + road_edge_ids + road_line_ids + cross_walk_ids + obstacle_ids
    #     num_polygons = len(lane_segment_ids) + len(road_edge_ids) + len(road_line_ids) + len(cross_walk_ids) + len(obstacle_ids)

    #     # Initialize tensors for polygon-level features.
    #     polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    #     polygon_light_type = torch.ones(num_polygons, dtype=torch.uint8) * 3

    #     # Initialize lists for point-level features for each polygon.
    #     point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    #     point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
    #     point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
    #     point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
    #     point_type: List[Optional[torch.Tensor]] = [None] * num_polygons

    #     for lane_segment in lane_segments:
    #         lane_segment = easydict.EasyDict(lane_segment)
    #         lane_segment_idx = polygon_ids.index(lane_segment.id)
    #         polyline_index = lane_segment.polyline_index
    #         valid_points = mask[polyline_index[0]:polyline_index[1]]
    #         centerline = all_polylines[polyline_index[0]:polyline_index[1], :][valid_points]
    #         centerline = torch.from_numpy(centerline).float()
    #         polygon_type[lane_segment_idx] = polyline_type[lane_segment.type] #_polygon_types.index(LANE_HASH[lane_segment.type])

    #         res = tf_current_light[tf_current_light["lane_id"] == str(lane_segment.id)]
    #         if len(res) != 0:
    #             polygon_light_type[lane_segment_idx] = traffic_light_state_to_int[res["state"].item()]#_polygon_light_type.index(res["state"].item())

    #         point_position[lane_segment_idx] = torch.cat([centerline[:-1, :dim]], dim=0)
    #         center_vectors = centerline[1:] - centerline[:-1]
    #         point_orientation[lane_segment_idx] = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
    #         point_magnitude[lane_segment_idx] = torch.norm(torch.cat([center_vectors[:, :2]], dim=0), p=2, dim=-1)
    #         point_height[lane_segment_idx] = torch.cat([center_vectors[:, 2]], dim=0)
    #         center_type = polyline_type["CENTERLINE"] #_point_types.index('CENTERLINE')
    #         point_type[lane_segment_idx] = torch.cat(
    #             [torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)

    #     for lane_segment in road_edges:
    #         lane_segment = easydict.EasyDict(lane_segment)
    #         lane_segment_idx = polygon_ids.index(lane_segment.id)
    #         polyline_index = lane_segment.polyline_index
    #         valid_points = mask[polyline_index[0]:polyline_index[1]]
    #         centerline = all_polylines[polyline_index[0]:polyline_index[1], :][valid_points]
    #         centerline = torch.from_numpy(centerline).float()
    #         polygon_type[lane_segment_idx] = polyline_type["LANE_SURFACE_STREET"] #_polygon_types.index("VEHICLE")

    #         point_position[lane_segment_idx] = torch.cat([centerline[:-1, :dim]], dim=0)
    #         center_vectors = centerline[1:] - centerline[:-1]
    #         point_orientation[lane_segment_idx] = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
    #         point_magnitude[lane_segment_idx] = torch.norm(torch.cat([center_vectors[:, :2]], dim=0), p=2, dim=-1)
    #         point_height[lane_segment_idx] = torch.cat([center_vectors[:, 2]], dim=0)
    #         center_type = polyline_type["ROAD_EDGE_BOUNDARY"] #polyline_type["BOUNDARY_LINE"] #_point_types.index('EDGE')
    #         point_type[lane_segment_idx] = torch.cat(
    #             [torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)

    #     for lane_segment in road_lines:
    #         lane_segment = easydict.EasyDict(lane_segment)
    #         lane_segment_idx = polygon_ids.index(lane_segment.id)
    #         polyline_index = lane_segment.polyline_index
    #         valid_points = mask[polyline_index[0]:polyline_index[1]]
    #         centerline = all_polylines[polyline_index[0]:polyline_index[1], :][valid_points]
    #         centerline = torch.from_numpy(centerline).float()

    #         polygon_type[lane_segment_idx] = polyline_type["LANE_SURFACE_STREET"] #_polygon_types.index("VEHICLE")

    #         point_position[lane_segment_idx] = torch.cat([centerline[:-1, :dim]], dim=0)
    #         center_vectors = centerline[1:] - centerline[:-1]
    #         point_orientation[lane_segment_idx] = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
    #         point_magnitude[lane_segment_idx] = torch.norm(torch.cat([center_vectors[:, :2]], dim=0), p=2, dim=-1)
    #         point_height[lane_segment_idx] = torch.cat([center_vectors[:, 2]], dim=0)
    #         center_type = polyline_type[lane_segment.type] #_point_types.index(boundary_type_hash[lane_segment.type])
    #         point_type[lane_segment_idx] = torch.cat(
    #             [torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)

    #     for crosswalk in crosswalks:
    #         crosswalk = easydict.EasyDict(crosswalk)
    #         lane_segment_idx = polygon_ids.index(crosswalk.id)
    #         polyline_index = crosswalk.polyline_index
    #         centerline = all_polylines[polyline_index[0]:polyline_index[1], :]
    #         centerline = torch.from_numpy(centerline).float()

    #         polygon_type[lane_segment_idx] = polyline_type["CROSSWALK"] #_polygon_types.index("PEDESTRIAN")

    #         point_position[lane_segment_idx] = torch.cat([centerline[:-1, :dim]], dim=0)
    #         center_vectors = centerline[1:] - centerline[:-1]
    #         point_orientation[lane_segment_idx] = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
    #         point_magnitude[lane_segment_idx] = torch.norm(torch.cat([center_vectors[:, :2]], dim=0), p=2, dim=-1)
    #         point_height[lane_segment_idx] = torch.cat([center_vectors[:, 2]], dim=0)
    #         center_type = polyline_type["CROSSWALK"] #_point_types.index("CROSSWALK")
    #         point_type[lane_segment_idx] = torch.cat(
    #             [torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            
    #     for obstacle in obstacles:
    #         obstacle = easydict.EasyDict(obstacle)
    #         lane_segment_idx = polygon_ids.index(obstacle.id)
    #         polyline_index = obstacle.polyline_index
    #         centerline = all_polylines[polyline_index[0]:polyline_index[1], :]
    #         centerline = torch.from_numpy(centerline).float()

    #         polygon_type[lane_segment_idx] = polyline_type["OBSTACLE"] 
    #         point_position[lane_segment_idx] = torch.cat([centerline[:-1, :dim]], dim=0)
    #         center_vectors = centerline[1:] - centerline[:-1]
    #         point_orientation[lane_segment_idx] = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
    #         point_magnitude[lane_segment_idx] = torch.norm(torch.cat([center_vectors[:, :2]], dim=0), p=2, dim=-1)
    #         point_height[lane_segment_idx] = torch.cat([center_vectors[:, 2]], dim=0)
    #         center_type = polyline_type["OBSTACLE"] 
    #         point_type[lane_segment_idx] = torch.cat(
    #             [torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            

    #     # polygon to polygon connectivity
    #     num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
    #     point_to_polygon_edge_index = torch.stack(
    #         [torch.arange(num_points.sum(), dtype=torch.long),
    #             torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        
    #     polygon_to_polygon_edge_index = []
    #     polygon_to_polygon_type = []
    #     for lane_segment in lane_segments:
    #         lane_segment = easydict.EasyDict(lane_segment)
    #         lane_segment_idx = polygon_ids.index(lane_segment.id)
    #         pred_inds = []
    #         for pred in lane_segment.entry_lanes:
    #             pred_idx = safe_list_index(polygon_ids, pred)
    #             if pred_idx is not None:
    #                 pred_inds.append(pred_idx)
    #         if len(pred_inds) != 0:
    #             polygon_to_polygon_edge_index.append(
    #                 torch.stack([torch.tensor(pred_inds, dtype=torch.long),
    #                             torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
    #             polygon_to_polygon_type.append(
    #                 torch.full((len(pred_inds),), polyline_relations_to_int['PRED'], dtype=torch.uint8))
    #         succ_inds = []
    #         for succ in lane_segment.exit_lanes:
    #             succ_idx = safe_list_index(polygon_ids, succ)
    #             if succ_idx is not None:
    #                 succ_inds.append(succ_idx)
    #         if len(succ_inds) != 0:
    #             polygon_to_polygon_edge_index.append(
    #                 torch.stack([torch.tensor(succ_inds, dtype=torch.long),
    #                             torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
    #             polygon_to_polygon_type.append(
    #                 torch.full((len(succ_inds),), polyline_relations_to_int['SUCC'], dtype=torch.uint8))
    #         if len(lane_segment.left_neighbors) != 0:
    #             left_neighbor_ids = lane_segment.left_neighbors
    #             for left_neighbor_id in left_neighbor_ids:
    #                 left_idx = safe_list_index(polygon_ids, left_neighbor_id)
    #                 if left_idx is not None:
    #                     polygon_to_polygon_edge_index.append(
    #                         torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
    #                     polygon_to_polygon_type.append(
    #                         torch.tensor([polyline_relations_to_int['LEFT']], dtype=torch.uint8))
    #         if len(lane_segment.right_neighbors) != 0:
    #             right_neighbor_ids = lane_segment.right_neighbors
    #             for right_neighbor_id in right_neighbor_ids:
    #                 right_idx = safe_list_index(polygon_ids, right_neighbor_id)
    #                 if right_idx is not None:
    #                     polygon_to_polygon_edge_index.append(
    #                         torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
    #                     polygon_to_polygon_type.append(
    #                         torch.tensor([polyline_relations_to_int['RIGHT']], dtype=torch.uint8))
    #     if len(polygon_to_polygon_edge_index) != 0:
    #         polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
    #         polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
    #     else:
    #         polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
    #         polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)


    #     # Assemble the final map data structure.
    #     map_data = {
    #         'map_polygon': {},
    #         'map_point': {},
    #         ('map_point', 'to', 'map_polygon'): {},
    #         ('map_polygon', 'to', 'map_polygon'): {},
    #     }
    #     map_data['map_polygon']['num_nodes'] = num_polygons
    #     map_data['map_polygon']['type'] = polygon_type
    #     map_data['map_polygon']['light_type'] = polygon_light_type
    #     if len(num_points) == 0:
    #         map_data['map_point']['num_nodes'] = 0
    #         map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
    #         map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
    #         map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
    #         if dim == 3:
    #             map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
    #         map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
    #         map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
    #     else:
    #         map_data['map_point']['num_nodes'] = num_points.sum().item()
    #         map_data['map_point']['position'] = torch.cat(point_position, dim=0)
    #         map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
    #         map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
    #         if dim == 3:
    #             map_data['map_point']['height'] = torch.cat(point_height, dim=0)
    #         map_data['map_point']['type'] = torch.cat(point_type, dim=0)
    #     map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
    #     map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
    #     map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

    #     return map_data
    