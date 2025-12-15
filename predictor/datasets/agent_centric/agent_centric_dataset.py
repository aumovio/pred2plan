"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

from typing import List, Dict, Any
from omegaconf import DictConfig
import compress_pickle as pickle
import numpy as np
import torch
from collections import defaultdict
from typing import List, Dict, Any, Union, Optional
# import matplotlib
# # matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from predictor.datasets.base_dataset import BaseDataset
from predictor.datasets.agent_centric.utils import rotate_points_along_z, rotate_points_along_z_tensor, find_true_segments
from predictor.utils.visualization import check_loaded_data
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
from metadrive.scenario import utils as sd_utils
from predictor.datasets.types import object_type, polyline_type, track_type
from common_utils.time_tracking import timeit
import time

object_type = defaultdict(lambda: 0, object_type)
polyline_type = defaultdict(lambda: 7, polyline_type)

class AgentCentricBaseDataset(BaseDataset):

    def __init__(self, config: DictConfig, data_splits: List[Dict]):
        super().__init__(config, data_splits)

    def process_single_scenario_file(self, file_path, starting_frame, tracks_to_predict, obstacles):
        scenario = sd_utils.read_scenario_data(file_path)
        scenario = self.postprocess(self.process(self.preprocess(scenario, starting_frame)))
        return scenario
    
    def process_single_scenario(self, scenario: Dict, starting_frame: int, tracks_to_predict: str, insert_static_obstacles: Optional[Dict[str, Any]]=None) -> Dict:
        converted_scenario = self.postprocess(self.process(self.preprocess(scenario, 
                                                                           split_name="nuplan_sim", 
                                                                           starting_frame=starting_frame, 
                                                                           tracks_to_predict = tracks_to_predict,
                                                                           obstacles = insert_static_obstacles
                                                                           )))

        batch_dict = self.collate_fn([converted_scenario])
        return batch_dict

    def _build_sample_idx(self, file_map: Dict, filter: Dict) -> List:
        selected_obj_types = set(object_type[t] for t in filter["center_object_types"])
        selected_track_types = set(track_type[t] for t in filter["center_track_types"])
        selected_ego_track_types = set(track_type[t] for t in filter["ego_track_types"])
        def is_whitelisted(sid, whitelist): return (not whitelist) or any(w in sid for w in whitelist)
        # filter samples
        sample_index = [
            (file, idx) 
            for file, file_info in file_map.items() 
            for idx, sid in enumerate(file_info["scenario_id"])
            if (sid not in self.config.blacklist)
                and is_whitelisted(sid, self.config.whitelist)
                and (file_info["center_objects_type"][idx] in selected_obj_types)
                and (filter["center_track_difficulty"][0] <= file_info["center_track_difficulty"][idx] < filter["center_track_difficulty"][1])
                and (file_info["center_track_type"][idx] in selected_track_types)
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

    # @timeit
    def process(self, internal_format: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes internal scenario data format into model-ready training samples based on configurations.
        
        Key processing steps:
        1. Extracts past and future trajectories for all agents
        2. Identifies and processes agents of interest (center objects)
        3. Processes HD map data and aligns it with center objects
        4. Applies attribute masking based on configuration
        5. Creates separate samples for each center object
        """
        info = internal_format
        if len(info['tracks_to_predict']['track_index']) == 0: # if no tracks_to_predict, abort
            return None
        
        scene_id = info['scenario_id']
        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)
        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        track_infos = info['track_infos']
        obj_types = np.array(track_infos['object_type'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        center_objects, track_index_to_predict = self.get_interesting_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )
        # if center_objects is None: return None

        sample_num = center_objects.shape[0]

        (
            obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state,
            obj_trajs_future_mask, center_gt_trajs,
            center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new
        ) = self.get_agent_data(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types
        )

        ret_dict = {
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],
            'map_center': info['map_center'],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict],

            'center_track_difficulty': info["tracks_to_predict"]["track_difficulties"],
            'center_track_type': info["tracks_to_predict"]["track_types"],
            'ego_track_difficulty': info["tracks_to_predict"]["ego_track_difficulty"]*len(center_objects),
            'ego_track_type': info["tracks_to_predict"]["ego_track_type"]*len(center_objects),
        }

        

        if info['map_infos']['all_polylines'].__len__() == 0:
            info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
            print(f'Warning: empty HDMap {scene_id}')

        # if self.config.manually_split_lane:
        #     map_polylines_data, map_polylines_mask, map_polylines_center = self.get_manually_split_map_data(
        #         center_objects=center_objects, map_infos=info['map_infos'])
        # else:
        map_polylines_data, map_polylines_mask, map_polylines_center = self.get_map_data(
            center_objects=center_objects, map_infos=info['map_infos'])
            
        # breakpoint()
        # matplotlib.use("TkAgg")
        # self.visualize_map_for_agent(map_polylines_data, map_polylines_mask, map_polylines_center, name=info["dataset_name"])
        # [tup["type"] for tup in info["map_infos"]["road_line"]]

        # Add map-related data to the return dictionary
        ret_dict.update({
            'map_polylines': map_polylines_data,
            'map_polylines_mask': map_polylines_mask.astype(bool),
            'map_polylines_center': map_polylines_center
        })
        
        # masking out unused attributes to Zero
        masked_attributes = self.config['masked_attributes']
        if 'z_axis' in masked_attributes:
            ret_dict['obj_trajs'][..., 2] = 0
            ret_dict['map_polylines'][..., 2] = 0
        if 'size' in masked_attributes:
            ret_dict['obj_trajs'][..., 3:6] = 0
        if 'velocity' in masked_attributes:
            ret_dict['obj_trajs'][..., 25:27] = 0
        if 'acceleration' in masked_attributes:
            ret_dict['obj_trajs'][..., 27:29] = 0
        if 'heading' in masked_attributes:
            ret_dict['obj_trajs'][..., 23:25] = 0

        # Convert all np.float64 arrays to np.float32 for memory efficiency
        ret_dict = {k: v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v
                for k, v in ret_dict.items()}

        ret_dict['map_center'] = ret_dict['map_center'].repeat(sample_num, axis=0)
        ret_dict['dataset_name'] = [info['dataset_name']] * sample_num
        ret_dict['dataset_split'] = [info['dataset_split']] * sample_num

        # Generate a list of dictionaries, where each dictionary corresponds to a single sample
        # from `ret_dict`

        ret_list = [{k: v[i] for k, v in ret_dict.items()} for i in range(sample_num)]

        return ret_list

    # @timeit
    def get_agent_data(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types
    ):

        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 1, :, 0] = 1
        object_onehot_mask[:, obj_types == 2, :, 1] = 1
        object_onehot_mask[:, obj_types == 3, :, 2] = 1
        object_onehot_mask[np.arange(num_center_objects), track_index_to_predict, :, 3] = 1
        object_onehot_mask[:, sdc_track_index, :, 4] = 1

        object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1
        object_time_embedding[:, :, :, -1] = timestamps

        object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]
        vel_pre = np.roll(vel, shift=1, axis=2)
        acce = (vel - vel_pre) / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]

        obj_trajs_data = np.concatenate([
            obj_trajs[:, :, :, 0:6],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9],
            acce,
        ], axis=-1)

        obj_trajs_mask = obj_trajs[:, :, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        obj_trajs_future_state = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1] #[A_tar, A, T, F]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]

        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        max_num_agents = self.config['max_num_agents']
        object_dist_to_center = np.linalg.norm(obj_trajs_data[:, :, -1, 0:2], axis=-1)

        object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10
        topk_idxs = np.argsort(object_dist_to_center, axis=-1)[:, :max_num_agents]

        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
        obj_trajs_future_state = np.take_along_axis(obj_trajs_future_state, topk_idxs, axis=1)
        obj_trajs_future_mask = np.take_along_axis(obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
        track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)

        obj_trajs_data = np.pad(obj_trajs_data, ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
        obj_trajs_mask = np.pad(obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
        obj_trajs_pos = np.pad(obj_trajs_pos, ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
        obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                    ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
        obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                        ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
        obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                       ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

        return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new)

    def get_interesting_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        center_objects_list = []
        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]
            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
        # if len(center_objects_list) == 0:
        #     print(f'Warning: no center objects at time step {current_time_index} for scene_id={scene_id}')
        #     return None, []
        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict)
        return center_objects, track_index_to_predict

    def transform_trajs_to_center_coords(self, obj_trajs, center_xyz, center_heading, heading_index,
                                         rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = np.tile(obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
            angle=-center_heading
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].reshape(num_center_objects, -1, 2),
                angle=-center_heading
            ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    # @timeit
    def get_map_data(self, center_objects, map_infos):
        # Number of center objects (agents)
        num_center_objects = center_objects.shape[0]
        # prefiltering out all map elements that cant possibly be inside the map range of a center object
        def transform_to_center_coordinates(neighboring_polylines):
            # Shift coordinates so that the center object is at the origin.
            neighboring_polylines[:, :, 0:3] -= center_objects[:, None, 0:3]
            # Rotate the x, y coordinates for geometric alignment.
            neighboring_polylines[:, :, 0:2] = rotate_points_along_z(
                points=neighboring_polylines[:, :, 0:2],
                angle=-center_objects[:, 6]
            )
            # Also rotate the directional features.
            neighboring_polylines[:, :, 3:5] = rotate_points_along_z(
                points=neighboring_polylines[:, :, 3:5],
                angle=-center_objects[:, 6]
            )
            return neighboring_polylines

        # Expand the 'all_polylines' array to match the number of center objects.
        polylines = np.expand_dims(map_infos['all_polylines'].copy(), axis=0).repeat(num_center_objects, axis=0)
        map_polylines = transform_to_center_coordinates(neighboring_polylines=polylines)
        num_of_src_polylines = self.config['max_num_roads']
        map_infos['polyline_transformed'] = map_polylines

        # Retrieve configuration parameters.
        all_polylines = map_infos['polyline_transformed']
        max_points_per_lane = self.config.get('max_num_points_per_lane', 20)
        line_type = self.config.get('line_type', [])
        map_range = self.config.get('map_range', None)
        center_offset = self.config.get('center_offset_of_map', (30.0, 0))
        num_agents = all_polylines.shape[0]
        
        # Lists to accumulate processed polyline segments and their corresponding masks.
        polyline_list = []
        polyline_mask_list = []

        # Iterate over each key in map_infos that corresponds to a specific map element type.
        for k, v in map_infos.items():
            if k == 'all_polylines' or k not in line_type:
                continue  # Skip keys that are not in our desired line types.
            if len(v) == 0:
                continue  # Skip if there is no data under this key.
                
            # Process each dictionary entry corresponding to an individual map element.
            for polyline_dict in v:
                polyline_index = polyline_dict.get('polyline_index', None)
                # Skip if polyline_index is missing or indicates zero points.
                if polyline_index is None or polyline_index[1] <= polyline_index[0]:
                    continue

                # Slice the polyline from all_polylines using the given start and end indices.
                polyline_segment = all_polylines[:, polyline_index[0]:polyline_index[1]]
                # Shift x and y coordinates by center_offset.
                polyline_segment_x = polyline_segment[:, :, 0] - center_offset[0]
                polyline_segment_y = polyline_segment[:, :, 1] - center_offset[1]
                # Create a boolean mask: True for points within map_range.
                in_range_mask = (abs(polyline_segment_x) < map_range) * (abs(polyline_segment_y) < map_range)

                # For each agent, detect contiguous valid segments in the in_range_mask.
                segment_index_list = []
                for i in range(polyline_segment.shape[0]):
                    if in_range_mask[i].size == 1:
                        # Special case: single point polyline.
                        if in_range_mask[i][0]:
                            segment_index_list.append([slice(0, 1)])
                        else:
                            segment_index_list.append([])
                    else:
                        # Use an external helper to find contiguous regions where points are valid.
                        segment_index_list.append(find_true_segments(in_range_mask[i]))

                # If all agents have no valid segments for this map element, skip it.
                if all(len(seg_list) == 0 for seg_list in segment_index_list):
                    continue

                # Determine the maximum number of segments across all agents for this element.
                max_segments = max([len(x) for x in segment_index_list])
                # Allocate arrays to hold segments and corresponding masks.
                segment_list = np.zeros([num_agents, max_segments, max_points_per_lane, 7], dtype=np.float32)
                segment_mask_list = np.zeros([num_agents, max_segments, max_points_per_lane], dtype=np.int32)

                # Process each agent's polyline segments.
                for i in range(polyline_segment.shape[0]):
                    # Skip if no valid points in this agent's polyline.
                    if in_range_mask[i].sum() == 0:
                        continue
                    segment_i = polyline_segment[i]
                    segment_indices = segment_index_list[i]
                    for num, seg_index in enumerate(segment_indices):
                        segment = segment_i[seg_index]
                        # If the segment is longer than allowed, sample uniformly.
                        if segment.shape[0] > max_points_per_lane:
                            indices = np.linspace(0, segment.shape[0] - 1, max_points_per_lane, dtype=int)
                            segment_list[i, num] = segment[indices]
                            segment_mask_list[i, num] = 1  # Mark all sampled points as valid.
                        else:
                            # Otherwise, copy available points and pad the rest with zeros.
                            segment_list[i, num, :segment.shape[0]] = segment
                            segment_mask_list[i, num, :segment.shape[0]] = 1

                # Append the processed segments and masks to the overall lists.
                polyline_list.append(segment_list)
                polyline_mask_list.append(segment_mask_list)

        # If no valid segments were found, return empty arrays.
        if len(polyline_list) == 0:
            return np.zeros((num_agents, 0, max_points_per_lane, 7)), np.zeros((num_agents, 0, max_points_per_lane))

        # Concatenate segments from different map elements along the polyline axis.
        batch_polylines = np.concatenate(polyline_list, axis=1)
        batch_polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        # Compute a distance measure for each polyline segment to select the most relevant ones.
        polyline_xy_offsetted = batch_polylines[:, :, :, 0:2] - np.reshape(center_offset, (1, 1, 1, 2))
        polyline_center_dist = np.linalg.norm(polyline_xy_offsetted, axis=-1).sum(-1) / np.clip(
            batch_polylines_mask.sum(axis=-1).astype(float), a_min=1.0, a_max=None)
        # Set distance to a very large value for segments with no valid points.
        polyline_center_dist[batch_polylines_mask.sum(-1) == 0] = 1e10
        topk_idxs = np.argsort(polyline_center_dist, axis=-1)[:, :num_of_src_polylines]

        # Expand dimensions so that we can index along the polyline dimension.
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        # Select the top-k nearest polyline segments.
        map_polylines = np.take_along_axis(batch_polylines, topk_idxs, axis=1)
        map_polylines_mask = np.take_along_axis(batch_polylines_mask, topk_idxs[..., 0], axis=1)

        # Pad along the polyline dimension to ensure a fixed number of source polylines.
        map_polylines = np.pad(map_polylines,
                            ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
        map_polylines_mask = np.pad(map_polylines_mask,
                                    ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        # Compute the center of each polyline segment using a weighted average of valid points.
        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(float)).sum(axis=-2)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0, a_max=None)

        # Prepare a shifted version of the x, y, z coordinates (used later for additional features).
        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        # Separate the map type (assumed to be in the last channel) from the geometry.
        map_types = map_polylines[:, :, :, -1]
        map_polylines = map_polylines[:, :, :, :-1]
        # One-hot encode the map types (15 defined types, reserving up to 21).
        map_types = np.eye(21)[map_types.astype(int)]
        # Concatenate the geometry, the shifted positions, and the one-hot encoded types.
        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
        # Zero out padded regions.
        map_polylines[map_polylines_mask == 0] = 0

        return map_polylines, map_polylines_mask, map_polylines_center

    def postprocess(self, output: List[Dict[str, Any]]) -> List[Dict[str,Any]]:
        """
        Sanity checks the processed samples
        """
        # simple sanity check
        output_kept = []
        for sample in output:
            if np.any(sample["map_polylines"]) and np.any(sample["obj_trajs"]):
                output_kept.append(sample)

        return output_kept

    def __getitem__(self, idx):
        samples = []
        for file, indices in self.sample_index[idx]:
            with open(file, 'rb') as f:
                all_samples =  pickle.load(f)
            selected_samples = [all_samples[i] for i in indices]
            # filter out any “empty” samples
            selected_samples = [
                s for s in selected_samples
                if np.any(s["map_polylines"]) and np.any(s["obj_trajs"])
            ]
            allowed_history = self.config["past_allowed"]
            for sample in selected_samples:
                sample["obj_trajs_mask"][:,:-allowed_history] = False
            samples.extend(selected_samples)
        return samples

    def collate_fn(self, data_list: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Collates a list of sample dictionaries into a single batch dictionary.
        """
        data_list = [sample for sublist in data_list for sample in sublist]
        batch_size = len(data_list)
        input_dict = {}

        # Create a dictionary where each key maps to a list of values from each sample.
        key_to_list = {key: [sample[key] for sample in data_list] for key in data_list[0].keys()}

        # Attempt to stack each list into a tensor.
        for key, val_list in key_to_list.items():
            try:
                # If the values are numpy arrays, stack them into one array and convert to tensor.
                input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
            except Exception:
                # Otherwise, keep the list as is.
                input_dict[key] = val_list

        # # Optionally, convert center_objects_type to a numpy array if it was successfully tensorized.
        # if 'center_objects_type' in input_dict and hasattr(input_dict['center_objects_type'], 'numpy'):
        #     input_dict['center_objects_type'] = input_dict['center_objects_type'].numpy()

        batch_dict = {
            'batch_size': batch_size,
            'input_dict': input_dict,
            'batch_sample_count': batch_size
        }
        return batch_dict

    
    # def visualize_map_for_agent(self, map_polylines, map_polylines_mask, map_polylines_center, agent_index=0, name=""):
    #     """
    #     Visualizes the map for a given center agent.
        
    #     Parameters:
    #         map_polylines: np.ndarray of shape (num_agents, num_segments, max_points, feature_dim)
    #             The processed map polylines for each agent.
    #         map_polylines_mask: np.ndarray of shape (num_agents, num_segments, max_points)
    #             A mask indicating valid points in each segment.
    #         map_polylines_center: np.ndarray of shape (num_agents, num_segments, 3)
    #             The computed center for each polyline segment.
    #         agent_index: int
    #             Index of the center agent to visualize.
    #     """
    #     # Extract data for the selected agent.
    #     agent_polylines = map_polylines[agent_index]  # shape: (num_segments, max_points, feature_dim)
    #     agent_masks = map_polylines_mask[agent_index]   # shape: (num_segments, max_points)
    #     agent_centers = map_polylines_center[agent_index]  # shape: (num_segments, 3)
        
    #     fig, ax = plt.subplots(figsize=(10, 10))
        
    #     num_segments = agent_polylines.shape[0]
    #     for seg_idx in range(num_segments):
    #         # Use the mask to select valid points.
    #         mask = agent_masks[seg_idx].astype(bool)
    #         if not mask.any():
    #             continue  # Skip segments with no valid points
            
    #         # Extract x and y coordinates; assuming they are in columns 0 and 1.
    #         segment = agent_polylines[seg_idx]
    #         valid_points = segment[mask]
    #         x = valid_points[:, 0]
    #         y = valid_points[:, 1]
            
    #         # Plot each segment as a continuous fine line.
    #         ax.plot(x, y, linestyle='-', linewidth=1, label=f"Segment {seg_idx}" if seg_idx == 0 else None)
        
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_title(f"Map Visualization for Center Agent {agent_index} from {name}")
    #     ax.legend()
    #     plt.grid(True)
    #     plt.show()