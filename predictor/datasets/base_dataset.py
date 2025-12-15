"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import os
import pickle
import shutil
from collections import defaultdict, OrderedDict
from multiprocessing import Pool
from typing import List, Dict, Any, Union
from omegaconf import DictConfig
from pathlib import Path
import random
from functools import partial, partialmethod
import compress_pickle as pickle
import math
import secrets
import time

import numpy as np
import torch
from metadrive.scenario import utils as sd_utils
from torch.utils.data import Dataset
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter

from predictor.datasets.data_utils import * #read_scenario, fill_into_buckets, get_polyline_dir,  generate_mask, classify_track, \
    # interpolate_polyline, estimate_kalman_filter, calculate_epe
from common_utils.time_tracking import timeit
from predictor.datasets.types import object_type, polyline_type, track_type

import logging
logger = logging.getLogger(__name__)

object_type = defaultdict(lambda: 0, object_type)
polyline_type = defaultdict(lambda: 7, polyline_type)

class BaseDataset(Dataset):

    def __init__(self, config: DictConfig, data_splits: List[Dict]):
        # past_len, future_len, etc are configured globally to ensure same tensor sizes
        self.config = config.data
        self.verbose = config.data.verbose_preprocessing
        self.plotting = config.data.plot_preprocessing
        self.cache_chunk_size = config.data.cache_chunk_size # chunk size in cached pkl files
        self.compression_cache = config.data.compression_cache
        self.compression_source = config.data.compression_source
        self.bucket_size = config.data.bucket_size
        self.sample_index = []
        # data splits contain split specific data_path, cache_path, data_usage and starting_frame 
        self.data_splits = data_splits

    def __len__(self):
        return len(self.sample_index)

    def load_from_cache(self) -> None:
        """
        Load all data splits from their caches and aggregate one dataset
        """
        self.sample_index = []
        for split in self.data_splits:
            if split["cache_path"].exists():
                sample_index= self._load_split_from_cache(split["cache_path"], split["filter"])
            else:
                raise FileNotFoundError(f"Error: The cache path {split['cache_path']} does not exist")
            self.sample_index+= sample_index
        random.shuffle(self.sample_index)
            # if self.config['store_data_in_memory']:
            #     self._load_data_into_memory(file_map)

        if self.bucket_size>1:
            files_to_indices = {}
            for file, idx in self.sample_index: 
                if file not in files_to_indices:
                    files_to_indices[file] = []
                files_to_indices[file].append(idx)
            rank_zero_info(f"Bucketing {len(self.sample_index)} samples...")
            buckets_full, buckets_to_fill = fill_into_buckets(files_to_indices, bucket_target_size=self.bucket_size)
            median_files_per_bucket = np.median([len(bucket) for bucket in buckets_full])
            bucketed_samples = sum([len(indices) for bucket in buckets_full for file, indices in bucket])
            rank_zero_info(f"Bucketed {bucketed_samples} samples into {len(buckets_full)} buckets " \
                        f"({2*self.cache_chunk_size} bucket size, median files per bucket: {median_files_per_bucket}) " \
                        f"with {len(self.sample_index)-bucketed_samples} samples lost.")
        else:
            buckets_full = [[(file, [sample])] for file, sample in self.sample_index]
        self.sample_index = buckets_full
        rank_zero_info('Data loading completed')

    def _load_split_from_cache(self, cache_path: Path, filter: Dict) -> Dict:
        """
        Load processed data from the cache if available.
        """
        rank_zero_info(f"Loading cached data from {cache_path}...")
        file_map_path = Path(cache_path, "file_map").with_suffix(self.compression_cache)
        if file_map_path.exists():
            with file_map_path.open('rb') as file:
                file_map = pickle.load(file)
        else:
            raise FileNotFoundError(f'Error: {file_map_path} not found')

        file_map = {cache_path / k: v for k,v in file_map.items()} # restore absolute path
        sample_index = self._build_sample_idx(file_map, filter)
        rank_zero_info(f'Loaded {len(sample_index)} samples from {cache_path}')
        return sample_index
    
    def _build_sample_idx(self, file_map: Dict, filter: Dict) -> List:
        raise NotImplementedError

    def cache_data(self) -> None:
        """
        Cache all data splits
        """
        for split in self.data_splits:
            if self.validate_split_cache(split["cache_path"]) and not self.config.get("overwrite_cache", False):
                rank_zero_info(f"Already existing cache {split['cache_path']} will be used")
                continue
            else:
                self._cache_datasplit(split["split"], split["data_path"], split["cache_path"], split["starting_frame"], split["tracks_to_predict"])

    def validate_split_cache(self, cache_path) -> bool:
        """
        Validates if cache_path actually exists with respective file_map
        """
        if cache_path.exists():
            matches = list(cache_path.glob("file_map.*"))
            if matches:
                if matches[0].suffix == self.compression_cache:
                    return True
                else:
                    raise ValueError(f"{matches[0]} has incorrect suffix. Check configured compression!")
            else:
                rank_zero_info(f"No file_map exists in cache folder")
        else:
            rank_zero_info(f"Cache folder does not exist yet")
        return False
    
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

        # concatenate the results
        file_map = {k: v for result in results for k, v in result.items()}

        with Path(cache_path, "file_map").with_suffix(self.compression_cache).open("wb") as f:
            pickle.dump(file_map, f)

        file_map = {cache_path / k: v for k,v in file_map.items()} # restore absolute path
        rank_zero_info(f'Cached {sum(v["sample_num"] for v in file_map.values())} samples from {data_path} in {cache_path}')

    def process_data_chunk(self, chunk_index: int, file_list: List, subset_list: List, split_name: str, data_path: Path, cache_path: Path, starting_frame: int, tracks_to_predict: str, total_workers: int, total_files: int) -> Dict:
        file_map = {}
        output_buffer = []
        save_cnt = 0
        worker_index = chunk_index % total_workers
        processed = chunk_index // total_workers
        max_chunk_size = self.config["max_processing_chunk_size"]

        for cnt, file_name in enumerate( tqdm(file_list)):
            # if chunk_index == 0 and cnt % max(int(len(file_list) / 10), 1) == 0:
            #     rank_zero_info(f'{cnt}/{len(file_list)} data chunks processed', flush=True)
            if worker_index == 0 and cnt % max(len(file_list) // 5, 1) == 0:
                print(f"{cnt + processed*max_chunk_size}/{total_files} files processed", flush=True)

            scenario = read_scenario(data_path, subset_list[cnt], file_name, self.compression_source)

            try:
                output = self.preprocess(scenario, split_name, starting_frame, tracks_to_predict)
                output = self.process(output)
                output = self.postprocess(output)
            except Exception as e:
                print('Error: {} in {}'.format(e, file_name), flush=True)
                output = None

            if output is None: continue

            output_buffer.extend(output)
            while len(output_buffer) >= self.cache_chunk_size:
                relative_save_path = Path(f"{chunk_index}_{save_cnt}").with_suffix(self.compression_cache) #cache_path / f'{worker_index}_{save_cnt}.pkl'
                to_save = output_buffer[:self.cache_chunk_size]
                output_buffer = output_buffer[self.cache_chunk_size:]
                with open(cache_path / relative_save_path , 'wb') as f:
                    pickle.dump(to_save, f)
                save_cnt += 1
                file_info = {}
                selected_keys = self.config.cache_metadata 
                for key in selected_keys:
                    file_info[key] = [x[key] for x in to_save]
                file_info['sample_num'] = len(to_save)
                file_map[relative_save_path] = file_info

        # flush the rest
        if output_buffer:
            relative_save_path = Path(f"{chunk_index}_{save_cnt}").with_suffix(self.compression_cache) #cache_path / f'{worker_index}_{save_cnt}.pkl'
            to_save = output_buffer[:self.cache_chunk_size]
            output_buffer = output_buffer[self.cache_chunk_size:]
            with open(cache_path / relative_save_path , 'wb') as f:
                pickle.dump(to_save, f)
            save_cnt += 1
            file_info = {}
            selected_keys = self.config.cache_metadata 
            for key in selected_keys:
                file_info[key] = [x[key] for x in to_save]
            file_info['sample_num'] = len(to_save)
            file_map[relative_save_path] = file_info
        return file_map

    # @timeit
    def preprocess(self, scenario: Dict[str, Any], split_name: str, starting_frame: int, tracks_to_predict: str, obstacles: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """
        Preprocesses a scenario to extract track, dynamic map, and HD map information.
        """
        # Configuration parameters
        past_length = self.config['past_len']
        future_length = self.config['future_len']
        total_steps = past_length + future_length
        current_time_index = self.config['past_len'] - 1
        trajectory_sample_interval = self.config['trajectory_sample_interval']
        frequency_mask = generate_mask(past_length - 1, total_steps, trajectory_sample_interval)
        if self.plotting: plot_raw_data(scenario, current_idx=starting_frame+current_time_index)
        # add inserted obstacles to scenario
        if obstacles: scenario["map_features"].update(obstacles) 

        metadata = scenario['metadata']
        if self.verbose: logger.debug(f"Preprocessing {metadata['scenario_id']} from {split_name}...")
        # Process components
        track_infos, new_obstacles = self._preprocess_tracks(scenario['tracks'], starting_frame, current_time_index, total_steps, frequency_mask, metadata["sdc_id"])
        scenario['map_features'].update(new_obstacles)
        map_infos = self._preprocess_map_features(scenario['map_features'])
        dynamic_map_infos = self._preprocess_dynamic_map_states(scenario['dynamic_map_states'], starting_frame, total_steps)

        # TODO: timestamp information might be misleading since it assumes starting_frame=0 always
        if "timestamps_seconds" not in metadata.keys():
            metadata['ts'] = metadata['ts'][starting_frame: starting_frame + total_steps] # trimm seconds
            metadata['timestamps_seconds'] = metadata.pop('ts')
        else:
            metadata["timestamps_seconds"] = metadata["timestamps_seconds"][starting_frame: starting_frame + total_steps]
        metadata['dataset_split'] = split_name

        sdc_track_index = track_infos["object_id"].index(metadata['sdc_id'])

        tracks_to_predict = self._select_tracks_to_predict(metadata, track_infos, tracks_to_predict, map_infos)
        # kalman_diffs = self._analyse_tracks(track_infos, tracks_to_predict, sdc_track_index)
        if self.verbose: logger.debug(f"Selected {len(tracks_to_predict['track_index'])} relevant tracks for prediction")
        if self.plotting: plot_preprocessed_data(map_infos, track_infos, tracks_to_predict, metadata, current_idx=current_time_index)

        return {
            "track_infos": track_infos,
            "dynamic_map_infos": dynamic_map_infos,
            "map_infos": map_infos,
            "scenario_id": metadata["scenario_id"],
            "dataset_name": metadata["dataset"],
            "dataset_split": split_name,
            "timestamps_seconds": metadata['timestamps_seconds'],
            "current_time_index": current_time_index,
            "sdc_track_index": sdc_track_index,
            "tracks_to_predict": tracks_to_predict,
            "map_center": metadata.get('map_center', np.zeros(3))[np.newaxis],
            "track_length": total_steps,
            "metadata": metadata
        }

    # @timeit
    def _preprocess_tracks(self, tracks: Dict[str, Any], starting_frame: int, current_time_index: int, total_steps: int, frequency_mask: np.ndarray, sdc_id: str) -> Dict[str, Any]:
        """
        Extracts track specific information such as states, object types, object ids and kalman difficulties of one scenario
        """
        object_ids, object_types, trajs, track_difficulties, track_types = [], [], [], [], []
        obstacles = {}

        # filter out ego
        ego_pos = tracks[sdc_id]["state"]["position"][...,:2]
        for k, v in tracks.items():
            state = v['state']
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            # aggregate
            # swap length and width if tracked the wrong way around in raw data
            if state["length"][starting_frame+current_time_index] < state["width"][starting_frame+current_time_index]:
                state_length = state["width"]
                state_width = state["length"]
            else:
                state_length = state["length"]
                state_width = state["width"]
            # 1st dimension: timesteps, 2nd dimension: position 0-2, length 3, width 4, height 5, heading 6, velocity 7-8, valid 9
            all_state = [state['position'], state_length, state_width, state['height'], state['heading'],
                        state['velocity'], state['valid']]
            all_state = np.concatenate(all_state, axis=-1)
            # fix nans
            all_state = np.nan_to_num(all_state, nan=0.0)
            # dont trust agent perceptions further than map_range/2 away from respective ego position (too noisy)
            if not self.config["trust_far_observations"]:
                distance = np.linalg.norm(all_state[:,:2] - ego_pos, axis=-1)
                all_state[:,-1] = (all_state[:,-1]).astype(bool) & (distance < self.config["map_range"]/2)
            # skip unobserved agents at current timestep
            observed = (all_state[:,-1]).astype(bool)
            if not observed[starting_frame+current_time_index] or not np.any(observed):
                continue

            # cut out frame
            all_state = all_state[starting_frame:]
            all_state = self._pad_all_state(all_state) # in case the length is too short
            all_state = all_state[:total_steps]
            assert all_state.shape[0] == total_steps, f'Error: {all_state.shape[0]} != {total_steps}'
            observed = (all_state[:,-1]).astype(bool)
            idx_obs = np.nonzero(observed)[0]
            num_obs = np.sum(observed)
            # correction of erroneous vehicle motion
            # e.g. in nuscenes motion end with 0 velocity, shifts has misaligned headings, etc.
            if num_obs>2 and np.all(all_state[idx_obs[-1],7:9] == 0): all_state[idx_obs[-1],7:9] = all_state[idx_obs[-2],7:9] 
            if v["type"] == "VEHICLE":
                all_state = self._correct_vehicle_motion(all_state)
            all_state = self._interpolate_missing_values(all_state)
            # if agent is unset, we assume it is a traffic obstacle instead 
            # TODO 18.6.2025: use actual shape of obstacle instead of small square (also consider tokenization for SMART)
            if object_type[v["type"]] == 0:
                obstacle = {}
                obstacle["type"] = "OBSTACLE"
                pt = all_state[observed,:3].mean(axis=0)
                obstacle_size=0.1
                obstacle["polyline"] = np.array([
                    [pt[0] - obstacle_size, pt[1] - obstacle_size, pt[2]],  # Bottom left
                    [pt[0] + obstacle_size, pt[1] - obstacle_size, pt[2]],  # Bottom right
                    [pt[0] + obstacle_size, pt[1] + obstacle_size, pt[2]],  # Top right
                    [pt[0] - obstacle_size, pt[1] + obstacle_size, pt[2]],   # Top left
                    [pt[0] - obstacle_size, pt[1] - obstacle_size, pt[2]]   # Closing back to Bottom left
                ])
                obstacles.update({k: obstacle})
                continue
            object_ids.append(k)
            object_types.append(object_type[v['type']])
            trajs.append(all_state)
            track_difficulties.append(self._compute_track_difficulty(all_state, horizon=self.config["future_required"]))
            track_types.append(self._compute_trajectory_type(all_state, horizon=self.config["future_required"]))
        trajs = np.stack(trajs, axis=0)
        trajs[..., -1] *= frequency_mask[np.newaxis]

        return {
            "object_id": object_ids, 
            "object_type": object_types,
            "track_difficulties": track_difficulties,
            "track_types": track_types,
            "trajs": trajs}, obstacles

    def _pad_all_state(self, track: np.ndarray) -> np.ndarray:
        past_required = self.config["past_required"]
        past_length = self.config["past_len"]
        future_required = self.config["future_required"]
        future_length = self.config["future_len"]
        track_length = track.shape[0]
        if track_length < past_required + future_required:
            raise ValueError(f"Track length {track_length} is smaller than past {past_required} and future {future_required} required")
        if track_length < past_length + future_required: # pad the past
            pad_past = past_length + future_required - track_length
        else:
            pad_past = 0
        track_length = track_length + pad_past

        if track_length < past_length + future_length: # pad the future
            pad_future = past_length + future_length - track_length
        else:
            pad_future = 0
        track = np.pad(track, ((pad_past, pad_future), (0,0)), mode="constant")
        return track
    
    def _interpolate_missing_values(self, track: np.ndarray) -> np.ndarray:
        observed = track[:,-1].astype(bool)
        observed_idx = np.where(observed)[0]
        if observed.sum() > 1:
            t_start, t_end = observed_idx[0], observed_idx[-1]
            t_in = np.arange(t_start, t_end + 1)
            f_interp1d = interp1d(observed_idx, track[:,:-1][observed], axis=0)
            track[t_in,:-1] = f_interp1d(t_in)
            track[t_in,-1] = True
        return track


    def _correct_vehicle_motion(self, track: np.ndarray) -> np.ndarray:  
        observed = track[:,-1].astype(bool)
        idx_obs = np.nonzero(observed)[0]
        num_obs = np.sum(observed)
        if num_obs > 2:
            delta_pos = np.diff(track[observed,:2], axis=0)
            cum_distance = np.sum(delta_pos, axis=0)
            # correct motionless vehicle
            if (np.abs(cum_distance) < num_obs/30).all(): 
                track[observed,:2] = np.median(track[observed,:2],axis=0)
                track[observed,6] = np.median(track[observed,6],axis=0)
                track[observed,7:9] = 0
                return track
            R, C = motion_heading_alignment(positions=track[...,:2], headings=track[...,6], observed=track[...,-1])
            # correct noisy motionless vehicle
            if (np.abs(cum_distance) < num_obs/10).all() and (np.abs(C)<=0.6):
                track[observed,:2] = np.median(track[observed,:2],axis=0)
                track[observed,6] = np.median(track[observed,6],axis=0)
                track[observed,7:9] = 0
                return track
            
            # sometimes there are repeated measurements at the end of trajectories
            acceleration = np.linalg.norm(np.diff(delta_pos,axis=0),axis=1)/0.1
            error_measurements = acceleration > 7
            error_rows = idx_obs[2:][error_measurements]
            track[error_rows, :] = 0
            observed = track[:,-1].astype(bool)
            idx_obs = np.nonzero(observed)[0]
            num_obs = np.sum(observed)
            delta_pos = np.diff(track[observed,:2], axis=0)

            # smooth velocity
            v_smoothed = savgol_filter(track[observed,7:9], window_length=min(7,num_obs), polyorder=1, axis=0)
            track[idx_obs,7:9]=v_smoothed

            # correct vehicle with wrong headings measurement
            if C < -0.6: 
                # we correct and smooth the heading
                dx, dy = delta_pos[:,0], delta_pos[:,1]
                headings_computed = np.arctan2(dy, dx)
                headings_computed = np.append(headings_computed, headings_computed[-1])
                track[idx_obs,6] = headings_computed
                # headings_measured = track[observed, 6]
                # raw_delta = headings_computed - headings_measured
                # deltas = (raw_delta + np.pi) % (2*np.pi) - np.pi
                # tol_rad = 5 * np.pi / 180
                # mask_180 = np.isclose(np.abs(deltas), np.pi, atol=tol_rad)
                # if np.any(mask_180):
                #     corrected_headings = headings_measured[mask_180] + np.pi
                #     corrected_headings_wrapped = (corrected_headings+ np.pi) % (2*np.pi) - np.pi
                #     corrected_rows = idx_obs[mask_180]
                #     track[corrected_rows,6] = corrected_headings_wrapped
                #     smoothed_headings = smooth_signal(track[:,6],track[:,-1],3)
                #     track[idx_obs,6] = smoothed_headings
        return track

        
    def _compute_track_difficulty(self, track: np.ndarray, horizon: int=40) -> float:
        '''
        Computes the kalman difficulty of a given track
        '''
        past_required = self.config["past_required"]
        past_length = self.config["past_len"]
        past_trajectory = track[:past_length,:2] # positions
        observed = track[:,-1].astype(bool) # validity
        # count the longest stretch of past observations from current timestep
        valid_past = np.sum(np.cumprod(observed[past_length-1::-1]))
        # count the longest stretch of future observations from current timestep
        valid_future = np.sum(np.cumprod(observed[past_length:]))
        past_trajectory = past_trajectory[observed[:past_length]] # check assumption: "occlusion" are no problem for kalman prediction
        # validity conditions
        if (valid_future>=horizon) and (valid_past>=past_required):
            kalman_pred = estimate_kalman_filter(past_trajectory, horizon)
            gt_future = track[past_length+horizon-1,:2]
            kalman_difficulty = calculate_epe(kalman_pred, gt_future)
        else:
            kalman_difficulty = 0 # -1
        return kalman_difficulty
    
    def _compute_trajectory_type(self, track: np.ndarray, horizon: int=40) -> int:
        """
        Computes the trajectory type of a given track
        """
        past_length = self.config["past_len"]
        start_pos = track[past_length,:3]
        end_pos = track[past_length+horizon,:3]
        start_vel = track[past_length,7:9]
        end_vel = track[past_length+horizon,7:9]
        start_head = track[past_length,6]
        end_head =  track[past_length+horizon,6]
        trajectory_type = classify_track(start_pos, end_pos, start_vel, end_vel, start_head, end_head)

        return trajectory_type
        
    # @timeit
    def _preprocess_dynamic_map_states(self, traffic_lights: Dict[str, Any], starting_frame: int, total_steps: int) -> Dict[str, Any]:
        """
        Extracts dynamic map states of one scenario (mostly traffic lights)
        """
        lane_id, state, stop_point = [], [], []
        
        for k, v in traffic_lights.items():
            lane_id.append([str(v['lane'])] * total_steps)
            state.append([s for s in v['state']['object_state'][starting_frame: starting_frame + total_steps]])
            stop_point.append([sp.tolist() if not isinstance(sp, list) else sp for sp in v['stop_point'][starting_frame: starting_frame + total_steps]])

        return {
            "lane_id" : np.array(lane_id), 
            "state" : np.array(state), 
            "stop_point": np.array(stop_point)
        }
    
    def _chunk_boundary_points(self, points, target=25):
        """
        Splits a list of points into chunks with roughly 'target' points per chunk.
        Ensures that the last point of a chunk is repeated as the first point of the next chunk.
        """
        n = len(points)
        if n == 0:
            return []
        # Calculate the number of chunks required
        num_chunks = math.ceil(n / target)
        # Use numpy.array_split to get nearly equal chunks
        raw_chunks = np.array_split(points, num_chunks)
        chunks = []
        for i, chunk in enumerate(raw_chunks):
            # Convert each point (a numpy array) to a tuple for reliable comparison.
            chunk_list = [tuple(pt) for pt in chunk]
            # For chunks after the first, ensure the first point repeats the last point of the previous chunk.
            if i > 0 and chunks:
                if chunks[-1][-1] != chunk_list[0]:
                    chunk_list.insert(0, chunks[-1][-1])
            chunks.append(chunk_list)
        return chunks
        
    # @timeit
    def _preprocess_map_features(self, map_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts static map elements of one scenario
        """
        lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, obstacles, sidewalk, polylines = [], [], [], [], [], [], [], [], []
        point_cnt = 0
        # split road boundaries into smaller segments to avoid worse sampling
        # should be handled by scenarionet but has been overlooked
        chunked_map_features = {}
        for k,v in map_features.items():
            if "boundary" in k and v["polyline"].shape[0]>40:
                for chunk in self._chunk_boundary_points(v["polyline"], target=25):
                    split_id = f"{k}{secrets.token_hex(2)}"
                    chunked_map_features[split_id] = {
                            "type": v["type"],
                            "polyline": np.asarray(chunk)
                        }
            else:
                chunked_map_features[k] = v

        for k, v in chunked_map_features.items():
            polyline_type_ = polyline_type[v['type']]
            if polyline_type_ == 0:
                continue
            cur_info = {'id': k, "type": v["type"]}

            # For lane segments (e.g., freeway, surface_street, bike_lane)
            if polyline_type_ in [1, 2, 3]:
                gi = v.get
                cur_info.update({
                "speed_limit_mph": gi("speed_limit_mph", None),
                "interpolating": gi("interpolating", None),
                "entry_lanes": gi("entry_lanes", None),
                "exit_lanes": gi("exit_lanes", None),
                "left_neighbors": gi("left_neighbor", []),
                "right_neighbors": gi("right_neighbor", []),
                })
                try:
                    cur_info['left_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                    } for x in v['left_neighbor']
                    ]
                    cur_info['right_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                    } for x in v['right_neighbor']
                    ]
                except:
                    cur_info['left_boundary'] = []
                    cur_info['right_boundary'] = []
                polyline = interpolate_polyline(v['polyline'])
                lane.append(cur_info)

            # For road lines.
            elif polyline_type_ in [6, 8, 9, 10, 11, 12, 13]:
                polyline = v.get("polyline", v.get("polygon"))
                polyline = interpolate_polyline(polyline)
                road_line.append(cur_info)

            # sidewalks
            elif polyline_type_ in [14]:
                polyline = v.get("polyline", v.get("polygon"))
                polyline = interpolate_polyline(polyline)
                sidewalk.append(cur_info)    

            # For special road lines (e.g. type 15,16).
            elif polyline_type_ in [7, 15, 16]:
                polyline = interpolate_polyline(v['polyline'])
                cur_info['type'] = "ROAD_LINE_SOLID_SINGLE_WHITE" #7
                road_edge.append(cur_info)

            # For stop signs.
            elif polyline_type_ in [17]:
                cur_info['lane_ids'] = v['lane']
                cur_info['position'] = v['position']
                stop_sign.append(cur_info)
                polyline = v['position'][np.newaxis]

            # For crosswalks and speed bumps (treated similarily in this version).
            elif polyline_type_ in [18, 19]:
                crosswalk.append(cur_info)
                pg = v["polygon"]
                if pg.shape[0] < 5: # scenarionet didnt close polygons for waymo
                    pg = np.vstack((pg, pg[0:1]))
                polyline = densify_polyline(pg, step=0.5)
            # For speed bumps (or treat as crosswalk in this version).
            # elif polyline_type_ in [19]:
            #     crosswalk.append(cur_info)
            #     polyline = v['polygon']

            # For obstacles
            elif polyline_type_ in [20]:
                obstacles.append(cur_info)
                polyline = densify_polyline(v["polyline"], step=0.5)

            # Build the [x, y, z, dx, dy, dz, type] row-block    
            if polyline.shape[-1] == 2:
                zcol = np.zeros((polyline.shape[0], 1), dtype=polyline.dtype)
                polyline = np.column_stack((polyline, zcol))  # (N, 3)
                # polyline = np.concatenate((polyline, np.zeros((polyline.shape[0], 1))), axis=-1)

            try:
                cur_polyline_dir = get_polyline_dir(polyline)
                type_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = polyline_type_
                cur_polyline = np.concatenate((polyline, cur_polyline_dir, type_array), axis=-1)
            except:
                cur_polyline = np.zeros((0, 7), dtype=np.float32)
            polylines.append(cur_polyline)
            cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
            point_cnt += len(cur_polyline)

        try:
            all_polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            all_polylines = np.zeros((0, 7), dtype=np.float32)

        return {
            "lane": lane,
            "road_line": road_line,
            "road_edge": road_edge,
            "sidewalk": sidewalk,
            "stop_sign": stop_sign,
            "crosswalk": crosswalk,
            "speed_bump": speed_bump,
            "obstacles": obstacles,
            "all_polylines": all_polylines
        }
    
    def _select_tracks_to_predict(self, metadata: Dict, track_infos: Dict, tracks_to_predict: str, map_infos: Dict) -> Dict:
        sdc_track_index = track_infos["object_id"].index(metadata['sdc_id']) # sdc_track_index
        min_track_difficulty = self.config["tracks_to_predict"]["min_track_difficulty"]
        if tracks_to_predict == "ego":
            track_index = sdc_track_index
        elif tracks_to_predict == "official":
            if metadata.get("tracks_to_predict", None) is None:
                raise ValueError(f"Selected {tracks_to_predict} tracks to predict but no official tracks in data")
            else:
                track_ids = list(metadata['tracks_to_predict'].keys())  # + ret.get('objects_of_interest', [])
                track_ids = list(set(track_ids))
                track_index = [track_infos["object_id"].index(id) for id in track_ids if
                                    id in track_infos['object_id']]
                # TODO: logger warning that a track to predict hasnt been found
        elif tracks_to_predict == "kalman":
            track_index = [idx for idx, diff in enumerate(track_infos["track_difficulties"]) if diff>min_track_difficulty]
            if sdc_track_index not in track_index: track_index += [sdc_track_index] # add ego track if not already included

        elif tracks_to_predict == "smart": # TODO: get all tracks in the scene for smart
            track_index = [idx for idx, diff in enumerate(track_infos["track_difficulties"])]
            if sdc_track_index not in track_index: track_index += [sdc_track_index] # add ego track if not already included
        else:
            raise ValueError(f"Selected {tracks_to_predict} tracks to predict not supported, please select either 'ego', 'official' or 'kalman'")

        if self.config["tracks_to_predict"]["do_filtering"]:
            track_index = self._remove_tracks_by_type(track_index, track_infos)
            track_index = self._remove_tracks_by_observation(track_index, track_infos)
            track_index = self._remove_tracks_by_distance_to_ego_and_number(track_index, track_infos, sdc_track_index)
            track_index = self._remove_tracks_by_distance_to_map(track_index, track_infos, map_infos)

        tracks_to_predict = {
            'track_index': track_index,
            'track_difficulties': [track_infos["track_difficulties"][idx] for idx in track_index],
            'track_types': [track_infos["track_types"][idx] for idx in track_index],
            'object_type': [track_infos["object_type"][idx] for idx in track_index],
            'track_id': [track_infos["object_id"][idx] for idx in track_index],
            'ego_track_difficulty': [track_infos["track_difficulties"][sdc_track_index]],
            'ego_track_type': [track_infos["track_types"][sdc_track_index]],
            #'median_track_difficulty': np.median([track_infos["track_difficulties"][idx] for idx in track_index]),
        }

        return tracks_to_predict
    
    def _remove_tracks_by_distance_to_ego_and_number(self, track_index: List, track_infos: Dict, sdc_track_index: int) -> List:
        if len(track_index)==0:
            return track_index
        
        max_distance = self.config["tracks_to_predict"]["max_distance_to_ego"]
        max_tracks = self.config["tracks_to_predict"]["max_num_agents"]
        # Extract ego trajectory positions (first two dimensions) and its validity mask (last dimension)
        ego_traj = track_infos["trajs"][sdc_track_index, :, :2]  # shape: (num_time, 2)
        ego_mask = track_infos["trajs"][sdc_track_index, :, -1].astype(bool)  # shape: (num_time,)

        filtered_track_index = []
        valid_tracks = []

        for idx in track_index:
            # Get track's positions and validity mask
            track_traj = track_infos["trajs"][idx, :, :2]  # shape: (num_time, 2)
            track_mask = track_infos["trajs"][idx, :, -1].astype(bool)

            # Consider only time points where both the ego and the track are valid
            valid_indices = ego_mask & track_mask
            # If there are valid observations, compute distances
            if np.any(valid_indices):
                # Compute Euclidean distances at valid timepoints
                distances = np.linalg.norm(ego_traj[valid_indices] - track_traj[valid_indices], axis=1)
                # If none of the distances are below the threshold, we keep this track
                min_distance = distances.min()
            else:
                # If there are no valid observations, decide if you want to keep or discard the track.
                # Here, we choose to keep the track as it was never observed close to the ego.
                min_distance = np.inf

            if min_distance <= max_distance:
                valid_tracks.append((idx, min_distance)) 

        valid_tracks.sort(key=lambda x: x[1])
        filtered_track_index = [idx for idx, _ in valid_tracks[:max_tracks]]

        removed_count = len(track_index) - len(filtered_track_index) 

        if removed_count > 0 and self.verbose: print(f"Removed {removed_count} of {len(track_index)} tracks to predict due to distance to ego across all timestamps.")
        return filtered_track_index 
    
    def _remove_tracks_by_distance_to_map(self, track_index: List, track_infos: Dict, map_infos: Dict) -> List:
        """
        Remove distant tracks of valid objects that have no map elements near them
        """
        if len(track_index)==0:
            return track_index
        map_range = self.config["map_range"] / 10 # tenth of map range to ensure sufficient map elements
        # Extract x-y positions for each track (shape: [N_tracks, T, 2])
        tracks = track_infos["trajs"][track_index, :, :2]
        # Extract observed mask for each track (shape: [N_tracks, T]); convert to boolean
        observed = track_infos["trajs"][track_index, :, -1].astype(bool)
        # Extract map points (shape: [M, 2])
        map_points = map_infos["all_polylines"][:, :2]
        tree = cKDTree(map_points)
         # Query distances for all positions in the tracks
        distances = tree.query(tracks.reshape(-1, 2))[0].reshape(len(track_index), -1)
        # For time steps that were not observed, set distance to infinity so they won't affect the min computation
        distances[~observed] = np.inf
        # Compute the minimum distance for each track over only the observed positions
        min_distances = np.min(distances, axis=1)
        # Keep track indices where at least one observed position is within the map_range
        filtered_track_index = [idx for i, idx in enumerate(track_index) if min_distances[i] <= map_range]

        removed_count = len(track_index) - len(filtered_track_index) 
        if removed_count > 0 and self.verbose: print(f"Removed {removed_count} of {len(track_index)} tracks to predict due to distance to map elements.")
        return filtered_track_index
    
    def _remove_tracks_by_type(self, track_index: List, track_infos: Dict) -> List:
        """
        Remove tracks with unselected object types
        """
        if len(track_index)==0:
            return track_index
        selected_types = set(object_type[t] for t in self.config["tracks_to_predict"]["allowed_object_types"])
        filtered_track_index = [idx for i, idx in enumerate(track_index) if track_infos["object_type"][idx] in selected_types]

        removed_count = len(track_index) - len(filtered_track_index) 
        if removed_count > 0 and self.verbose: print(f"Removed {removed_count} of {len(track_index)} tracks to predict with unselected object type")
        return filtered_track_index
    
    def _remove_tracks_by_observation(self, track_index: List, track_infos: Dict) -> List:
        """
        Remove tracks with too short past or future
        """
        if len(track_index)==0:
            return track_index
        past_length = self.config["past_len"]
        past_required = self.config["past_required"]
        future_required = self.config["future_required"]
        observed = track_infos["trajs"][track_index, :, -1].astype(bool)
        observed_past_required = np.sum(observed[:,past_length-past_required:past_length], axis=1)
        observed_future_required = np.sum(observed[:,past_length:past_length+future_required], axis=1)
        filtered_track_index = [idx for i,idx in enumerate(track_index) if (observed_past_required[i]>=past_required) and (observed_future_required[i]>=future_required)]

        removed_count = len(track_index) - len(filtered_track_index) 
        if removed_count > 0 and self.verbose: print(f"Removed {removed_count} of {len(track_index)} tracks to predict due to too short observation interval.")
        return filtered_track_index
    
    def process(self, internal_format: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def postprocess(self, output: List[Dict[str, Any]]) -> List[Dict[str,Any]]:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

