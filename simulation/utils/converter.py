"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
from typing import List, Tuple
import numpy as np
import pickle
from pathlib import Path

from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from simulation.utils.scenarionet.utils import compute_angular_velocity
from simulation.utils.scenarionet.nuplan.type import get_traffic_obj_type, NuPlanEgoType
from simulation.utils.scenarionet.nuplan.utils import convert_nuplan_scenario, parse_ego_vehicle_state, parse_object_state, extract_map_features, parse_ego_vehicle_state_trajectory

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer

from predictor.datasets import build_dataset
from omegaconf import DictConfig
from copy import deepcopy

EGO = "ego"


def convert_and_cache_scenario(scenario, cache_dir, file_name, version="1.1"):
    # ensure cache existence
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / file_name
    # original scenario
    if file_path.exists():
        return file_path
    else:
        predictor_scenario = convert_nuplan_scenario(scenario, version="1.1")
        with open(file_path, "wb") as f: pickle.dump(predictor_scenario, f)
    return file_path

def load_converted_scenario(file_path):
    with open(Path(file_path), "rb") as f: scenario = pickle.load(f)
    return scenario

def nuplan_to_predictor(
        nuplan_scenario,
        predictor_scenario,
        predictor_dataloader,
        obstacle_array,
        iteration,
        history,
        first_ego_state,
        ):
    history_len = predictor_dataloader.config.past_len
    result = deepcopy(predictor_scenario)
    history_tracks = _extract_traffic_for_history(nuplan_scenario, history, first_ego_state, history_len)

    #if history_tracks is not None and iteration < history_len:  # add history to converted scenario
    ids_to_delete = []
    for nuplan_id in result[SD.TRACKS].keys():
        if nuplan_id not in history_tracks.keys():  # only keep tracks we have history for
            ids_to_delete.append(nuplan_id)
            continue
        
        # result[SD.TRACKS][nuplan_id]["state"]["position"] = np.concatenate((history_tracks[nuplan_id]["state"]["position"], result[SD.TRACKS][nuplan_id]["state"]["position"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["heading"]  = np.concatenate((history_tracks[nuplan_id]["state"]["heading"], result[SD.TRACKS][nuplan_id]["state"]["heading"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["velocity"] = np.concatenate((history_tracks[nuplan_id]["state"]["velocity"], result[SD.TRACKS][nuplan_id]["state"]["velocity"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["length"]   = np.concatenate((history_tracks[nuplan_id]["state"]["length"], result[SD.TRACKS][nuplan_id]["state"]["length"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["width"]    = np.concatenate((history_tracks[nuplan_id]["state"]["width"], result[SD.TRACKS][nuplan_id]["state"]["width"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["height"]   = np.concatenate((history_tracks[nuplan_id]["state"]["height"], result[SD.TRACKS][nuplan_id]["state"]["height"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["valid"]   = np.concatenate((history_tracks[nuplan_id]["state"]["valid"], result[SD.TRACKS][nuplan_id]["state"]["valid"][:-history_len]), axis=0)
        start_idx = iteration # if iteration <120 else 0 # no idea why 120 
        len_results = result[SD.TRACKS][nuplan_id]["state"]["height"].shape[0] -history_len +start_idx
        result[SD.TRACKS][nuplan_id]["state"]["position"] = np.concatenate((history_tracks[nuplan_id]["state"]["position"], result[SD.TRACKS][nuplan_id]["state"]["position"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["heading"]  = np.concatenate((history_tracks[nuplan_id]["state"]["heading"], result[SD.TRACKS][nuplan_id]["state"]["heading"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["velocity"] = np.concatenate((history_tracks[nuplan_id]["state"]["velocity"], result[SD.TRACKS][nuplan_id]["state"]["velocity"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["length"]   = np.concatenate((history_tracks[nuplan_id]["state"]["length"], result[SD.TRACKS][nuplan_id]["state"]["length"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["width"]    = np.concatenate((history_tracks[nuplan_id]["state"]["width"], result[SD.TRACKS][nuplan_id]["state"]["width"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["height"]   = np.concatenate((history_tracks[nuplan_id]["state"]["height"], result[SD.TRACKS][nuplan_id]["state"]["height"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["valid"]   = np.concatenate((history_tracks[nuplan_id]["state"]["valid"], result[SD.TRACKS][nuplan_id]["state"]["valid"][start_idx:len_results]), axis=0)
        
        len_results = result[SD.TRACKS][nuplan_id]["state"]["height"].shape[0]
        # making sure trajectories always have len 150
        if len(result[SD.TRACKS][nuplan_id]["state"]["position"]) < 150:
            result[SD.TRACKS][nuplan_id]["state"]["position"] = np.pad(result[SD.TRACKS][nuplan_id]["state"]["position"], ((0, 150 - len_results), (0,0)))
            result[SD.TRACKS][nuplan_id]["state"]["heading"]  = np.pad(result[SD.TRACKS][nuplan_id]["state"]["heading"], ((0, 150 - len_results)))
            result[SD.TRACKS][nuplan_id]["state"]["velocity"] = np.pad(result[SD.TRACKS][nuplan_id]["state"]["velocity"], ((0, 150 - len_results), (0,0)))
            result[SD.TRACKS][nuplan_id]["state"]["length"]   = np.pad(result[SD.TRACKS][nuplan_id]["state"]["length"], ((0, 150 - len_results), (0,0)))
            result[SD.TRACKS][nuplan_id]["state"]["width"]    = np.pad(result[SD.TRACKS][nuplan_id]["state"]["width"], ((0, 150 - len_results), (0,0)))
            result[SD.TRACKS][nuplan_id]["state"]["height"]   = np.pad(result[SD.TRACKS][nuplan_id]["state"]["height"], ((0, 150 -len_results), (0,0)))
            result[SD.TRACKS][nuplan_id]["state"]["valid"]    = np.pad(result[SD.TRACKS][nuplan_id]["state"]["valid"], ((0, 150 - len_results)))
            # result[SD.TRACKS][nuplan_id]["state"]["position"] = np.pad(result[SD.TRACKS][nuplan_id]["state"]["position"], ((0, 150 - iteration+1 - history_len), (0,0)))
            # result[SD.TRACKS][nuplan_id]["state"]["heading"]  = np.pad(result[SD.TRACKS][nuplan_id]["state"]["heading"], ((0, 150 - iteration+1 - history_len)))
            # result[SD.TRACKS][nuplan_id]["state"]["velocity"] = np.pad(result[SD.TRACKS][nuplan_id]["state"]["velocity"], ((0, 150 - iteration+1 - history_len), (0,0)))
            # result[SD.TRACKS][nuplan_id]["state"]["length"]   = np.pad(result[SD.TRACKS][nuplan_id]["state"]["length"], ((0, 150 - iteration+1 - history_len), (0,0)))
            # result[SD.TRACKS][nuplan_id]["state"]["width"]    = np.pad(result[SD.TRACKS][nuplan_id]["state"]["width"], ((0, 150 - iteration+1 - history_len), (0,0)))
            # result[SD.TRACKS][nuplan_id]["state"]["height"]   = np.pad(result[SD.TRACKS][nuplan_id]["state"]["height"], ((0, 150 - iteration+1 - history_len), (0,0)))
            # result[SD.TRACKS][nuplan_id]["state"]["valid"]    = np.pad(result[SD.TRACKS][nuplan_id]["state"]["valid"], ((0, 150 - iteration+1 - history_len)))

    # remove objects without history
    #print(f"track ids to remove: {ids_to_delete}")
    for nuplan_id in ids_to_delete:
        result[SD.TRACKS].pop(nuplan_id)

    # add relevant obstacle track_ids to track_to_predict
    result[SD.METADATA]["tracks_to_predict"] = {}
    for obs in obstacle_array:
        result[SD.METADATA]["tracks_to_predict"][obs.obj_num] = \
                {
                    "track_index": list(result[SD.TRACKS].keys()).index(obs.obj_num),
                    "track_id": obs.obj_num,
                    "difficulty": 0,
                    "object_type": obs.obj_type,
                }

    predictor_input = predictor_dataloader.process_single_scenario(result, starting_frame=0, tracks_to_predict="official")
    return predictor_input


def nuplan_to_scenarionet(
        scenario: NuPlanScenario,
        model_cfg:DictConfig,
        obstacle_array:list,
        iteration:int,
        history:SimulationHistoryBuffer=None,
        first_ego_state:EgoState=None,
        scenario_converted=None,
        ) -> Tuple[dict, set, dict]:
    """ Converts Nuplan Scenario to scenarionet format
        then to batch_dict
        then transforms batch_dict, so ego is centered at (0,0) and drives "to the right"
    """
    history_len = 11  # 22 but we only need 20 + the last one is the current state
    print(f'  got {len(obstacle_array)} obstacles to convert')

    if iteration == 0:  # convert scenario only in the first timestep, then reuse it
        print("> Converting nuPlan scneario to scenarionet")
        result = convert_nuplan_scenario(scenario, version="1.1")
        scenario_converted = deepcopy(result)
    else:
        result = deepcopy(scenario_converted)

    # convert history every time because it changes
    print("> Converting history to scenarionet")
    history_tracks = _extract_traffic_for_history(scenario, history, first_ego_state, history_len)

    #if history_tracks is not None and iteration < history_len:  # add history to converted scenario
    ids_to_delete = []
    for nuplan_id in result[SD.TRACKS].keys():
        if nuplan_id not in history_tracks.keys():  # only keep tracks we have history for
            ids_to_delete.append(nuplan_id)
            continue
        
        # result[SD.TRACKS][nuplan_id]["state"]["position"] = np.concatenate((history_tracks[nuplan_id]["state"]["position"], result[SD.TRACKS][nuplan_id]["state"]["position"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["heading"]  = np.concatenate((history_tracks[nuplan_id]["state"]["heading"], result[SD.TRACKS][nuplan_id]["state"]["heading"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["velocity"] = np.concatenate((history_tracks[nuplan_id]["state"]["velocity"], result[SD.TRACKS][nuplan_id]["state"]["velocity"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["length"]   = np.concatenate((history_tracks[nuplan_id]["state"]["length"], result[SD.TRACKS][nuplan_id]["state"]["length"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["width"]    = np.concatenate((history_tracks[nuplan_id]["state"]["width"], result[SD.TRACKS][nuplan_id]["state"]["width"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["height"]   = np.concatenate((history_tracks[nuplan_id]["state"]["height"], result[SD.TRACKS][nuplan_id]["state"]["height"][:-history_len]), axis=0)
        # result[SD.TRACKS][nuplan_id]["state"]["valid"]   = np.concatenate((history_tracks[nuplan_id]["state"]["valid"], result[SD.TRACKS][nuplan_id]["state"]["valid"][:-history_len]), axis=0)
        start_idx = iteration # if iteration <120 else 0 # no idea why 120 
        len_results = result[SD.TRACKS][nuplan_id]["state"]["height"].shape[0] -history_len +start_idx
        result[SD.TRACKS][nuplan_id]["state"]["position"] = np.concatenate((history_tracks[nuplan_id]["state"]["position"], result[SD.TRACKS][nuplan_id]["state"]["position"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["heading"]  = np.concatenate((history_tracks[nuplan_id]["state"]["heading"], result[SD.TRACKS][nuplan_id]["state"]["heading"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["velocity"] = np.concatenate((history_tracks[nuplan_id]["state"]["velocity"], result[SD.TRACKS][nuplan_id]["state"]["velocity"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["length"]   = np.concatenate((history_tracks[nuplan_id]["state"]["length"], result[SD.TRACKS][nuplan_id]["state"]["length"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["width"]    = np.concatenate((history_tracks[nuplan_id]["state"]["width"], result[SD.TRACKS][nuplan_id]["state"]["width"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["height"]   = np.concatenate((history_tracks[nuplan_id]["state"]["height"], result[SD.TRACKS][nuplan_id]["state"]["height"][start_idx:len_results]), axis=0)
        result[SD.TRACKS][nuplan_id]["state"]["valid"]   = np.concatenate((history_tracks[nuplan_id]["state"]["valid"], result[SD.TRACKS][nuplan_id]["state"]["valid"][start_idx:len_results]), axis=0)
        
        len_results = result[SD.TRACKS][nuplan_id]["state"]["height"].shape[0]
        # making sure trajectories always have len 150
        if len(result[SD.TRACKS][nuplan_id]["state"]["position"]) < 150:
            result[SD.TRACKS][nuplan_id]["state"]["position"] = np.pad(result[SD.TRACKS][nuplan_id]["state"]["position"], ((0, 150 - len_results), (0,0)))
            result[SD.TRACKS][nuplan_id]["state"]["heading"]  = np.pad(result[SD.TRACKS][nuplan_id]["state"]["heading"], ((0, 150 - len_results)))
            result[SD.TRACKS][nuplan_id]["state"]["velocity"] = np.pad(result[SD.TRACKS][nuplan_id]["state"]["velocity"], ((0, 150 - len_results), (0,0)))
            result[SD.TRACKS][nuplan_id]["state"]["length"]   = np.pad(result[SD.TRACKS][nuplan_id]["state"]["length"], ((0, 150 - len_results), (0,0)))
            result[SD.TRACKS][nuplan_id]["state"]["width"]    = np.pad(result[SD.TRACKS][nuplan_id]["state"]["width"], ((0, 150 - len_results), (0,0)))
            result[SD.TRACKS][nuplan_id]["state"]["height"]   = np.pad(result[SD.TRACKS][nuplan_id]["state"]["height"], ((0, 150 -len_results), (0,0)))
            result[SD.TRACKS][nuplan_id]["state"]["valid"]    = np.pad(result[SD.TRACKS][nuplan_id]["state"]["valid"], ((0, 150 - len_results)))
            # result[SD.TRACKS][nuplan_id]["state"]["position"] = np.pad(result[SD.TRACKS][nuplan_id]["state"]["position"], ((0, 150 - iteration+1 - history_len), (0,0)))
            # result[SD.TRACKS][nuplan_id]["state"]["heading"]  = np.pad(result[SD.TRACKS][nuplan_id]["state"]["heading"], ((0, 150 - iteration+1 - history_len)))
            # result[SD.TRACKS][nuplan_id]["state"]["velocity"] = np.pad(result[SD.TRACKS][nuplan_id]["state"]["velocity"], ((0, 150 - iteration+1 - history_len), (0,0)))
            # result[SD.TRACKS][nuplan_id]["state"]["length"]   = np.pad(result[SD.TRACKS][nuplan_id]["state"]["length"], ((0, 150 - iteration+1 - history_len), (0,0)))
            # result[SD.TRACKS][nuplan_id]["state"]["width"]    = np.pad(result[SD.TRACKS][nuplan_id]["state"]["width"], ((0, 150 - iteration+1 - history_len), (0,0)))
            # result[SD.TRACKS][nuplan_id]["state"]["height"]   = np.pad(result[SD.TRACKS][nuplan_id]["state"]["height"], ((0, 150 - iteration+1 - history_len), (0,0)))
            # result[SD.TRACKS][nuplan_id]["state"]["valid"]    = np.pad(result[SD.TRACKS][nuplan_id]["state"]["valid"], ((0, 150 - iteration+1 - history_len)))

    # remove objects without history
    #print(f"track ids to remove: {ids_to_delete}")
    for nuplan_id in ids_to_delete:
        result[SD.TRACKS].pop(nuplan_id)

    # add relevant obstacle track_ids to track_to_predict
    result[SD.METADATA]["tracks_to_predict"] = {}
    for obs in obstacle_array:
        result[SD.METADATA]["tracks_to_predict"][obs.obj_num] = \
                {
                    "track_index": list(result[SD.TRACKS].keys()).index(obs.obj_num),
                    "track_id": obs.obj_num,
                    "difficulty": 0,
                    "object_type": obs.obj_type,
                }

    # build batch_dict
    dataset = build_dataset(model_cfg, split="train_val", load_data_from_file=False)
    print(f"> Creating batch_dict")
    if iteration < history_len:
        batch_dict = dataset.process_single_scenario(result, 0) #iteration)
    else:
        batch_dict = dataset.process_single_scenario(result, 0) #iteration-history_len)

    return batch_dict, scenario_converted


def _extract_traffic_for_history(scenario:NuPlanScenario, history:SimulationHistoryBuffer, first_ego_state: EgoState, history_len) -> dict:
    """ Method: from scenarionet.converter.nuplan.utils import extract_traffic
        converts nuplan scenario history to scenarionet format and adds it to converted scenario
    """
    state = scenario.get_ego_state_at_iteration(0)
    center = [state.waypoint.x, state.waypoint.y]

    detection_ret = []
    all_objs = set()
    all_objs.add(EGO)
    
    for frame_data in history.observations[22-history_len:]:  # skip oldest timepoint # TODO: 22 is nuplan history buffer length
        new_frame_data = {}
        for obj in frame_data.tracked_objects:
            new_frame_data[obj.track_token] = obj
            all_objs.add(obj.track_token)
        detection_ret.append(new_frame_data)

    tracks = {
        k: dict(
            type=MetaDriveType.UNSET,
            state=dict(
                position=np.zeros(shape=(history_len, 3)),
                heading=np.zeros(shape=(history_len, )),
                velocity=np.zeros(shape=(history_len, 2)),
                valid=np.zeros(shape=(history_len, )),
                length=np.zeros(shape=(history_len, 1)),
                width=np.zeros(shape=(history_len, 1)),
                height=np.zeros(shape=(history_len, 1))
            ),
            metadata=dict(track_length=history_len, nuplan_type=None, type=None, object_id=k, nuplan_id=k)
        )
        for k in list(all_objs)
    }

    tracks_to_remove = set()

    for frame_idx, frame in enumerate(detection_ret):
        for nuplan_id, obj_state, in frame.items():
            assert isinstance(obj_state, Agent) or isinstance(obj_state, StaticObject)
            obj_type = get_traffic_obj_type(obj_state.tracked_object_type)
            if obj_type is None:
                tracks_to_remove.add(nuplan_id)
                continue
            tracks[nuplan_id][SD.TYPE] = obj_type
            if tracks[nuplan_id][SD.METADATA]["nuplan_type"] is None:
                tracks[nuplan_id][SD.METADATA]["nuplan_type"] = int(obj_state.tracked_object_type)
                tracks[nuplan_id][SD.METADATA]["type"] = obj_type

            state = parse_object_state(obj_state, center)
            tracks[nuplan_id]["state"]["position"][frame_idx] = [state["position"][0], state["position"][1], 0.0]
            tracks[nuplan_id]["state"]["heading"][frame_idx] = state["heading"]
            tracks[nuplan_id]["state"]["velocity"][frame_idx] = state["velocity"]
            tracks[nuplan_id]["state"]["valid"][frame_idx] = 1
            tracks[nuplan_id]["state"]["length"][frame_idx] = state["length"]
            tracks[nuplan_id]["state"]["width"][frame_idx] = state["width"]
            tracks[nuplan_id]["state"]["height"][frame_idx] = state["height"]

    for track in list(tracks_to_remove):
        tracks.pop(track)

    data = [parse_ego_vehicle_state(first_ego_state, center) for i in range(history_len)]
    for i in range(len(data) - 1):
        data[i]["angular_velocity"] = compute_angular_velocity(
            initial_heading=data[i]["heading"], final_heading=data[i + 1]["heading"], dt=scenario.database_interval
        )
    sdc_traj = data

    ego_track = tracks[EGO]

    for frame_idx, obj_state in enumerate(sdc_traj):
        obj_type = MetaDriveType.VEHICLE
        ego_track[SD.TYPE] = obj_type
        if ego_track[SD.METADATA]["nuplan_type"] is None:
            ego_track[SD.METADATA]["nuplan_type"] = int(NuPlanEgoType)
            ego_track[SD.METADATA]["type"] = obj_type
        state = obj_state
        ego_track["state"]["position"][frame_idx] = [state["position"][0], state["position"][1], 0.0]
        ego_track["state"]["valid"][frame_idx] = 1
        ego_track["state"]["heading"][frame_idx] = state["heading"]
        # this velocity is in ego car frame, abort
        # ego_track["state"]["velocity"][frame_idx] = state["velocity"]

        ego_track["state"]["length"][frame_idx] = state["length"]
        ego_track["state"]["width"][frame_idx] = state["width"]
        ego_track["state"]["height"][frame_idx] = state["height"]

    # get velocity here
    vel = ego_track["state"]["position"][1:] - ego_track["state"]["position"][:-1]
    ego_track["state"]["velocity"][:-1] = vel[..., :2] / 0.1
    ego_track["state"]["velocity"][-1] = ego_track["state"]["velocity"][-2]

    # check
    assert EGO in tracks
    for track_id in tracks:
        assert tracks[track_id][SD.TYPE] != MetaDriveType.UNSET

    return tracks