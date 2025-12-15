'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import json
import numpy as np
import pandas as pd

    

def list_intersection(l1, l2):
    return list(set(l1) & set(l2))


def list_slice(l, indicies):
    assert isinstance(l,list)
    return [l[i] for i in indicies]


def waymo_protobuf_to_dataframe(scenario, hist_len, min_obs_len):
     # convert to DataFrame
        data_list = [
                {
                 'track_id': track.id,
                 'object_type': track.object_type,
                 'valid': state.valid,
                 'position_x': state.center_x,
                 'position_y': state.center_y,
                 'heading': state.heading,
                 'velocity_x': state.velocity_x,
                 'velocity_y': state.velocity_y,
                 'timestep': timestep,
                 }
                for track in scenario.tracks for (timestep, state) in enumerate(track.states)]
      

        df = pd.DataFrame(data_list)
        historical_df = df[df['timestep'] < hist_len]
        timesteps = list(np.sort(df['timestep'].unique()))
        
        actor_ids = list(historical_df['track_id'].unique())
        actor_ids = list(filter(lambda actor_id: np.sum(historical_df[historical_df['track_id'] == actor_id]['valid'])>=min_obs_len, actor_ids))
        historical_df = historical_df[historical_df['track_id'].isin(actor_ids)]
        df = df[df['track_id'].isin(actor_ids)]
        
        av_id = scenario.tracks[scenario.sdc_track_index].id
        to_predict_agt_id = [scenario.tracks[tracks_to_predict.track_index].id for tracks_to_predict in scenario.tracks_to_predict if tracks_to_predict.track_index != scenario.sdc_track_index]
        if len(to_predict_agt_id) == 0:
            return {}, False
        
        observed_steps = np.sum(np.array([(df[df['track_id'] == agt_id].iloc)[hist_len-1:]['valid'] for agt_id in to_predict_agt_id]), axis=1)
        valid_idx = np.where(observed_steps == 42)[0] # should be valid at current time and future (1+41)
        if len(valid_idx) == 0: # if no valid focal agent found, return False
            return {}, False
        
        focal_track_id = np.array(to_predict_agt_id)[valid_idx][0]
        scored_track_id = [agt_id for agt_id in to_predict_agt_id if agt_id != focal_track_id]
        
        # DataFrame for AV ang focal agent
        av_df = df[df['track_id'] == av_id].iloc
        av_index = actor_ids.index(av_df[0]['track_id'])
        agt_df = df[df['track_id'] == focal_track_id].iloc
        agent_index = actor_ids.index(agt_df[0]['track_id'])
        
        return {
                'df': df,
                'av_df': av_df,
                'agt_df': agt_df,
                'historical_df': historical_df,
                'actor_ids': actor_ids,
                'av_id': av_id,
                'focal_track_id': focal_track_id,
                'scored_track_id': scored_track_id,
                'av_index': av_index,
                'agent_index': agent_index,
               }, True


def waymo_object_type_converter(waymo_obj_type):
    '''
        convert object type from waymo to argo2
    '''
    if waymo_obj_type == 0: # unset
        return 9 # unknown
    elif waymo_obj_type == 1: # vehicle
        return 0 # vehicle
    elif waymo_obj_type == 2: # pedestrian
        return 1 # pedestrian
    elif waymo_obj_type == 3: # cyclist
        return 3 # cyclist
    else: # others
        return 9 # unknown

def waymo_lane_type_converter(waymo_lane_type):
    '''
        convert lane type from waymo to argo2
    '''
    if waymo_lane_type == 0: # TYPE_UNDEFINED 
        return 0 # 'VEHICLE'
    elif waymo_lane_type == 1: # TYPE_FREEWAY  
        return 0 # 'VEHICLE'
    elif waymo_lane_type == 2: # TYPE_SURFACE_STREET   
        return 0 # 'VEHICLE'
    elif waymo_lane_type == 3: # TYPE_BIKE_LANE    
        return 1 # 'BIKE'
    elif waymo_lane_type == 4: # TYPE_CROSSWALK    
        return 3 # 'PEDESTRIAN'

    
def waymo_boundary_type_converter(waymo_boundary_type):
    '''
        convert boundary type from waymo to argo2
    '''
    if waymo_boundary_type == 0: # TYPE_UNKNOWN
        return 14 # 'UNKNOWN'
    elif waymo_boundary_type == 1: # TYPE_BROKEN_SINGLE_WHITE
        return 2 # 'DASHED_WHITE'
    elif waymo_boundary_type == 2: # TYPE_SOLID_SINGLE_WHITE 
        return 9 #'SOLID_WHITE'
    elif waymo_boundary_type == 3: # TYPE_SOLID_DOUBLE_WHITE 
        return 5 # 'DOUBLE_SOLID_WHITE'
    elif waymo_boundary_type == 4: # TYPE_BROKEN_SINGLE_YELLOW  
        return 3 # 'DASHED_YELLOW'
    elif waymo_boundary_type == 5: # TYPE_BROKEN_DOUBLE_YELLOW   
        return 6 # 'DOUBLE_DASH_YELLOW'
    elif waymo_boundary_type == 6: # TYPE_SOLID_SINGLE_YELLOW   
        return 8 # 'SOLID_YELLOW'
    elif waymo_boundary_type == 7: # TYPE_SOLID_DOUBLE_YELLOW  
        return 4 # 'DOUBLE_SOLID_YELLOW'
    elif waymo_boundary_type == 8: # TYPE_PASSING_DOUBLE_YELLOW   
        return 11 # 'SOLID_DASH_YELLOW'
    elif waymo_boundary_type == 9: # crosswalk
        return 15 #'CROSSWALK'
    elif waymo_boundary_type == 10: # centerline
        return 16 #'CENTERLINE'
    else: # others
        return 14 

