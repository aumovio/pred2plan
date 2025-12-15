'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

from shapely import LineString, Point, Polygon
import numpy as np
import warnings
import hashlib
import torch
from copy import deepcopy

from utils.ep_utils.spline import Spline
from utils.ep_utils.preprocess_utils.converter_utils import waymo_lane_type_converter, waymo_boundary_type_converter

class MapInterpreter():
    def __init__(self, 
                 scenario, 
                 mapel_deg):
        
        self.mapel_deg = mapel_deg
        #self.map_infos = self.get_map_info(scenario)
        
        
    def get_map_features(self, map_data, global_map, origin = None, R = None, radius = 300):
        # initialization
        lane_cps_list = [] #np.zeros((num_lanes, 3, 4, 2), dtype=float)
        lane_type_list = [] #np.zeros((num_lanes), dtype=np.uint8)        

        cw_cps_list = [] #np.zeros((num_cross_walks, 3, self.mapel_deg+1, 2), dtype=np.float32)
        cw_type_list = [] #np.zeros((num_cross_walks), dtype=np.uint8) 
        
        for k, v in map_data.items():
            polyline_type_ = self.polyline_type[v['type']]
            
            if polyline_type_ == 0:
                continue
            
            elif polyline_type_ in [1, 2, 3]: # lane
                pts = v['polyline'][:, :2] # [num_points, 2]
                
#                 if pts.shape[0] < 2: # too less points
#                     continue

#                 dist = np.linalg.norm(origin - pts[pts.shape[0]//2])
#                 if dist > radius:
#                     continue
                
                rounded_pts = np.round(pts, decimals=2)
                hash_id = hashlib.md5(rounded_pts.tobytes()).hexdigest()
                
                try:
                    cps_list = []
                    if hash_id in global_map.keys(): # already processed
                        cps_list = global_map[hash_id]
                    else:
                        map_utils.recurrent_fit_line(pts=pts, cps_list=cps_list, degree=self.mapel_deg)
                        global_map[hash_id] = deepcopy(to_float(cps_list))
                        
                    lane_cps_list = lane_cps_list+ cps_list
                    l_type = polyline_type_
                    type_list = [l_type for _ in range(len(cps_list))]
                    lane_type_list = lane_type_list + type_list
                except:
                    if hash_id in global_map.keys(): # already processed
                        cps_list = global_map[hash_id]
                    else:
                        cps_list = [fit_line(pts, degree = self.mapel_deg, no_clean = True, use_borgespastva=True)]
                        global_map[hash_id] = deepcopy(to_float(cps_list))

                    lane_cps_list = lane_cps_list+ cps_list
                    lane_type_list.append(l_type)
                
                
            elif polyline_type_ == 18: # crosswalk
                edge_1, edge_2 = map_utils.find_edges(v['polygon'][:, :2])
                center = (edge_1 + edge_2)/2.
                
                # dist = np.linalg.norm(origin - center[center.shape[0]//2])
                # if dist > radius:
                #     continue

                cw_cps = map_utils.fit_line(center, degree = self.mapel_deg, use_borgespastva = False, num_sample_point=4)
                cw_cps_reverse = cw_cps[::-1]
                cw_cps_list.append(cw_cps)
                cw_cps_list.append(cw_cps_reverse)

                cw_type_list.append(4)
                cw_type_list.append(4)

                          

        num_lanes = len(lane_cps_list)
        num_cws = len(cw_cps_list)

        lane_segment_ids = torch.zeros(num_lanes, dtype=torch.long)
        lane_cps_list = torch.tensor(np.array(lane_cps_list), dtype=torch.float32)
        
        if origin is not None and R is not None:
            lane_cps_list = (lane_cps_list - origin) @ R 

        lane_type_list = torch.tensor(np.array(lane_type_list), dtype=torch.long)


        cross_walk_ids = torch.zeros(num_cws, dtype=torch.long)
        cw_cps_list = torch.tensor(np.array(cw_cps_list), dtype=torch.float32)
        if num_cws >0 and origin is not None and R is not None:
            cw_cps_list = (cw_cps_list-origin)@R

        cw_type_list = torch.tensor(np.array(cw_type_list), dtype=torch.long)

        mapel_ids = torch.concat((lane_segment_ids, cross_walk_ids), dim=0)
        mapel_cps = torch.concat((lane_cps_list, cw_cps_list), dim=0)
        mapel_types = torch.concat((lane_type_list, cw_type_list), dim=0)
        num_mapels = num_lanes + num_cws

        map_data = {
                'mapel_ids': mapel_ids,
                'mapel_cps': mapel_cps,
                'mapel_types': mapel_types,
                'num_nodes': num_mapels,
                }                

        return map_data

            
################################ centerline ###########################################
def recurrent_fit_line(pts, cps_list, degree, current_iter =0, max_iter = 3):
    num_pts = pts.shape[0]
    lane_cps = fit_line(pts, degree=degree, use_borgespastva=True)

    if current_iter == max_iter:
        cps_list.append(lane_cps)
        return

    fit_error = np.linalg.norm(pts[[0, -1]] - lane_cps[[0, -1]], axis=-1)

    if np.max(fit_error) > 0.1 and num_pts >= 8:
        recurrent_fit_line(pts[:((num_pts // 2) +1)], cps_list, degree=degree, current_iter=current_iter + 1)
        recurrent_fit_line(pts[(num_pts // 2):], cps_list, degree=degree, current_iter=current_iter + 1)
    else:
        cps_list.append(lane_cps)
        return
    

def fit_line(line: np.ndarray,
             degree: int,
             use_borgespastva = False,
             num_sample_point = 12,
             maxiter = 2,
             no_clean = False):
    '''
    fit line and find control points.
    
    parameter:
        - line: [N, 2]
        
    return:
        - resampled (interpolated) line [deg + 1,2]
    '''
    if line.shape[0] == 2 or no_clean:
        l = resample_line(line, num_sample_point)
    else:
        l = resample_line(clean_lines(line)[0], num_sample_point) #av2.geometry.interpolate.interp_arc(num_sample_point, line)
        
    s = Spline(degree)
    cps = s.find_control_points(l)
    
    if use_borgespastva:
        t0 = s.initial_guess_from_control_points(l, cps)
        t, converged , errors = s.borgespastva(l, k = degree, t0 = t0, maxiter=maxiter)
        cps = s.find_control_points(l, t=t)
        
    return cps


def resample_line(line: np.ndarray, num_sample_point = 12):
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


def clean_lines(lines):
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

def angle_between_vectors(v1, v2):
    return np.arccos(v1@v2.T / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

def side_to_directed_lineseg(
        query_point,
        start_point,
        end_point) -> str:
    cond = np.cross((end_point - start_point), (query_point - start_point))
    if cond > 0:
        return 'LEFT'
    elif cond < 0:
        return 'RIGHT'
    else:
        return 'CENTER'
    

def line_string_to_xy(ls):
    x, y = ls.coords.xy
    return np.vstack([x,y]).T

def transform_cw_id(cw_id, additional_id):
    return int(str(cw_id) + str(cw_id) + str(additional_id))

def find_edges(xy):
    if xy.shape[0] != 4:
        polygon = Polygon(xy)
        rect = polygon.minimum_rotated_rectangle
        x, y = rect.boundary.xy
        # xy = np.concatenate([np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)], axis=1)
        xy = np.concatenate([np.expand_dims(x[:-1], axis=1), np.expand_dims(y[:-1], axis=1)], axis=1)
    
    dist_1 = np.linalg.norm(xy[0] - xy[1], axis=-1)
    dist_2 = np.linalg.norm(xy[1] - xy[2], axis=-1)
    
    if dist_1 >= dist_2:
        return xy[:2], xy[[-1, -2]]
    else:
        return xy[[1,2]], xy[[0,3]]
    
def is_single_point_lane(lines):
    cleaned_lines = clean_lines(lines)
    num_points = np.array([l.shape[0] for l in cleaned_lines])
    return num_points<=1