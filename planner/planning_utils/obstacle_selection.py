"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import numpy as np
import functools
import time 

from planner.planning_utils.obstacles import Obstacle
from common_utils.time_tracking import timeit

from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import STATIC_OBJECT_TYPES, AGENT_TYPES

from shapely.geometry import LineString, Point, Polygon
import yaml
import logging 
import secrets

logger = logging.getLogger(__name__)

class ObstacleSelector():
    """
    Select and maintain hard and soft obstacles for scenario-based MPC around the ego vehicle.

    This helper class takes the ego reference path, drivable area geometry, and
    driving corridor information and uses them to filter, rank, and classify
    surrounding obstacles into:
    - hard obstacles, which are modeled explicitly in the MPC as hard (or
      slackened) collision constraints, and
    - soft obstacles, which are handled via soft costs or less strict constraints.

    The selection is based on geometric relations in Frenet coordinates, ego and
    obstacle kinematics, and a “sticky” hysteresis mechanism that stabilizes the
    set of hard obstacles across planning cycles. It also maintains a simple
    estimate of the prevailing traffic speed and a density flag that can be used
    to adapt MPC behavior in dense traffic.

    Parameters
    ----------
    ego_path_linestring : shapely.geometry.LineString
        Geometric centerline of the ego reference path used for projections.
    ego_path : object
        Path object providing `get_state_at_progress` and progress limits used
        to obtain reference pose along the route.
    drivable_polygon : shapely.geometry.Polygon
        Polygon describing the drivable surface; used to reject obstacles
        outside the road.
    driving_corridor : object
        Driving-corridor model providing Frenet corridor envelopes along the
        path (e.g. via `f_dc`).
    c_p : np.ndarray or casadi.DM
        Discrete samples of the reference curve in Frenet/world coordinates,
        used for corridor and obstacle-in-corridor tests.
    roadbound_dist_left : array_like
        Distance from the reference path centerline to the left road boundary
        at each discretization point.
    roadbound_dist_right : array_like
        Distance from the reference path centerline to the right road boundary
        at each discretization point.
    v_limit_at_progress : callable
        Function mapping longitudinal progress to a local speed limit.
    map_api : object
        Map interface capable of querying semantic layers (e.g. crosswalks)
        used to keep pedestrians near crosswalks even if they are almost static.
    params : dict
        Configuration dictionary containing MPC and selection parameters
        (e.g. horizon length, sampling time, max number of hard/soft obstacles).

    Attributes
    ----------
    buffer_obs : dict
        Hysteresis buffer counting how many consecutive frames each candidate
        obstacle has been selected, used for sticky hard-obstacle selection.
    v_traffic : float or None
        Current estimate of traffic flow velocity based on surrounding vehicles.
    traffic_is_dense : bool
        Flag indicating whether traffic around the ego is considered dense.
    dummy_obstacle : Obstacle
        Placeholder obstacle used to pad lists when fewer real obstacles than
        required are available.
    """
    def __init__(self, 
                 ego_path_linestring,
                 ego_path,
                 drivable_polygon,
                 driving_corridor,
                 c_p,
                 roadbound_dist_left,
                 roadbound_dist_right,
                 v_limit_at_progress,
                 map_api,
                 params,
                 ):
        self.ego_path_linestring = ego_path_linestring
        self.ego_path = ego_path
        self.drivable_polygon = drivable_polygon.buffer(0.1)
        self.driving_corridor = driving_corridor
        self.c_p = c_p
        self.roadbound_dist_left = roadbound_dist_left
        self.roadbound_dist_right = roadbound_dist_right
        self.v_limit_at_progress = v_limit_at_progress
        self.map_api = map_api
        self.params = params

        self.buffer_obs = {}
        self.v_traffic = None
        self.traffic_is_dense = False
        dummy_obstacle = Obstacle(
                x = -1000,
                y = -1000,
                psi = 0,
                l = 6, #agents[i].box.dimensions.length,
                w = 2, #agents[i].box.dimensions.width,
                v = 0,
                obj_num = secrets.token_hex(6), #i+1,
                obj_type = TrackedObjectType.VEHICLE,
                S=self.params["S"],
                is_dummy = True
            )
        dummy_obstacle.prediction(self.params["N"], self.params["dt"])
        self.dummy_obstacle = dummy_obstacle
        

    def update_obstacle_selection(self, obstacles, Kh, Ks, max_count=2):
        """
        Sticky selection with hysteresis (previously selected hard obstacles stay for one more frame if still in set)
        """
        buffer = self.buffer_obs
        obstacle_ids = [obs.obj_num for obs in obstacles if not obs.is_static]
        new_candidate_ids = obstacle_ids[:Kh]

        # count up new candidates
        for oid in new_candidate_ids:
            buffer[oid] = min(buffer.get(oid, 0) + 1, max_count)
            
        # count down old candidates
        for oid in list(buffer.keys()):
            if oid not in new_candidate_ids:
                buffer[oid] = max(buffer[oid] - 1, 0)
            # might be better to check if the id is still in all present obstacles, not just this subset
            # prune if disappeared entirely or decayed to zero
            if buffer[oid] <= 0 or oid not in obstacle_ids:
                del buffer[oid]
        # implicitely, older entries win ties
        sorted_ids = sorted(buffer.keys(), key=lambda id: buffer[id], reverse=True)
        hard_obs = [obs for obs in obstacles if obs.obj_num in sorted_ids and not obs.is_static][:Kh]
        soft_obs = [obs for obs in obstacles if obs not in hard_obs][:Ks] #obs not in hard_obs and obs.obj_type!=TrackedObjectType.VEHICLE

        self.buffer_obs = buffer
        return hard_obs, soft_obs
    
    def pad_obstacle_selection(self, obstacles, reservoir, K):
        # if obstacles smaller K and reservoir existing than take first from reservoir
        while len(obstacles) < K and reservoir:
            obs = reservoir.pop(0)
            obstacles.append(obs)
        # if reservoir empty and obstacles still smaller, make dummies
        while len(obstacles) < K:
            obstacles.append(self.dummy_obstacle)
        return obstacles

    @timeit
    def select_obstacles(self,
                         ego_state,
                         obstacles,
                         stop_lines,
                         history,
                         ):
        logger.info(f"Selecting hard and soft constraint obstacles...")
        start_progress = self.ego_path.get_start_progress()
        end_progress = self.ego_path.get_end_progress()
        ego_speed = min(float(ego_state.dynamic_car_state.speed), self.params["v_max"])
        ego_phi = ego_state.center.heading
        ego_token = ego_state.agent.track_token
        ego_x, ego_y = ego_state.center.point.array
        ego_xref, ego_yref, ego_phiref, ego_progress, ego_lat_delta = self.compute_projection_to_ref(ego_x, ego_y)

        dynamic_obstacles = obstacles.get_agents()
        static_obstacles = obstacles.get_static_objects()
        obstacles = dynamic_obstacles + static_obstacles
        
        obstacles, non_participating_obstacles = self.remove_non_participating_obstacles(ego_token, obstacles)
        # up until here we dealt with the standard nuplan obstacle lists, now we transform these to constant velocity predictions 
        obstacles = self.get_obstacle_objects_with_predictions(obstacles) + stop_lines
        obstacles = self.sort_obstacles_by_ego_distance(ego_x, ego_y, ego_phi, ego_speed, obstacles)
        obstacles = self.filter_generic_obstacle_objects(obstacles)

        leader_obs_in_lane, follower_obs_in_lane = self.filter_same_lane_obstacles(ego_x, ego_y, ego_phiref, ego_progress, end_progress, ego_lat_delta, obstacles)
        obstacles = [obs for obs in obstacles if obs not in follower_obs_in_lane]

        leader_obs_token = leader_obs_in_lane.obj_num if leader_obs_in_lane else None
        obstacles_in_driving_corridor, obstacles_outside_driving_corridor = self.filter_obstacles_in_driving_corridor(ego_state, start_progress, end_progress, ego_progress, ego_speed, ego_lat_delta, obstacles, leader_obs_token)
        hard_obs, soft_obs = self.update_obstacle_selection(obstacles_in_driving_corridor, Kh=self.params["MAX_HARD_OBS"], Ks=self.params["MAX_SOFT_OBS"])
        
        self.populate_obstacle_history(hard_obs + soft_obs + obstacles_outside_driving_corridor, history)
        self.v_traffic = self.compute_v_traffic(hard_obs + soft_obs, leader_obs_token)
        self.traffic_is_dense = self.check_traffic_density(ego_state, hard_obs + soft_obs, leader_obs_token)

        hard_obs = self.pad_obstacle_selection(hard_obs, [obs for obs in obstacles_outside_driving_corridor if not obs.is_static], K=self.params["MAX_HARD_OBS"])
        soft_obs = self.pad_obstacle_selection(soft_obs, [obs for obs in obstacles_outside_driving_corridor if obs not in hard_obs], K=self.params["MAX_SOFT_OBS"]) #obs not in hard_obs and obs.obj_type!=TrackedObjectType.VEHICLE

        # self.v_traffic = self.compute_v_traffic(obstacles_in_driving_corridor, leader_obs_in_lane)
        logger.info(f"Selecting hard and soft constraint obstacles... DONE!")
        return hard_obs, soft_obs, leader_obs_in_lane
        
    def populate_obstacle_history(self, obstacles, history):
        target_tokens = set(obstacle.obj_num for obstacle in obstacles)
        history_obstacles = [{obj.metadata.track_token: obj for obj in t.tracked_objects if obj.metadata.track_token in target_tokens} for t in history.observation_buffer]
        for obs in obstacles:
            obs.populate_history_from_buffer(history_obstacles)

    def compute_v_traffic(self, obstacles, leader_obs_token):
        PARKING_VELOCITY = 0.3 # in m/s
        DEFAULT_VELOCITY = 13.41 # in m/s
        relevant_obstacles = []
        for obs in obstacles:
            if obs.obj_type == TrackedObjectType.VEHICLE:
                if not all(abs(v) < 0.3 for v in obs.v_hist):
                    relevant_obstacles.append(obs)
        if not relevant_obstacles:
            return self.v_traffic if self.v_traffic else DEFAULT_VELOCITY

        leader_obstacle = [obs for obs in relevant_obstacles if obs.obj_num == leader_obs_token] if leader_obs_token else []
        if leader_obstacle and leader_obstacle[0].obj_type == TrackedObjectType.VEHICLE:
            v_leader = leader_obstacle[0].v
        else:
            v_leader = None
        # LOWER_BOUND = self.params["v_max"]/5 # in m/s
        velocities = np.asarray(sorted([obs.v for obs in relevant_obstacles], reverse=True))
        lo, hi = np.quantile(velocities, [0.20, 0.80])
        # velocities = np.clip(velocities, lo, hi)
        v_traffic = hi + PARKING_VELOCITY
        # v_traffic = min(v_traffic, LOWER_BOUND)
        # smoothing to prevent sudden jumps
        v_traffic = max(v_traffic, v_leader) if v_leader else v_traffic
        v_traffic = (self.v_traffic + v_traffic)/2 if self.v_traffic else v_traffic
        logger.debug(f"Traffic flow velocity {v_traffic} of {len(relevant_obstacles)} relevant obstacles")
        return v_traffic
    
    def check_traffic_density(self, ego_state, obstacles, leader_obs_token):
        LANE_WIDTH = 3.5
        NUM_DENSE_OBS = 5

        relevant_obstacles = [obs for obs in obstacles if obs.obj_type == TrackedObjectType.VEHICLE]
        if not relevant_obstacles:
            logger.debug(f"Traffic dense: {False} - No relevant obstacles.")
            return False
        ego_x, ego_y = ego_state.center.point.array
        _, _, phi_ref, ego_progress, ego_lat_delta = self.compute_projection_to_ref(ego_x, ego_y)

        idx = int(self.params["M"] * ego_progress / self.params["s_final"])
        idx = max(0, min(self.params["M"] - 1, idx))  # clamp to [0, M-1]
        space_left = max(0.0, self.roadbound_dist_left[idx] - ego_lat_delta)
        space_right = max(0.0, self.roadbound_dist_right[idx] + ego_lat_delta)
        margin_left = min(space_left, 1.5 * LANE_WIDTH)
        margin_right = min(space_right, 1.5 * LANE_WIDTH)
        num_lanes = np.ceil((margin_left + margin_right)/ LANE_WIDTH)
        corridor_grid = np.arange(-LANE_WIDTH, LANE_WIDTH+1e-6, LANE_WIDTH)

        lateral_points = []
        for obs in relevant_obstacles:
            _, _, _, obs_progress, obs_lat_delta = self.compute_projection_to_ref(obs.x, obs.y)
            rel_lat_delta = obs_lat_delta - ego_lat_delta
            rel_progress = obs_progress - ego_progress
            if -10 < rel_progress < 60:
                lateral_points.append(rel_lat_delta)
        
        if len(lateral_points) > 0:
            lateral_points = np.array(lateral_points)
            diffs = np.abs(corridor_grid[:,None] - lateral_points[None, :])
            closest_centerline = np.min(diffs, axis=1)
            occupied_lanes = closest_centerline <= (LANE_WIDTH / 2.0 + 0.2)
            num_occupied_lanes = occupied_lanes.sum()
        else:
            num_occupied_lanes = 0

        leader_obstacle = [obs for obs in relevant_obstacles if obs.obj_num == leader_obs_token] if leader_obs_token else []
        has_vehicle_lead = (bool(leader_obstacle) and leader_obstacle[0].obj_type is TrackedObjectType.VEHICLE)
        # if same lane and at least other lane occupied and we have 5 relevant obstacles, then consider traffic dense
        traffic_is_dense = (num_occupied_lanes >= num_lanes/2+0.5) and (len(relevant_obstacles) >= NUM_DENSE_OBS) and has_vehicle_lead
        logger.debug(f"Traffic dense: {traffic_is_dense} - {len(relevant_obstacles)} relevant obstacles occupying {num_occupied_lanes}/{num_lanes} lanes.")
        return traffic_is_dense

    def compute_projection_to_ref(self, x, y):
        long_progress = self.ego_path_linestring.project(Point(x, y))
        _, x_ref, y_ref, phi_ref = self.ego_path.get_state_at_progress(min(long_progress, self.ego_path.get_end_progress()))
        lat_delta = -np.sin(phi_ref) * (x - x_ref) + np.cos(phi_ref) * (y - y_ref) # lateral delta of position to reference path
        return x_ref, y_ref, phi_ref, long_progress, lat_delta

    def remove_non_participating_obstacles(self, ego_token, obstacles):
        """
        Removes non-moving obstacles that are not on the drivable area with the exception of pedestrians near a crosswalk
        """
        SEARCH_RADIUS = 2 # in meter
        VELOCITY_THRESHOLD = 0.3 # in meter/seconds
        non_participating_obstacles = set()
        for obs in obstacles:
            pedestrian_or_parking = (obs.tracked_object_type == TrackedObjectType.PEDESTRIAN or obs.velocity.magnitude() < VELOCITY_THRESHOLD)
            inside_driving_polygon = any([self.drivable_polygon.contains(Point(p)) for p in obs.box.all_corners()])
            if  pedestrian_or_parking and not inside_driving_polygon or obs.track_token == ego_token: # TODO: not center but bounding boxes
                if obs.tracked_object_type == TrackedObjectType.PEDESTRIAN:
                    crosswalks = self.map_api.get_proximal_map_objects(Point(obs.center.x,obs.center.y),(SEARCH_RADIUS), [SemanticMapLayer.CROSSWALK])
                    if crosswalks[SemanticMapLayer.CROSSWALK]: # if some crosswalks near the obstacle exist, keep it
                        continue
                non_participating_obstacles.add(obs)
        participating_obstacles = set(obstacles) - non_participating_obstacles
        return participating_obstacles, non_participating_obstacles

    def sort_obstacles_by_ego_distance(self, ego_x, ego_y, ego_phi, ego_speed, obstacles):
        """ sort obstacles by their weighted distance to an offset position in front of the ego's driving direction"""
        FRONT_OFFSET = self.params["length"] + 3*self.params["dt"] * ego_speed # in m # vehicle length as slight offset in front of car + 3 velocity scaled timesteps
        VELOCITY_THRESHOLD = 0.3
        MULTIPLIERS = {
            TrackedObjectType.VEHICLE: 1.0,
            TrackedObjectType.BICYCLE: 2.0,
            TrackedObjectType.PEDESTRIAN: 2.0,
            TrackedObjectType.BARRIER: 1.5,
            TrackedObjectType.GENERIC_OBJECT: 4.0,
        }
        def distance_multiplier(obj_type, velocity):
            if obj_type == TrackedObjectType.VEHICLE:
                if velocity < VELOCITY_THRESHOLD:
                    return 2.0
            return MULTIPLIERS.get(obj_type, 2.0)

        ref_x = ego_x + FRONT_OFFSET*np.cos(ego_phi)
        ref_y = ego_y + FRONT_OFFSET*np.sin(ego_phi)
        obstacles = list(obstacles)
        obstacles.sort(key=lambda obs: float(np.hypot(ref_x - obs.x, ref_y - obs.y)*distance_multiplier(obs.obj_type, obs.v)))
        return obstacles

    def filter_generic_obstacle_objects(self, obstacles):
        # generic obstacles often occur in huge bulks but are rarely relevant
        # we limit their number in addition to distance based ordering to make space for other types of soft obs
        generic = [obs for obs in obstacles if obs.obj_type == TrackedObjectType.GENERIC_OBJECT]

        kept = []
        not_kept = []
        min_dist = 2.0
        r2 = min_dist * min_dist

        for obs in generic:
            if not kept:
                kept.append(obs)
                continue

            # Check distance to all kept points
            too_close = False
            for q in kept:
                if (obs.x - q.x)**2 + (obs.y - q.y)**2 < r2:
                    too_close = True
                    break
            if not too_close:
                kept.append(obs)
            if len(kept) <= self.params["MAX_SOFT_OBS"]/2:
                break
        
        # to make sure ordering is the same
        not_kept = [obs for obs in generic if obs not in kept]
        return [obs for obs in obstacles if obs not in not_kept]

    def get_obstacle_objects_with_predictions(self, obstacles):
        DEFAULT_LENGTH = 1.2 # we had 3.0
        DEFAULT_WIDTH = 1.0 # we had 3.0
        SAFETY_MARGIN = 0.2
        transformed_obstacles = []
        for obs in obstacles:
            transformed_obstacles.append(Obstacle(
                x = obs.center.x,
                y = obs.center.y,
                psi = obs.center.heading,
                l = obs.box.dimensions.length+SAFETY_MARGIN, #max(6,
                w = obs.box.dimensions.width+SAFETY_MARGIN, #max(1.5,obs.box.dimensions.width),
                v = obs.velocity.magnitude(),
                obj_num = obs.metadata.track_token, #i+1,
                obj_type = obs.tracked_object_type,
                S=self.params["S"]
            ))
            
            if transformed_obstacles[-1].obj_type == TrackedObjectType.PEDESTRIAN:
                transformed_obstacles[-1].v = 0
                transformed_obstacles[-1].l = max(DEFAULT_LENGTH,obs.box.dimensions.length)
                transformed_obstacles[-1].w = max(DEFAULT_WIDTH,obs.box.dimensions.width)
            # constant velocity prediction
            transformed_obstacles[-1].prediction(self.params["N"], self.params["dt"]) 
        return transformed_obstacles

    def filter_same_lane_obstacles(self, ego_x, ego_y, phi_ref, ego_progress, end_progress, ego_lat_delta_to_ref, obstacles):
        """
        filter out follower obstacles and the first leading obstacle in the same lane
        """
        # searching all objects in same lane as ego, closest in front is lead
        # all other vehicles in this lane should not be added to relevant obstacles
        CORRIDOR_LATERAL_MARGIN = 10 # in m
        LANE_LATERAL_MARGIN = 1.75 # in m (half the width of a standard lane)
        LEADER_FOLLOWER_MARGIN = -3
         # in m

        follower_obs = []
        leader_obs = None
        for obs in obstacles:
            obs_x = obs.x
            obs_y = obs.y
            obs_lat_delta_to_ego = np.abs(-np.sin(phi_ref) * (obs_x - ego_x) + np.cos(phi_ref) * (obs_y - ego_y)) 
            in_same_corridor = obs_lat_delta_to_ego < CORRIDOR_LATERAL_MARGIN
            on_drivable_area = self.drivable_polygon.contains(Point((obs_x, obs_y)))

            if ego_progress < 5 or (on_drivable_area and in_same_corridor):
                obs_progress = min(self.ego_path_linestring.project(Point(*[obs_x, obs_y])), end_progress)
                _, xo_ref, yo_ref, phio_ref = self.ego_path.get_state_at_progress(obs_progress)
                obs_lat_delta_to_ref = -np.sin(phio_ref) * (obs_x- xo_ref) + np.cos(phio_ref) * (obs_y - yo_ref)

                in_same_lane = np.abs(obs_lat_delta_to_ref - ego_lat_delta_to_ref) < LANE_LATERAL_MARGIN
                in_front = ego_progress - obs_progress < LEADER_FOLLOWER_MARGIN
                in_back = ego_progress - obs_progress > LEADER_FOLLOWER_MARGIN

                if in_same_lane:
                    if in_front and leader_obs is None:
                        leader_obs = obs
                    elif in_back and obs.obj_type != TrackedObjectType.PEDESTRIAN and obs.obj_type not in STATIC_OBJECT_TYPES:
                        follower_obs.append(obs)
        return leader_obs, follower_obs

    def filter_obstacles_in_driving_corridor(self, ego_state, start_progress, end_progress, ego_progress, ego_speed, ego_lat_delta_to_ref, obstacles, leader_obs_token):
        """
        At each timestep, a rectangular corridor is spanned and it is checked if the candidate obstacles are inside this corridor.
        Requires constant velocity predictions for the candidate obstacles. Resulting list length is upper bounded.
        """
        dc_frenet = self.driving_corridor.f_dc(ego_progress , ego_speed,[], self.v_limit_at_progress(ego_progress))
        obstacles_inside_corridor = [obs for obs in obstacles if obs.obj_num == leader_obs_token] if leader_obs_token else []
        # at each
        for i in range(0, self.params["N"], 2):
            # span out a rectangle in frenet coordinates for each time step (ego driving corridor)
            REAR_MARGIN = 5 # rear boundary factor
            FRONT_MARGIN = 15 #10 # front boundary factor
            LATERAL_FACTOR = 0.12 # the more we progress the more lateral deviations we allow
            LATERAL_MARGIN = 4 # to vehicles own geometry
            LAG_ERROR_MARGIN = 2.5

            # rear and front longitudenal boundaries
            min_prog = max(start_progress,float(dc_frenet[i,0].full()) - self.params["rob_r"]*REAR_MARGIN)
            max_prog = min(end_progress,float(dc_frenet[i,1].full()) + self.params["rob_r"]*FRONT_MARGIN)

            # left and right lateral boundaries (ego_lat_delta_to_ref as offset)
            i_min = max(0, int(self.params["M"] * min_prog/self.params["s_final"]))
            i_max = min(self.params["M"] - 1, int(self.params["M"]*max_prog/self.params["s_final"]))
            lat_reach =  (float(dc_frenet[i,1].full()) - ego_progress) * LATERAL_FACTOR #heuristic v_max_lat = k*v_max_long 0.08
            max_width_left = min(max(self.roadbound_dist_left[i_max], self.roadbound_dist_left[i_min]), ego_lat_delta_to_ref + lat_reach+LATERAL_MARGIN*self.params["rob_r"]) 
            max_width_right = min(max(self.roadbound_dist_right[i_max], self.roadbound_dist_right[i_min]), -ego_lat_delta_to_ref + lat_reach+LATERAL_MARGIN*self.params["rob_r"])

            # plot_things(ego_state, obstacles, self.drivable_polygon, self.c_p, self.roadbound_dist_left, self.roadbound_dist_right, dc_frenet, ego_lat_delta_to_ref, ego_progress, self.params)
            for obs in obstacles:
                # TODO: one could accelerate this loop by preemptively sorting negatively lagging obstacles into the obstacles_outside_corridor list
                if obs in obstacles_inside_corridor:
                    continue
                else:
                    # obs_lag_error longitudenal error to frenet rectangle and obs_lat_delta_to_ref lateral deviation on frenet line
                    obs_lag_error, obs_lat_delta, *_  = point_to_curve_lat_lon(self.c_p[i_min:i_max,0], self.c_p[i_min:i_max,1], self.c_p[i_min:i_max,2], obs.x_preds[0][i], obs.y_preds[0][i])      
                    # obs_xref, obs_yref, obs_phiref, obs_progress, obs_lat_delta = self.compute_projection_to_ref(obs.x_preds[0][i], obs.y_preds[0][i])
                    # buffer for intersections with different orientation from ego
                    intersect_left = obs.v_preds[0][i] if fix_angle_jumps([0,(obs.psi_preds[0][i]- self.c_p[i_min,2])])[1] > (np.pi/2-0.2) else 0 
                    intersect_right = obs.v_preds[0][i] if fix_angle_jumps([0,(obs.psi_preds[0][i]- self.c_p[i_min,2])])[1] < (-np.pi/2+0.2) else 0 
                    # check if it is inside rectangular frenet corridor
                    if np.abs(obs_lag_error) < LAG_ERROR_MARGIN and obs_lat_delta < max_width_left + intersect_left and -obs_lat_delta < max_width_right+ intersect_right:
                    #if min_prog < obs_progress < max_prog and obs_lat_delta< max_width_left + intersect_left and -obs_lat_delta < max_width_right+ intersect_right:
                        obstacles_inside_corridor.append(obs)
                        # lateral_points.append(obs_lat_delta)
                        # is_full = len(obstacles_inside_corridor) == self.max_num_hard_obs
            #     if is_full:
            #         break
            # if is_full:
            #     break

        obstacles_outside_corridor = [obs for obs in obstacles if obs not in obstacles_inside_corridor]
        return obstacles_inside_corridor, obstacles_outside_corridor


def fix_angle_jumps(angles):
    diffs = np.diff(angles)
    wrapped_diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    fixed_angles = np.cumsum(np.concatenate(([angles[0]], wrapped_diffs)))
    return fixed_angles


def point_to_curve_lat_lon(x_ref, y_ref, psi_ref, x0, y0):
    dx = x_ref[1:] - x_ref[:-1]
    dy = y_ref[1:] - y_ref[:-1]
    px = x0 - x_ref[:-1]
    py = y0 - y_ref[:-1]
    segment_len_sq = dx**2 + dy**2
    t = (px * dx + py * dy) / segment_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = x_ref[:-1] + t * dx
    proj_y = y_ref[:-1] + t * dy
    dist_sq = (proj_x - x0)**2 + (proj_y - y0)**2
    min_idx = np.argmin(dist_sq)
    x_closest = proj_x[min_idx]
    y_closest = proj_y[min_idx]
    psi_closest = psi_ref[min_idx]

    # delta vector in global frame
    dx_global = x0 - x_closest
    dy_global = y0 - y_closest

    # Rotate into local Frenet frame
    cos_psi = np.cos(psi_closest)
    sin_psi = np.sin(psi_closest)
    s_local =  cos_psi * dx_global + sin_psi * dy_global  # longitudinal
    d_local = -sin_psi * dx_global + cos_psi * dy_global  # lateral

    return s_local, d_local, x_closest, y_closest
