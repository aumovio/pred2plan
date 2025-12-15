"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
# general
from typing import List, Tuple, Any, Dict
import numpy as np
import imageio
import logging 
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from scipy.interpolate import interp1d
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union

import time
import pickle
from pathlib import Path

# for Predictor
from predictor.models import build_model, load_predictor
from predictor.datasets import build_dataset

# for Planner
from planner.planning_utils.ScenarioSelection.DrivingCorridorSelector import DrivingCorridor
from planner.planning_utils.obstacles import Obstacle
from planner.planning_utils.obstacle_selection import ObstacleSelector


# utils for evaluation
from simulation.utils import predictor_utils
from simulation.utils import planner_utils
from simulation.utils import converter
from simulation.utils.nuplan_scenario_render import NuplanScenarioRender
from common_utils.time_tracking import timeit
# nuplan
from nuplan.common.actor_state.ego_state import  EgoState
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2, path_to_linestring
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.path.utils import calculate_progress

from nuplan.common.actor_state.tracked_objects import TrackedObjects, TrackedObjectType
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)


import multiprocessing
from easydict import EasyDict


EGO = "ego"
DEFAULT_SPEED_LIMIT = 13.41 # meters per second, equals roughly 49 km per hour or 30 miles per hour
DEFAULT_LANE_WIDTH = 3.5


class ModularPlanner(AbstractPlanner):

    requires_scenario: bool = True

    def __init__(self, 
                 scenario,
                 planner,
                 predictor,
                 log_dir,
                 cache_dir,
                 visualization,
                 ):
        """ This method initializes the planner with important static information including the high level goal,
            represented as a (x, y, heading) pose, which in practice might be provided a higher level routing system,
            as well as the interface for interacting with relevant map information.
        """
        logger.info(f'Initialize ModularPlanner for scenario {scenario.scenario_name, scenario.scenario_type, scenario.log_name}...')
        debug = False

        self.planner = planner
        self._predictor_cfg = EasyDict(predictor)
        device_idx = self._predictor_cfg.trainer.devices
        self._device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available()  and not debug else "cpu")
        self._cache_dir = cache_dir
        #self.predictor = load_predictor(predictor_cfg, predictor_cfg["trainer"]["resume_ckpt_path"], device=self._device)
        # self.predictor_dataloader = build_dataset(self._predictor_cfg, data_splits=[])
        self.scenario = scenario

        self._predictor_name = self._predictor_cfg["model"]["model_name"]
        self._planner_name = self.planner.__class__.__name__
        self._scenario_id = f"{scenario.scenario_name}_{scenario.log_name}"
        self._predictor_scenario = converter.convert_and_cache_scenario(scenario, cache_dir, f"{self._scenario_id}_scenarionet")

        # self.horizon_seconds = TimePoint(int(self.planner.N * self.planner.dt * 1e6))
        # self.sampling_time = TimePoint(int(self.planner.dt * 1e6))

        if visualization: 
            self.scene_render = NuplanScenarioRender(fps=2)
            self._imgs = []
            self.log_dir = Path(log_dir)
        self.visualization = visualization

        self.timestep_logs = {"metadata": 
                              {"planner_name": self._planner_name,
                               "predictor_name": self._predictor_name,
                               "scenario_name": scenario.scenario_name, 
                               "scenario_log": scenario.log_name,
                               "scenario_type": scenario.scenario_type,
                               }
                            } # a dictionary for each timestep

        self.initialized = False
        logger.info(f'Initialize ModularPlanner for scenario {scenario.scenario_name} with {self._predictor_name}-{self._planner_name} pairing... DONE!')

    @property   
    def predictor_scenario(self):
        with open(Path(self._predictor_scenario), "rb") as f: scenario = pickle.load(f)
        return scenario
    
    # @property
    # def scenario(self):
    #     with open(Path(self._scenario), "rb") as f: scenario = pickle.load(f)
    #     return scenario

    def name(self) -> str:
        """ planner name """
        return self.planner.__class__.__name__
    
    def __getstate__(self):
        state = {k:v for k,v in self.__dict__.items() if k not in ["planner", "obstacle_selector"]}
        return state
    
    def observation_type(self) -> type[Observation]:
        """ This dictates what type of observations the planner will consume to inform it's decision making.
            Options here include Sensors (raw sensor information such as images or pointclouds) and 
            DetectionsTracks (outputs of an earlier perception system designed to consume sensor information and produce meaningful detections).
        """
        return DetectionsTracks
    
    @timeit
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass"""
        self._initialization = initialization
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        self.initialized=False

    def _load_route_dicts(self, route_roadblock_ids: List[str]) -> None:
        """
        Loads roadblock and lane dictionaries of the target route from the map-api.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        # remove repeated ids while remaining order in list
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        self._route_roadblock_dict = {}
        self._route_lane_dict = {}

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane

    def _route_roadblock_correction(self, ego_state: EgoState) -> None:
        """
        Corrects the roadblock route and reloads lane-graph dictionaries.
        :param ego_state: state of the ego vehicle.
        """
        route_roadblock_ids = planner_utils.route_roadblock_correction(
            ego_state, self._map_api, self._route_roadblock_dict
        )
        self._load_route_dicts(route_roadblock_ids)

    @timeit
    def _initialize_ego_reference_path_and_corridor(self, ego_state) -> None:
        """ Initialize the ego reference path and driving corridor from goal route at current ego state"""
        logger.info("Initializing ego reference path and driving corridor...")
        ######### INIT EGO REF PATH ###########
        roadblocks = list(self._route_roadblock_dict.values())
        roadblocks_polygon = [block.polygon for block in roadblocks]
        candidate_lane_edge_ids = list(self._route_lane_dict.keys())
        route_plan, _ = planner_utils._breadth_first_search(ego_state, roadblocks, candidate_lane_edge_ids)
        discrete_path = []
        first_pos_edge = []
        speed_limit_list = []
        for edge in route_plan:
            discrete_path.extend(edge.baseline_path.discrete_path)
            first_pos_edge.append(edge.baseline_path.discrete_path[0])
            speed_limit_list.append(edge.speed_limit_mps or DEFAULT_SPEED_LIMIT)
            
        progress = np.array(calculate_progress(first_pos_edge))
        speed_limits = np.array(speed_limit_list)
        self.true_v_limit_at_progress = interp1d(progress, speed_limits, kind='linear',bounds_error=False, fill_value='extrapolate') 
        uniform_progress = np.linspace(progress[0], progress[-1], 1000)
        smoothed_limits = self.true_v_limit_at_progress(uniform_progress)
        smoothed_limits[2:-2] = np.convolve(smoothed_limits, [0.01, 0.1, 0.2, 0.3, 0.39], mode="same")[2:-2]
        self.soft_v_limit_at_progress = interp1d(uniform_progress, smoothed_limits, kind="linear", bounds_error=False, fill_value="extrapolate")
        self._ego_path = create_path_from_se2(discrete_path)
        self._ego_path_linestring = path_to_linestring(discrete_path)

        ref_lane = self._ego_path
        s_ref =  np.array([point.progress for point in ref_lane.get_sampled_path()[:]]) 
        x_ref =np.interp(np.linspace(0,self.planner.s_final, self.planner.M) , s_ref, np.array([point.x for point in ref_lane.get_sampled_path()[:]])  )
        y_ref =np.interp(np.linspace(0,self.planner.s_final, self.planner.M) , s_ref, np.array([point.y for point in ref_lane.get_sampled_path()[:]]) )
        phi_ref =np.interp(np.linspace(0,self.planner.s_final, self.planner.M)  , s_ref, np.array([point.heading for point in ref_lane.get_sampled_path()[:]]) )
        # fix angle jump
        diffs = np.diff(phi_ref)
        wrapped_diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
        phi_ref= np.cumsum(np.concatenate(([phi_ref[0]], wrapped_diffs)))
        c_p = np.stack(( x_ref, y_ref, phi_ref)).T
        c_p[:,2] = planner_utils.smooth_outliers(c_p[:,2], 1.0)
        self.path_ref = c_p

        ######### INIT EGO REF CORRIDOR ###########
        SLACK_MARGIN_ROADBOUND = -0.1
        EXPAND = 0.2
        SIMPLIFY_TOL = 0.2
        # correct if nuplan given corridor is disconnected
        drivable_corridor = unary_union(roadblocks_polygon).buffer(1e-6)

        if isinstance(drivable_corridor, MultiPolygon):
            new_poly = []
            for i, poly in enumerate(drivable_corridor.geoms):
                new_poly.append(poly.buffer(EXPAND))
            drivable_corridor = unary_union(new_poly).buffer(1e-6)
            drivable_corridor = drivable_corridor.buffer(-EXPAND)

        if isinstance(drivable_corridor, MultiPolygon):
            ref_pol = planner_utils.line_to_centered_polygon(c_p, width=DEFAULT_LANE_WIDTH) # width is default lane width
            drivable_corridor = unary_union(drivable_corridor+[ref_pol]).simplify(SIMPLIFY_TOL)

        if isinstance(drivable_corridor, MultiPolygon):
            new_poly = []
            for i, poly in enumerate(drivable_corridor.geoms):
                new_poly.append(poly.buffer(EXPAND*2))
            drivable_corridor = unary_union(new_poly).buffer(1e-6)
            drivable_corridor = drivable_corridor.buffer(-EXPAND*2)
        self.drivable_corridor = drivable_corridor
        # build boundaries for driving corridor
        """get left and right boundaries"""
        try:
            _, _, self._roadbound_dist_right, self._roadbound_dist_left = \
                planner_utils.get_road_boundaries(
                    self._ego_path, drivable_corridor, SLACK_MARGIN_ROADBOUND, c_p
            )  
        except Exception as e:
            logger.warning(f"Error computing road boundaries: {e}, falling back to default road width")
            self._roadbound_dist_left, self._roadbound_dist_right = DEFAULT_LANE_WIDTH/2*np.ones((self.planner.M,)), DEFAULT_LANE_WIDTH/2*np.ones((self.planner.M,))  
        

        s_uniform = np.linspace(0, self.planner.s_final, self.planner.M)
        self.roadbound_left_at_progress = interp1d(
            s_uniform,
            self._roadbound_dist_left,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        self.roadbound_right_at_progress = interp1d(
            s_uniform,
            self._roadbound_dist_right,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        logger.info("Initializing ego reference path and driving corridor... DONE!")

    def _initialize_obstacle_selector(self):
        """init obstacle selection"""
        dc_obj = DrivingCorridor(
            lane_id=0, mode_id=0, N=self.planner.N, dt=self.planner.dt, v_max=self.planner.v_max, v_min=0,
            a_max=self.planner.a_max, a_min=self.planner.a_min*0.5, f=0
        )
        
        dc_obj.init_casadi_function(0)

        self.obstacle_selector = ObstacleSelector(
            ego_path_linestring = self._ego_path_linestring,
            ego_path = self._ego_path,
            drivable_polygon = self.drivable_corridor,
            driving_corridor = dc_obj,
            c_p = self.path_ref,
            roadbound_dist_left = self._roadbound_dist_left,
            roadbound_dist_right = self._roadbound_dist_right,
            v_limit_at_progress = self.soft_v_limit_at_progress,
            map_api = self._map_api,
            params = self.planner.flattened_params,
        )

    def _initialize_predictor(self):
        self.predictor_dataloader = build_dataset(self._predictor_cfg, data_splits=[])
        self.predictor = load_predictor(self._predictor_cfg, self._predictor_cfg["trainer"]["resume_ckpt_path"], device=self._device)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: List of planner inputs for which trajectory needs to be computed.
        :return: Trajectories representing the planned ego's position in future
        """
        iteration = current_input.iteration
        history = current_input.history
    
        print(f'--------------------------- ITERATION {iteration.index}-----------------------------')
        
        # if self._planner_name == Planner.GroundTruth.value['method']:
        #     print("Ground Truth")
        #     return self.gt_plan(current_input, self.scenario)

        ego_state, observations = current_input.history.current_state
        ego_token = ego_state.agent.track_token

        ## initialize the planner
        if current_input.iteration.index == 0:
            self.planner.x0_ = None
            if not self.initialized:
                self._initialize_predictor()
                self._route_roadblock_correction(ego_state)
                self._initialize_ego_reference_path_and_corridor(ego_state)
                self._initialize_obstacle_selector()
                self.timestep_logs["metadata"]["iterations"] = 0
                self.initialized = True

            self.ego_psi = float(ego_state.rear_axle.heading)
            self.number_of_iterations = self.scenario.get_number_of_iterations()

        """ Prediction """
        stop_lines = []
        traffic_light_status = current_input.traffic_light_data
        tls = {str(tl.lane_connector_id): tl.status for tl in traffic_light_status if tl.status==TrafficLightStatusType.RED}
        for lc_id in tls.keys():
            lc = self._map_api._get_lane_connector(lc_id)
            for stop_line in lc.stop_lines:
                if stop_line.id in [obs.obj_num for obs in stop_lines]: continue
                bbox_center, length, width, psi = planner_utils.oriented_bbox_from_polygon(stop_line.polygon)
                if self.drivable_corridor.contains(Point(bbox_center)):
                    logger.debug(f"Center added stop_line: {bbox_center} and {stop_line.id} with width {width} and length {length} and angle {psi}")
                    stop_lines.append(Obstacle(
                            x = bbox_center[0],
                            y = bbox_center[1],
                            psi = psi,
                            l = length, 
                            w = width, #max(1.5,agent.box.dimensions.width),
                            v = 0,
                            obj_num = stop_line.id, #i+1,
                            obj_type = TrackedObjectType.BARRIER,
                            S=self.planner.flattened_params["S"]
                        ))		
                    stop_lines[-1].prediction(self.planner.flattened_params["N"], self.planner.flattened_params["dt"])

        hard_obs, soft_obs, leader_obs = self.obstacle_selector.select_obstacles(ego_state, observations.tracked_objects, stop_lines, current_input.history)

        hard_obs_to_refine = [obs for obs in hard_obs if obs.is_dummy == False]
        if len(hard_obs_to_refine)>0:
            self.refine_obstacle_predictions(obstacles = hard_obs_to_refine,
                                            iteration = iteration.index,
                                            history = current_input.history,)
        else:
            self.timestep_logs[iteration.index] = {
                "predicted_trajectory": [],
                "predicted_probability": [],
                "predicted_object_ids": [],
                "predicted_track_difficulty": [],
                "predicted_track_type": [],
                "ego_track_difficulty": [],
                "current_velocity": [],
            }
            self.timestep_logs["metadata"]["iterations"] += 1


        """ Planner """
        ego_x = float(ego_state.rear_axle.x) 
        ego_y = float(ego_state.rear_axle.y) 
        self.ego_psi = planner_utils.fix_angle_jumps([self.ego_psi,float(ego_state.rear_axle.heading)])[1] # compute heading changes
        ego_progress = self._ego_path_linestring.project(Point(*ego_state.rear_axle.point.array))#self.ego_progress #
        ego_speed = float(ego_state.dynamic_car_state.speed)
        ego_accel =  0 if current_input.iteration.index==0 and ego_speed<0.2  else float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x) # taking x in ego coordinates
        ego_delta = float(ego_state.tire_steering_angle)

        x0 = np.array([
            ego_x, ego_y, self.ego_psi, ego_speed, ego_accel, ego_delta, ego_progress,
        ]).reshape(-1,1)

        #vmax = max(self.soft_v_limit_at_progress(ego_progress+5), ego_speed+self.planner.dt*(ego_accel-self.planner.dt*6*self.planner.j_max))
        vmax = self.soft_v_limit_at_progress(ego_progress+5)
        if self.obstacle_selector.traffic_is_dense: vmax = max(vmax/3, min(vmax, self.obstacle_selector.v_traffic))
        dc_is_narrow = all([self.roadbound_left_at_progress(ego_progress+i) + self.roadbound_right_at_progress(ego_progress+i) < DEFAULT_LANE_WIDTH for i in range(5,10)])
        if dc_is_narrow: vmax = min(vmax, 3/4*DEFAULT_SPEED_LIMIT) 

        xref = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]).reshape(-1, 1)

        traj_interpolated, u_full, x_full, success, costs = self.planner.solve(
            x0 = x0,
            xref = xref,
            hard_obs = hard_obs,
            soft_obs = soft_obs,
            path = self.path_ref,
            wl = self._roadbound_dist_left,
            wr = self._roadbound_dist_right,
            vmax = vmax,
            pool = None,
            ego_state = ego_state,
            scenario = self.scenario,
            iteration = iteration,
        )
        
        self.timestep_logs[iteration.index]["planner_costs"] = np.array(costs)[0]
        """ Plot """
        roadbound_left, roadbound_right = planner_utils.compute_boundaries(self.path_ref, self._roadbound_dist_left, self._roadbound_dist_right)
        # roadbound_left, roadbound_right = np.array(self._left_boundary.coords), np.array(self._right_boundary.coords)
        if self.visualization:
            self.visualize(current_input, 
                        [obs for obs in hard_obs if not obs.is_dummy], 
                        [obs for obs in soft_obs if not obs.is_dummy], 
                        x_full, 
                        traj_interpolated, 
                        ego_progress,
                        vmax,
                        self.path_ref,
                        roadbound_left, 
                        roadbound_right)

        return traj_interpolated 

    @timeit
    def refine_obstacle_predictions(self, obstacles, iteration, history,):
        start_time = time.time()
        predictor_input = converter.nuplan_to_predictor(
            self.scenario,
            self.predictor_scenario,
            self.predictor_dataloader,
            obstacles,
            iteration,
            history,
            history.ego_states[0],
        )
        dura = time.time() - start_time
        logger.debug(f"[TIMEIT] scenario conversion {dura}")

        start_prediction = time.time()

        with torch.no_grad():
            pred_dict, metrics = self.predictor.predict_and_evaluate(predictor_utils.batch_dict_tensors_to_device(predictor_input, self._device), 0) # shape: [num_required_obstacles, 6, 60, 5]

        pred_trajs = pred_dict['predicted_trajectory'].cpu().numpy() 
        pred_probs = pred_dict['predicted_probability'].cpu().numpy() 
        duration_prediction = time.time() - start_prediction
        logger.debug(f"[TIMEIT] hard constraint obstacle prediction {duration_prediction}")
        
        simulation_input = predictor_utils.get_simulation_input(predictor_input)
        oids = simulation_input['oids']

        current_velocity = np.zeros(len(oids))
        for idx, obj_id in enumerate(oids):
            obs = [obstacle for obstacle in obstacles if obstacle.obj_num == obj_id][0]
            current_velocity[idx] = obs.v0
        
        self.timestep_logs[iteration] = {
            "predicted_trajectory": pred_trajs,
            "predicted_probability": pred_probs,
            "predicted_object_ids": oids,
            "predicted_track_difficulty": simulation_input['predicted_track_difficulty'],
            "predicted_track_type": simulation_input['predicted_track_type'],
            "ego_track_difficulty": simulation_input['ego_track_difficulty'],
            "current_velocity": current_velocity,
        }
        self.timestep_logs[iteration].update(metrics)
        self.timestep_logs["metadata"]["iterations"] += 1

        # move predicted trajectories into ego-centered coords and refine the obstacle objects
        start_time = time.time()
        probs = np.zeros((self.planner.modes,))
        sc = self.planner.modes if self._planner_name == "RBMPCC" else self.planner.S 
        #sc = self.planner.S
        for idx, obj_id in enumerate(oids): 
            if obj_id == 'ego':
                continue

            obs = [obstacle for obstacle in obstacles if obstacle.obj_num == obj_id][0]
            # rotate and move
            
            R = predictor_utils.rot_mat(-obs.psi)

            pred_traj_w = {}
            obs_x_pred_t_m = []
            obs_y_pred_t_m = []
            obs_psi_pred_t_m = []
            obs_v_pred_t_m = []
            obs_stdx_pred_t_m = []
            obs_stdy_pred_t_m = []
            obs_rho_pred_t_m = []

            trajs = pred_trajs[idx]  #pred_dict[0]['predicted_trajectory'][idx].cpu().numpy()  # still in obstacle-centered coords [6, 60, 5]
            sorted_indices = sorted(range(len(pred_probs[idx])), key=lambda i: pred_probs[idx][i], reverse=True)
            trajs = trajs[sorted_indices]
            probs += pred_probs[idx][sorted_indices]/self.planner.num_hard_obs
            # if parking i.e. if not moving prediction also not moving
            if all(v < 0.3 and abs(v) < 0.6 for v in obs.v_hist):
                trajs = trajs*0

            for m in range(self.planner.modes):
                traj_w = (trajs[m, :, :2] @ R) + [obs.x, obs.y]
                pred_traj_w[m] = traj_w  # now in ego-centered (world) coords

            
            for j in range(sc):
                traj_w = (trajs[j, :, :2] @ R) + [obs.x, obs.y]
                std_xy_w = np.exp(trajs[j, :, 2:4]) #
                rho = trajs[j, :, 4]
                
                x = traj_w[:self.planner.N + 1, 0]
                y = traj_w[:self.planner.N + 1, 1]
                dx = np.diff(x)
                dy = np.diff(y)

                # Compute psi
                if all(abs(v) < 0.3 for v in obs.v_hist):
                    psi = np.ones_like(x)*obs.psi
                else:
                    psi = np.arctan2(dy, dx)
                    unwrapped = np.unwrap(psi) 
                    if obs.obj_type!=TrackedObjectType.PEDESTRIAN:
                        psi = planner_utils.clamp_orientation(psi , 0.4)

                # Compute velocity
                v = np.sqrt(dx**2 + dy**2) / self.planner.dt
                v = np.concatenate([v, np.repeat(v[-1], 1)]) # extrapolate last state by timestep
                obs_x_pred_t_m.append(traj_w[:self.planner.N + 1, 0])
                obs_y_pred_t_m.append(traj_w[:self.planner.N + 1, 1])
                obs_psi_pred_t_m.append(np.concatenate((psi, np.array([psi[-1]]))))
                obs_v_pred_t_m.append(v)
                obs_stdx_pred_t_m.append(std_xy_w[:self.planner.N + 1, 0])
                obs_stdy_pred_t_m.append(std_xy_w[:self.planner.N + 1, 1])
                obs_rho_pred_t_m.append(rho[:self.planner.N + 1])

            obs.trajs_w_modes = pred_traj_w    
            obs.S = sc

            obs.get_prediction(
                obs_x_pred_t_m, obs_y_pred_t_m, obs_psi_pred_t_m, self.planner.N, obs_v_pred_t_m, obs_stdx_pred_t_m,
                obs_stdy_pred_t_m, probs[:sc], obs_rho_pred_t_m#:self.planner["S"]]#final_pred_dicts[1]["pred_scores"]
            ) 
        
        dura = time.time() - start_time
        logger.debug(f"[TIMEIT] ego centration {dura}")


    @timeit
    def visualize(self, 
                current_input, 
                hard_constraint_obstacles, 
                soft_constraint_obstacles, 
                x_full=None, 
                traj_interpolated=None, 
                ego_progress=0, 
                vmax=0, 
                reference_path=None, 
                roadbound_left=None, 
                roadbound_right=None,):
        video_dir = self.log_dir / "video" / self._scenario_id
        video_dir.mkdir(exist_ok=True, parents=True)
        img = self.scene_render.visualize_from_planner(
            scenario=self.scenario,
            current_input=current_input,
            initialization=self._initialization,
            video_dir=video_dir,
            iteration_index=current_input.iteration.index,
            traj_interpolated=traj_interpolated,
            x_full=x_full,
            hard_constraint_obstacles=hard_constraint_obstacles,
            soft_constraint_obstacles=soft_constraint_obstacles,
            reference_path = reference_path,
            roadbound_left=roadbound_left, 
            roadbound_right=roadbound_right,
            ego_progress = ego_progress,
            vmax = vmax,
            planner_name=self._planner_name,
            params=self.planner.flattened_params,
        )

        self._imgs.append(img.copy())
        if current_input.iteration.index >= self.number_of_iterations - 2:
            logger.info('Saving Video')
            imageio.mimsave(
                video_dir / f"{self._scenario_id}.mp4",
                self._imgs,
                fps=10,
            )
            logger.info(f"\n video saved to {video_dir} / {self._scenario_id}.mp4 \n")
