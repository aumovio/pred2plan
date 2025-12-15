"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
from typing import Dict, List, Set
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import imageio

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.interpolated_trajectory import  InterpolatedTrajectory
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)

from simulation.utils.vis_utils import *
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

AGENT_COLOR_MAPPING = {
    TrackedObjectType.VEHICLE: "#001eff",
    TrackedObjectType.PEDESTRIAN: "#ae00ff",
    TrackedObjectType.BICYCLE: "#00aeff",
}

TRAFFIC_LIGHT_COLOR_MAPPING = {
    TrafficLightStatusType.GREEN: "#2ca02c",
    TrafficLightStatusType.YELLOW: "#ff7f0e",
    TrafficLightStatusType.RED: "#d62728",
}


class NuplanScenarioRender:
    def __init__(
        self,
        future_horizon: float = 8,
        sample_interval: float = 0.1,
        bounds=60,
        offset=20,
        fps = 1,
        disable_agent=False,
    ) -> None:
        super().__init__()

        self.future_horizon = future_horizon
        self.future_samples = int(self.future_horizon / sample_interval)
        self.sample_interval = sample_interval
        self.ego_params = get_pacifica_parameters()
        self.length = self.ego_params.length
        self.width = self.ego_params.width
        self.bounds = bounds
        self.offset = offset
        self.disable_agent = disable_agent
        self.initialize = False

        self.sampling = (1/self.sample_interval)/fps

        self.need_update = False
        self.candidate_index = None
        self._history_trajectory = []
        self._expert_history_trajectory = []

        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        self.static_objects_types = [
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.BARRIER,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]
        self.road_elements = [
            # SemanticMapLayer.ROADBLOCK,
            # SemanticMapLayer.ROADBLOCK_CONNECTOR,
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            # SemanticMapLayer.CROSSWALKh
        ]

        self._fig, self._ax = plt.subplots(figsize=(10,10))
        
    def visualize_from_planner(
        self,
        scenario,
        current_input,
        initialization,
        video_dir,
        iteration_index,
        traj_interpolated=None,
        x_full=None,
        hard_constraint_obstacles=None,
        soft_constraint_obstacles=None,
        reference_path = None,
        roadbound_left=None, 
        roadbound_right=None,
        ego_progress = 0,
        vmax = 0,
        planner_name=None,
        params=None,
        ):
        ego_state, _ = current_input.history.current_state

        hard_constraint_obstacles = list(hard_constraint_obstacles or [])
        soft_constraint_obstacles = list(soft_constraint_obstacles or [])
        stacked_scenarios, stacked_modes, stacked_sc_base_predictions = [], [], []

        if x_full is not None:
            if len(hard_constraint_obstacles)>0:
                for obs in hard_constraint_obstacles:
                    xy_pred = obs.trajs_w_modes
                    for m in range(params["MODES"]):
                        stacked_modes.append(np.vstack((xy_pred[m][:params['N'], 0], xy_pred[m][:params['N'], 1], np.array(obs.psi_preds[0][:-1]))).T)
                    for s1 in range(params["S"]):
                        s = min(s1, len(obs.x_preds) - 1)
                        stacked_scenarios.append(np.vstack((np.array(obs.x_preds[s][:-1]), np.array(obs.y_preds[s][:-1]), np.array(obs.psi_preds[s][:-1]))).T)
                stacked_modes = np.stack(stacked_modes)
                stacked_scenarios = stacked_modes if planner_name == "RBMPCC" else np.stack(stacked_scenarios)
                # stacked_modes = np.stack(stacked_modes)
            else:
                stacked_modes = None
                stacked_scenarios = None

            if len(soft_constraint_obstacles)>0:
                for obs in soft_constraint_obstacles:
                    for s1 in range(params["S"]):
                        s = min(s1, len(obs.x_preds) -1)
                        stacked_sc_base_predictions.append(np.vstack((np.array(obs.x_preds[s][:-1]), np.array(obs.y_preds[s][:-1]), np.array(obs.psi_preds[s][:-1]))).T)
                stacked_sc_base_predictions = np.stack(stacked_sc_base_predictions)
            else:
                stacked_sc_base_predictions = None

            contingency_plans = []
            for s in range(params["S"] - 1):
                k_start = 2 + (s + 1) * (params['N'] - 1)
                k_end = 2 * params['N'] + (params['N'] - 1) * s - 1
                contingency_plans.append(np.vstack((x_full[k_start:k_end, 0], x_full[k_start:k_end, 1], x_full[k_start:k_end, 2])).T)
            if len(contingency_plans)>0:
                contingency_plans = np.stack(contingency_plans)
                candidate_trajectories = self._global_to_local(contingency_plans, ego_state)
            else:
                candidate_trajectories = None

            planning_traj_local = self._global_to_local(traj_interpolated, ego_state) if traj_interpolated is not None else None
            pred_local = self._global_to_local(stacked_scenarios, ego_state) if stacked_scenarios is not None else None
            pred_sc_local = self._global_to_local(stacked_sc_base_predictions, ego_state) if stacked_sc_base_predictions is not None else None
            pred_modes_local = self._global_to_local(stacked_modes, ego_state) if stacked_modes is not None else None
            if pred_local is not None and pred_sc_local is not None:
                pred_local = np.concatenate([pred_local, pred_sc_local], axis=0)
            else:
                pred_local = pred_local
                
            img, fig = self.render_from_simulation(
                current_input=current_input,
                initialization=initialization,
                route_roadblock_ids= initialization.route_roadblock_ids, 
                scenario=scenario,
                iteration=iteration_index,
                planning_trajectory=planning_traj_local,
                candidate_trajectories=candidate_trajectories,
                predictions=pred_local,
                candidate_predictions=pred_modes_local,
                hard_constraint_agents=[obs.obj_num for obs in hard_constraint_obstacles] if len(hard_constraint_obstacles)>0 else [],
                soft_constraint_agents=[obs.obj_num for obs in soft_constraint_obstacles] if len(soft_constraint_obstacles)>0 else [],
                selected_stopline_ids = [obs.obj_num for obs in soft_constraint_obstacles if obs.obj_type==TrackedObjectType.BARRIER] if len(soft_constraint_obstacles)>0 else [],
                reference_path = reference_path,
                roadbound_left=roadbound_left, 
                roadbound_right=roadbound_right,    
                ego_progress = ego_progress,
                vmax = vmax,
                return_img=True,
                fig=self._fig,
            )

        else:
            img, fig = self.render_from_simulation(
                current_input=current_input,
                initialization=initialization,
                route_roadblock_ids= initialization.route_roadblock_ids, 
                scenario=scenario,
                iteration=iteration_index,
                planning_trajectory=self._global_to_local(traj_interpolated, ego_state),
                return_img=True,
                fig=self._fig,
            )

        if iteration_index % self.sampling == 0:
            fig.savefig(video_dir / f"timestep_{iteration_index:06d}.svg", format="svg")
            #fig.savefig(video_dir / f"timestep_{iteration_index:06d}.png", dpi=110, bbox_inches="tight", pad_inches=0)

        self._ax.clear()
        return img

    def render_from_simulation(
        self,
        current_input: PlannerInput = None,
        initialization: PlannerInitialization = None,
        route_roadblock_ids: List[str] = None,
        scenario=None,
        iteration=None,
        planning_trajectory=None,
        candidate_trajectories=None,
        predictions=None,
        candidate_predictions=None,
        rollout_trajectories=None,
        agent_attn_weights=None,
        candidate_index=None,
        hard_constraint_agents=None,
        soft_constraint_agents=None,
        selected_stopline_ids=None,
        reference_path = None,
        roadbound_left=None, 
        roadbound_right=None,
        ego_progress = 0,
        vmax = 0,
        return_img=True,
        fig=None
    ):
        current_timestep = current_input.iteration.index
        ego_state = current_input.history.ego_states[-1]
        map_api = initialization.map_api
        tracked_objects = current_input.history.observations[-1]
        traffic_light_status = current_input.traffic_light_data
        mission_goal = initialization.mission_goal
        if route_roadblock_ids is None:
            route_roadblock_ids = initialization.route_roadblock_ids

        self.candidate_index = candidate_index

        if scenario is not None:
            gt_state = scenario.get_ego_state_at_iteration(iteration)
            gt_trajectory = scenario.get_ego_future_trajectory(
                iteration=iteration,
                time_horizon=self.future_horizon,
                num_samples=self.future_samples,
            )
        else:
            gt_state, gt_trajectory = None, None

        return self.render(
            map_api=map_api,
            ego_state=ego_state,
            route_roadblock_ids=route_roadblock_ids,
            tracked_objects=tracked_objects,
            traffic_light_status=traffic_light_status,
            mission_goal=mission_goal,
            gt_state=gt_state,
            gt_trajectory=gt_trajectory,
            planning_trajectory=planning_trajectory,
            candidate_trajectories=candidate_trajectories,
            rollout_trajectories=rollout_trajectories,
            predictions=predictions,
            candidate_predictions=candidate_predictions,
            agent_attn_weights=agent_attn_weights,
            hard_constraint_agents=hard_constraint_agents,
            soft_constraint_agents=soft_constraint_agents,
            selected_stopline_ids=selected_stopline_ids,
            reference_path = reference_path,
            roadbound_left=roadbound_left, 
            roadbound_right=roadbound_right,
            ego_progress = ego_progress,
            vmax = vmax,
            current_timestep = current_timestep,
            return_img=return_img,
            fig=fig
        )

    def render_from_scenario(
        self,
        scenario: AbstractScenario,
        ego_state: EgoState = None,
        iteration=0,
        planning_trajectory=None,
        candidate_trajectories=None,
        rollout_trajectories=None,
        predictions=None,
        return_image=True,
    ):
        if ego_state is None:
            ego_state = scenario.get_ego_state_at_iteration(iteration)
        map_api = scenario.map_api
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        tracked_objects = scenario.get_tracked_objects_at_iteration(iteration)
        traffic_light_status = scenario.get_traffic_light_status_at_iteration(iteration)
        mission_goal = scenario.get_mission_goal()
        gt_state = scenario.get_ego_state_at_iteration(iteration)
        gt_trajectory = scenario.get_ego_future_trajectory(
            iteration=iteration,
            time_horizon=self.future_horizon,
            num_samples=self.future_samples,
        )

        return self.render(
            map_api=map_api,
            ego_state=ego_state,
            route_roadblock_ids=route_roadblock_ids,
            tracked_objects=tracked_objects,
            traffic_light_status=traffic_light_status,
            mission_goal=mission_goal,
            gt_state=gt_state,
            gt_trajectory=gt_trajectory,
            planning_trajectory=planning_trajectory,
            candidate_trajectories=candidate_trajectories,
            rollout_trajectories=rollout_trajectories,
            predictions=predictions,
            return_img=return_image,
        )

    def render(
        self,
        map_api: AbstractMap,
        ego_state: EgoState,
        route_roadblock_ids: List[str],
        tracked_objects: TrackedObjects,
        traffic_light_status: Dict[int, TrafficLightStatusData],
        mission_goal: StateSE2,
        gt_state=None,
        gt_trajectory=None,
        planning_trajectory=None,
        candidate_trajectories=None,
        rollout_trajectories=None,
        predictions=None,
        candidate_predictions=None,
        agent_attn_weights=None,
        hard_constraint_agents=None,
        soft_constraint_agents=None,
        selected_stopline_ids=None,
        reference_path = None,
        roadbound_left=None, 
        roadbound_right=None,
        ego_progress = 0,
        vmax = 0,
        current_timestep = 0,
        return_img=False,
        fig = None
    ):
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            ax = fig.gca()
            ax.clear()

        self._history_trajectory.append(ego_state.rear_axle.array)
        if gt_state is not None:
            self._expert_history_trajectory.append(gt_state.rear_axle.array)


        self.origin = ego_state.rear_axle.array
        self.angle = ego_state.rear_axle.heading
        self.rot_mat = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ],
            dtype=np.float64,
        )

        self._plot_map(
            ax,
            map_api,
            ego_state.center.point,
            traffic_light_status,
            set(route_roadblock_ids),
            selected_stopline_ids,
            reference_path,
            roadbound_left, 
            roadbound_right,
        )
        ax.text(
            0.98, 0.98, 
            f"vcurr={ego_state.dynamic_car_state.speed:.2f}", 
            transform=ax.transAxes,   # use normalized axes coords
            ha="right", va="top",     # align top-right
            fontsize=10, 
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")  # nice box
        )
        ax.text(
            0.98, 0.96, 
            f"acurr={ego_state.dynamic_car_state.acceleration:.2f}", 
            transform=ax.transAxes,   # use normalized axes coords
            ha="right", va="top",     # align top-right
            fontsize=10, 
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")  # nice box
        )
        ax.text(
            0.98, 0.94, 
            f"vmax={vmax:.2f}", 
            transform=ax.transAxes,   # use normalized axes coords
            ha="right", va="top",     # align top-right
            fontsize=10, 
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")  # nice box
        )
        ax.text(
            0.98, 0.92, 
            f"ego_progress={ego_progress:.2f}", 
            transform=ax.transAxes,   # use normalized axes coords
            ha="right", va="top",     # align top-right
            fontsize=10, 
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")  # nice box
        )
        ax.text(
            0.98, 0.90, 
            f"current_timestep={current_timestep:.2f}", 
            transform=ax.transAxes,   # use normalized axes coords
            ha="right", va="top",     # align top-right
            fontsize=10, 
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")  # nice box
        )


        self._plot_ego(ax, ego_state)

        if gt_state is not None:
            self._plot_ego(ax, gt_state, gt=True)
            gt_trajectory = np.array([state.rear_axle.array for state in gt_trajectory])
            gt_trajectory = np.matmul(gt_trajectory - self.origin, self.rot_mat)
            #ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], color="blue", alpha=0.5)

        if not self.disable_agent:
            for track in tracked_objects.tracked_objects:
                self._plot_tracked_object(ax, track, agent_attn_weights, hard_constraint_agents, soft_constraint_agents)

        if planning_trajectory is not None:
            self._plot_planning(ax, planning_trajectory)

        if candidate_trajectories is not None:
            self._plot_candidate_trajectories(ax, candidate_trajectories)

        if rollout_trajectories is not None:
            self._plot_rollout_trajectories(ax, rollout_trajectories)

        if candidate_predictions is not None:
            self._plot_candidate_predictions(ax, candidate_predictions)
        if predictions is not None:
            self._plot_prediction(ax, predictions)

        self._plot_mission_goal(ax, mission_goal)
        self._plot_history(ax)

        ax.axis("equal")
        ax.set_xlim(xmin=-self.bounds + self.offset, xmax=self.bounds + self.offset)
        ax.set_ylim(ymin=-self.bounds, ymax=self.bounds)
        ax.axis("off")
        plt.tight_layout(pad=0)

        if return_img:
            fig.canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            img = np.asarray(fig.canvas.buffer_rgba()).reshape(int(height), int(width), 4)[: , :, :3]
            # img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(
            #     int(height), int(width), 4
            # )[:, :, 1:4]
            return img, fig  # return figure instead return plt.gca() oder plt.gcf()
        else:
            return fig
            #plt.show()

    def _plot_map(
        self,
        ax,
        map_api: AbstractMap,
        query_point: Point2D,
        traffic_light_status: Dict[int, TrafficLightStatusData],
        route_roadblock_ids: Set[str],
        selected_stopline_ids: List[str],
        reference_path = None,
        roadbound_left=None, 
        roadbound_right=None,
    ):
        road_objects = map_api.get_proximal_map_objects(
            query_point, self.bounds + self.offset, self.road_elements
        )
        road_objects = (
            road_objects[SemanticMapLayer.LANE]
            + road_objects[SemanticMapLayer.LANE_CONNECTOR]
        )
        tls = {tl.lane_connector_id: tl.status for tl in traffic_light_status}

        # reference_line = Line2D(reference_path[:,0],
        #                         reference_path[:,1],
        #                         color="#e100ff",
        #                         linewidth=3.0,
        #                         alpha=0.8,
        #                         zorder=2
        #                         )
        # ax.add_artist(reference_line)
        reference_line = np.matmul(reference_path[:,:2] - self.origin, self.rot_mat)
        ax.plot(reference_line[:,0],
                reference_line[:,1],
                color="#e100ff",
                linestyle="--",
                linewidth=2.0,
                alpha=0.8,
                zorder=1
                )
        
        roadbound_left_line = np.matmul(roadbound_left[:,:2] - self.origin, self.rot_mat)
        roadbound_right_line = np.matmul(roadbound_right[:,:2] - self.origin, self.rot_mat)
        ax.plot(roadbound_left_line[:,0],
                roadbound_left_line[:,1],
                # "kx"
                color="#3d0045",
                linestyle="--",
                linewidth=2.0,
                alpha=0.8,
                zorder=1
                )
        
        ax.plot(roadbound_right_line[:,0],
                roadbound_right_line[:,1],
                # "kx",
                color="#3d0045",
                linestyle="--",
                linewidth=2.0,
                alpha=0.8,
                zorder=1
                )


        plotted_stopline=set()
        for obj in road_objects:
            obj_id = int(obj.id)
            kwargs = {"color": "lightgray", "alpha": 0.4, "ec": None, "zorder": 0}
            if obj.get_roadblock_id() in route_roadblock_ids:
                kwargs["color"] = "dodgerblue"
                kwargs["alpha"] = 0.1
                kwargs["zorder"] = 1
            ax.add_artist(self._polygon_to_patch(obj.polygon, **kwargs))

            cl_color, linewidth = "gray", 1.0
            if obj_id in tls:
                cl_color = TRAFFIC_LIGHT_COLOR_MAPPING.get(tls[obj_id], "gray")
                linewidth = 1

            for stopline in obj.stop_lines:
                if stopline.id in plotted_stopline:
                    continue
                if stopline.id in selected_stopline_ids: 
                    kwargs = {"color":  "#ae00ff", "alpha": 0.6, "ec": None, "zorder": 1}
                else:
                    kwargs = {"color": cl_color, "alpha": 0.3, "ec": None, "zorder": 1} 
                centroid = stopline.polygon.centroid
                ax.plot(centroid.x, centroid.y, marker="x", markersize=12, color="black", zorder=1)
                ax.add_artist(self._polygon_to_patch(stopline.polygon, **kwargs))
                plotted_stopline.add(stopline.id)


            cl = np.array([[s.x, s.y] for s in obj.baseline_path.discrete_path])
            cl = np.matmul(cl - self.origin, self.rot_mat)
            ax.plot(
                cl[:, 0],
                cl[:, 1],
                color=cl_color,
                alpha=0.5,
                linestyle="--",
                zorder=1,
                linewidth=linewidth,
            )

        crosswalks = map_api.get_proximal_map_objects(
            query_point, self.bounds + self.offset, [SemanticMapLayer.CROSSWALK]
        )
        for obj in crosswalks[SemanticMapLayer.CROSSWALK]:
            xys = np.array(obj.polygon.exterior.coords.xy).T
            xys = np.matmul(xys - self.origin, self.rot_mat)
            polygon = Polygon(
                xys, color="gray", alpha=0.4, ec=None, zorder=3, hatch="///"
            )
            ax.add_patch(polygon)

    def _plot_ego(self, ax, ego_state: EgoState, gt=False):
        kwargs = {"lw": 1.5}
        if gt:
            ax.add_patch(
                self._polygon_to_patch(
                    ego_state.car_footprint.geometry,
                    color="gray",
                    alpha=0.3,
                    zorder=9,
                    **kwargs,
                )
            )
        else:
            ax.add_patch(
                self._polygon_to_patch(
                    ego_state.car_footprint.geometry,
                    ec="#ff7f0e",
                    fill=False,
                    zorder=10,
                    **kwargs,
                )
            )

        ax.plot(
            [1.69, 1.69 + self.length * 0.75],
            [0, 0],
            color="#ff7f0e",
            linewidth=1.5,
            zorder=11,
        )

    def _plot_tracked_object(self, ax, track: TrackedObject, agent_attn_weights=None, hard_constraint_agents=None, soft_constraint_agents=None):
        center, angle = track.center.array, track.center.heading
        center = np.matmul(center - self.origin, self.rot_mat)
        angle = angle - self.angle

        direct = np.array([np.cos(angle), np.sin(angle)]) * track.box.length / 1.5
        direct = np.stack([center, center + direct], axis=0)

        color = AGENT_COLOR_MAPPING.get(track.tracked_object_type, "k")
        if hard_constraint_agents is None or track.metadata.track_token in hard_constraint_agents:
            color = "#ee1515"
            ax.add_patch(
                self._polygon_to_patch(
                    track.box.geometry, ec=color, fill=False, alpha=1.0, zorder=4, lw=1.5
                )
            )
        elif soft_constraint_agents is None or track.metadata.track_token in soft_constraint_agents:
            ax.add_patch(
                self._polygon_to_patch(
                    track.box.geometry, ec=color, fill=False, alpha=1.0, zorder=4, lw=1.5
                )
            )
        else: 
            ax.add_patch(
                self._polygon_to_patch(
                    track.box.geometry, ec=color, fill=False, alpha=0.4, zorder=4, lw=1.5
                )
            )
        
            
        if color != "k":
            ax.plot(direct[:, 0], direct[:, 1], color=color, linewidth=1, zorder=4)
        if agent_attn_weights is not None and track.track_token in agent_attn_weights:
            weight = agent_attn_weights[track.track_token]
            ax.text(
                center[0],
                center[1] + 0.5,
                f"{weight:.2f}",
                color="red",
                zorder=5,
                fontsize=7,
            )
        
        # ax.text(
        #     center[0],
        #     center[1] - 0.5,
        #     str(track.track_token),
        #     color="black",
        #     zorder=5,
        #     fontsize=5
        # )

    def _polygon_to_patch(self, polygon: shapely.geometry.Polygon, **kwargs):
        polygon = np.array(polygon.exterior.xy).T
        polygon = np.matmul(polygon - self.origin, self.rot_mat)
        return patches.Polygon(polygon, **kwargs)

    def _plot_planning(self, ax, planning_trajectory: np.ndarray):
        plot_polyline(
            ax,
            [planning_trajectory],
            linewidth=4,
            arrow=False,
            zorder=6,
            alpha=1.0,
            cmap="spring",
        )

    def _plot_candidate_trajectories(self, ax, candidate_trajectories: np.ndarray):
        for traj in candidate_trajectories:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color="red",
                alpha=0.8,
                zorder=6,
                linewidth=3,
            )
            #ax.scatter(traj[-1, 0], traj[-1, 1], color="gray", zorder=5, s=10)
            
            
    def _plot_candidate_predictions(self, ax, candidate_trajectories: np.ndarray):
        for traj in candidate_trajectories:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color="gray",
                alpha=0.2,
                zorder=5,
                linewidth=2,
            )
            #ax.scatter(traj[-1, 0], traj[-1, 1], color="gray", zorder=5, s=10)

    def _plot_rollout_trajectories(self, ax, candidate_trajectories: np.ndarray):
        for i, traj in enumerate(candidate_trajectories):
            kwargs = {"lw": 1.5, "zorder": 5, "color": "cyan"}
            if self.candidate_index is not None and i == self.candidate_index:
                kwargs = {"lw": 5, "zorder": 6, "color": "red"}
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, **kwargs)
            ax.scatter(traj[-1, 0], traj[-1, 1], color="cyan", zorder=5, s=10)

    def _plot_prediction(self, ax, predictions: np.ndarray):
        kwargs = {"lw": 3}
        for pred in predictions:
            pred = pred[:, ..., :2]
            self._plot_polyline(ax, pred, cmap="winter", **kwargs)

    def _plot_polyline(self, ax, polyline, cmap="spring", **kwargs) -> None:
        arc = get_polyline_arc_length(polyline)
        polyline = polyline.reshape(-1, 1, 2)
        segment = np.concatenate([polyline[:-1], polyline[1:]], axis=1)
        norm = plt.Normalize(arc.min(), arc.max())
        lc = LineCollection(
            segment,
            cmap=cmap,
            norm=norm,
            array=arc,
            **kwargs,
        )
        ax.add_collection(lc)


    def _plot_mission_goal(self, ax, mission_goal: StateSE2):
        if mission_goal:
            point = np.matmul(mission_goal.point.array - self.origin, self.rot_mat)
            ax.plot(point[0], point[1], marker="*", markersize=5, color="gold", zorder=6)

    def _plot_history(self, ax):
        history = np.array(self._history_trajectory)
        history = np.matmul(history - self.origin, self.rot_mat)
        ax.plot(
            history[:, 0],
            history[:, 1],
            color="#ff7f0e",
            alpha=0.5,
            zorder=6,
            linewidth=2,
        )


            
    def _global_to_local(self, global_trajectory: np.ndarray, ego_state: EgoState):
        if isinstance(global_trajectory, InterpolatedTrajectory):
            states: List[EgoState] = global_trajectory.get_sampled_trajectory()
            global_trajectory = np.stack(
                [
                    np.array(
                        [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                    )
                    for state in states
                ],
                axis=0,
            )

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)