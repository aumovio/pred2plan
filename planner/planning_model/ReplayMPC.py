"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import itertools
import logging
from typing import List, Optional, Type
import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

logger = logging.getLogger(__name__)

from planner.planning_model.SCMPCC import SCMPCC

class ReplayMPC(SCMPCC):

    def solve(self, scenario, iteration, **kwargs):
        current_state = scenario.get_ego_state_at_iteration(iteration.index)
        try:
            states = scenario.get_ego_future_trajectory(
                iteration.index, self.N*self.dt, self.N
            )
            self._trajectory = list(itertools.chain([current_state], states))
        except AssertionError:
            logger.warning("Cannot retrieve future ego trajectory. Using previous computed trajectory.")
            if self._trajectory is None:
                raise RuntimeError("Future ego trajectory cannot be retrieved from the scenario!")
            
        return InterpolatedTrajectory(self._trajectory), np.zeros((self.N, 3)), self.transform_trajectory_to_state(self._trajectory, **kwargs), 1, [0]
    
    def transform_trajectory_to_state(self, trajectory, **kwargs):
        """
        Convert List[EgoState] -> np.ndarray of shape [N, 3]
        containing [x, y, psi] of the rear axle for each state.
        """
        N = len(trajectory)
        x_full = np.zeros((N, 7), dtype=float)

        for i, state in enumerate(trajectory):
            ra = state.rear_axle  # usually an SE2 / pose container

            # Extract the rear-axle pose
            x_full[i, 0] = ra.x
            x_full[i, 1] = ra.y
            x_full[i, 2] = ra.heading  # or ra.theta, depending on your API
            x_full[i, 3] = state.dynamic_car_state.speed
            x_full[i, 4] = state.dynamic_car_state.rear_axle_acceleration_2d.x
            x_full[i, 5] = state.tire_steering_angle
            x_full[i, 6] = 0

        return x_full