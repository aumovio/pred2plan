"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import time
import os
import json
import random
import shutil
from pathlib import Path
import hydra
from typing import Optional, List

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import GetScenariosFromDbFileParams, get_scenarios_from_db_file
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

def line_segment_intersection(p1, p2, p3, p4):
    """
    Find the intersection point of two line segments defined by points (p1, p2) and (p3, p4).
    Optimized version with vector operations.
    
    Returns:
    - intersection point (x, y) if lines intersect
    - None if lines don't intersect
    """
    # Line segment 1 as p1 + t*(p2-p1)
    # Line segment 2 as p3 + s*(p4-p3)
    
    
    if np.sqrt((p1[0] - p3[0])**2 +  (p1[1] - p3[1])**2)<1:
        return (p1[0], p1[1])
    
    
    # Calculate direction vectors
    d1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    d2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
    
    # Calculate the determinant
    det = d1[0] * d2[1] - d1[1] * d2[0]
    
    # If det is zero, lines are parallel
    if abs(det) < 1e-10:
        return None
    
    # Calculate the parameters
    d = np.array([p3[0] - p1[0], p3[1] - p1[1]])
    t = (d2[1] * d[0] - d2[0] * d[1]) / det
    s = (d1[1] * d[0] - d1[0] * d[1]) / det
    
    # Check if intersection is within line segments
    if 0 <= t <= 1 and 0 <= s <= 1:
        # Calculate the intersection point
        return (p1[0] + t * d1[0], p1[1] + t * d1[1])
    

    
    return None

def construct_trajectory_segments(positions):
    """
    Construct line segments from trajectory points.
    This helps avoid repeated calculations in the intersection detection.
    
    Parameters:
    - positions: List of (x, y) tuples
    
    Returns:
    - List of segment tuples ((x1,y1), (x2,y2))
    """
    segments = []
    for i in range(len(positions) - 1):
        segments.append((positions[i], positions[i+1]))
    return segments

def find_trajectory_intersections(ego_trajectory, agent_trajectories_dict):
    """
    Find intersection points between ego trajectory and agent trajectories.
    Optimized for performance.
    
    Parameters:
    - ego_trajectory: List of (x, y, speed) tuples for ego vehicle
    - agent_trajectories_dict: Dictionary of agent trajectories keyed by track_token
    
    Returns:
    - List of dictionaries containing intersection information
    """
    # Start timing
    start_time = time.time()
    
    # Extract positions from ego trajectory
    ego_positions = [(pos[0], pos[1]) for pos in ego_trajectory]
    
    # Construct ego segments once to avoid repeated calculations
    ego_segments = construct_trajectory_segments(ego_positions)
    
    intersecting_agents = []
    
    # Check each agent
    for track_token, agent_data in agent_trajectories_dict.items():
        # Skip agents with insufficient data
        if len(agent_data['trajectory']) < 2:
            continue
            
        agent_positions = [(pos[0], pos[1]) for pos in agent_data['trajectory']]
        agent_timesteps = agent_data['timesteps']
        
        # Construct agent segments once
        agent_segments = construct_trajectory_segments(agent_positions)
        
        # Track intersections for this agent
        agent_intersections = []
        
        # Check each ego segment against each agent segment
        for i, ego_segment in enumerate(ego_segments):
            ego_p1, ego_p2 = ego_segment
            ego_time = i  # Timestep for first point of this ego segment
            
            for j, agent_segment in enumerate(agent_segments):
                
                agent_p1, agent_p2 = agent_segment
                agent_time = agent_timesteps[j]  # Timestep for first point of this agent segment
                if agent_time>i: # if ego goes first
                    break
                elif agent_time<i-50: #if too far away
                    continue
                
                # Check for intersection
                intersection_result = line_segment_intersection(ego_p1, ego_p2, agent_p1, agent_p2)
                
                if intersection_result:
                    intersection_point= intersection_result
                    
                    
                    # Speed at intersection (interpolate)
                    ego_speed = ego_trajectory[i][2]
                    agent_speed = agent_data['trajectory'][j][2]
                    
                    agent_intersections.append({
                        'intersection_point': intersection_point,
                        'ego_segment': (i, i+1),
                        'agent_segment': (j, j+1),
                        'ego_time': ego_time,
                        'agent_time': agent_time,
                        'ego_speed': ego_speed,
                        'agent_speed': agent_speed
                    })
        
        # If we found intersections for this agent, add to our results
        if agent_intersections:
            
            # Add the earliest intersection
            first_intersection = agent_intersections[0]
            intersecting_agents.append({
                'track_token': track_token,
                'intersection_point': first_intersection['intersection_point'],
                'ego_segment': first_intersection['ego_segment'],
                'agent_segment': first_intersection['agent_segment'],
                'ego_time': first_intersection['ego_time'],
                'agent_time': first_intersection['agent_time'],
                'ego_speed': first_intersection['ego_speed'],
                'agent_speed': first_intersection['agent_speed']
            })
            continue
    
    # Report performance
    elapsed = time.time() - start_time
    #print(f"Intersection detection completed in {elapsed:.3f} seconds")
    
    return intersecting_agents



def analyze_trajectory_intersections(scenario, duration):
    """
    Analyze a scenario for trajectory intersections between ego vehicle and agents.
    Optimized for performance and large datasets.
    
    Parameters:
    - scenario: The scenario object containing trajectories
    - duration: Duration to analyze
    
    Returns:
    - List of intersecting agents
    - Dictionary of all agent trajectories
    """
    # Start timing
    total_start_time = time.time()
    
    print(scenario._log_name,  scenario._initial_lidar_token)
    
    # Extract ego trajectory
    ego_trajectory = []
    for ego_state in list(scenario.get_ego_future_trajectory(0, duration)):
        ego_trajectory.append((ego_state.center.x, ego_state.center.y, ego_state.dynamic_car_state.speed))

    # Create a dictionary to store agent trajectories by track token
    agent_trajectories_dict = {}

    # Track how many agents we're processing
    agent_count = 0
    
    # Process all timesteps and gather agent data
    data_start_time = time.time()
    for time_idx, tracks in enumerate(scenario.get_future_tracked_objects(0, duration)):
        for agent in tracks.tracked_objects.get_agents():
            track_token = agent.metadata.track_token
            
            # Initialize if this is a new agent
            if track_token not in agent_trajectories_dict:
                agent_trajectories_dict[track_token] = {
                    'trajectory': [],
                    'timesteps': []
                }
                agent_count += 1
            
            # Add this timestep's data
            agent_trajectories_dict[track_token]['trajectory'].append(
                (agent.center.x, agent.center.y, agent.velocity.magnitude())
            )
            agent_trajectories_dict[track_token]['timesteps'].append(time_idx)
    
    data_time = time.time() - data_start_time
    #print(f"Data collection completed in {data_time:.3f} seconds")
    #print(f"Processing {agent_count} unique agents")

    # Find intersecting agents
    intersecting_agents = find_trajectory_intersections(ego_trajectory, agent_trajectories_dict)

    # Print intersection information
    if len(intersecting_agents)>0:
        print(f"Found {len(intersecting_agents)} agents with trajectory intersections.")
        for agent in intersecting_agents:
            print(f"Agent Track Token: {agent['track_token']}")
            print(f"Intersection point: ({agent['intersection_point'][0]:.2f}, {agent['intersection_point'][1]:.2f})")
            print(f"Ego time at intersection: {agent['ego_time']:.2f}")
            print(f"Agent time at intersection: {agent['agent_time']:.2f}")
            print(f"Ego speed at intersection: {agent['ego_speed']:.2f}")
            print(f"Agent speed at intersection: {agent['agent_speed']:.2f}")
            print("-" * 30)
    
    # # Plot trajectories with intersections
#     if len(intersecting_agents)>0:
#         plot_trajectories_with_intersections(ego_trajectory, agent_trajectories_dict, intersecting_agents)
    
    total_time = time.time() - total_start_time
    print(f"Total analysis completed in {total_time:.3f} seconds")
    
    return len(intersecting_agents)>0, intersecting_agents, agent_trajectories_dict, ego_trajectory

# Example call (to be replaced with actual scenario and duration)
# intersecting_agents, agent_trajectories_dict = analyze_trajectory_intersections(scenario, dur)

# def find_trajectory_intersections_with_proximity(ego_trajectory, agent_trajectories_dict, distance_threshold=2.0):
#     """
#     Find trajectory intersections plus conflict points based on proximity threshold.
#     This is an optional function if you want both exact intersections and proximity conflicts.
    
#     Parameters:
#     - ego_trajectory: List of (x, y, speed) tuples for ego vehicle
#     - agent_trajectories_dict: Dictionary of agent trajectories keyed by track_token
#     - distance_threshold: Minimum distance in meters to consider a conflict point (optional)
    
#     Returns:
#     - List of dictionaries containing intersection information
#     - True intersections are marked with is_proximity=False
#     - Proximity conflicts are marked with is_proximity=True
#     """
#     # Find exact intersections first using the optimized function
#     intersecting_agents = find_trajectory_intersections(ego_trajectory, agent_trajectories_dict)
    
#     # Mark these as not proximity-based
#     for agent in intersecting_agents:
#         agent['is_proximity'] = False
    
#     # Optionally add proximity conflicts
#     # This part would be implemented if you want to combine both approaches
    
#     return intersecting_agents


#dfdf


def calculate_progress(trajectory, start_idx, end_idx):
    """
    Calculate the total distance traveled along a path within a specified interval.
    
    Parameters:
    -----------
    trajectory : numpy.ndarray
        Array of shape (n, 2) containing the (x,y) coordinates of the agent's trajectory.
    start_idx : int
        Starting index in the trajectory (inclusive).
    end_idx : int
        Ending index in the trajectory (inclusive).
    
    Returns:
    --------
    float
        Total distance traveled along the path from start_idx to end_idx.
    """
    # Ensure indices are within bounds
    if start_idx < 0 or end_idx >= len(trajectory) or start_idx > end_idx:
        raise ValueError("Invalid indices: ensure 0 <= start_idx <= end_idx < len(trajectory)")
    
    # Extract the relevant portion of the trajectory
    path_segment = trajectory[start_idx:end_idx+1]
    
    # Calculate differences between consecutive points
    diffs = np.diff(path_segment, axis=0)
    
    # Calculate Euclidean distances for each step
    step_distances = np.sqrt(np.sum(diffs**2, axis=1))
    
    # Sum up all distances
    total_distance = np.sum(step_distances)
    
    return total_distance

def get_Delta_TTCP_t(dist_ego, speed_ego, dist_agent,speed_agent):
    if speed_ego == 0 or speed_agent == 0:
        return float('inf')
    return abs((dist_ego/speed_ego) - (dist_agent/speed_agent)) 

def fill_missing_timesteps(agent, T):
    traj_dim = 3  # assuming Lx3 as you said

    # Initialize full arrays
    full_timesteps = np.full(T, -1, dtype=np.float32)
    full_trajectory = np.full((T, traj_dim), -1, dtype=np.float32)

    # Convert to numpy for fast indexing
    existing_timesteps = np.array(agent["timesteps"])
    existing_trajectory = np.array(agent["trajectory"])

    # Fill only existing timesteps
    full_timesteps[existing_timesteps] = existing_timesteps
    full_trajectory[existing_timesteps] = existing_trajectory

    # Update agent
    agent["timesteps"] = full_timesteps.tolist()
    agent["trajectory"] = full_trajectory.tolist()

    return agent


def find_relevant_scenarios(scenarios, lookback=6):
    """
    Find scenarios where Delta TTCP is less than 3.
    
    Parameters:
    -----------
    scenarios : list
        List of dictionaries containing scenario information.
    lookback : int, optional
        Number of indices to look back from the minimum index.
        
    Returns:
    --------
    tuple
        Relevant scenarios and the minimum Delta TTCP found.
    """
    relevant_scenarios = []
    
    for s, scenario in enumerate(scenarios["scenarios"]):
        min_Delta_TTCP = float("inf")
        ego_states = np.array(scenarios['ego_trajectories'][s])
        ego_trajectory =  ego_states[:,:2]
        speed_ego = ego_states[:,2]
        
        intersecting_agents = scenarios["intersecting_agents"][s]
        
        skip = False
        for intersecting_agent in intersecting_agents:
            
            agent_idx = intersecting_agent["agent_time"]
            ego_idx = intersecting_agent["ego_time"]
            agent_token = intersecting_agent['track_token']
            agent_trajectories_dicts = fill_missing_timesteps(scenarios['agent_trajectories_dicts'][s][agent_token], ego_trajectory.shape[0])
            agent_visible_timesteps = agent_trajectories_dicts["timesteps"]
            agent_states = np.array(agent_trajectories_dicts["trajectory"])
            agent_trajectory = agent_states[:,:2]
            speed_agent = agent_states[:,2]
            
            min_idx = min(ego_idx, agent_idx)
            for i in range(max(0, min_idx - lookback), min_idx):
                if not i in agent_visible_timesteps:
                    continue
                dist_ego = calculate_progress(ego_trajectory, i, ego_idx)
                dist_agent = calculate_progress(agent_trajectory, i, agent_idx)
                print(i)
                print("dist_ego", dist_ego)
                print("dist_agent", dist_agent)
                print("speed_ego", speed_ego[i])
                print("speed_agent", speed_agent[i])
                delta_ttcp = get_Delta_TTCP_t(dist_ego, speed_ego[i], dist_agent, speed_agent[i])
                print("delta_ttcp", delta_ttcp,scenario)
                min_Delta_TTCP = min(min_Delta_TTCP, delta_ttcp)
                
                if min_Delta_TTCP < 3:
                    relevant_scenarios.append(scenario)
                    skip = True
                    break
                    
            if skip:
                break
    
    return relevant_scenarios


def plot_trajectories_with_intersections(ego_trajectory, agent_trajectories_dict, intersecting_agents):
    """
    Plot trajectories of ego vehicle and agents, highlighting intersection points.
    
    Parameters:
    - ego_trajectory: List of (x, y, speed) tuples for ego vehicle
    - agent_trajectories_dict: Dictionary of agent trajectories keyed by track_token
    - intersecting_agents: List of dictionaries containing intersection information
    """
    plt.figure(figsize=(32, 20))
    
    # Plot ego trajectory
    ego_x = [pos[0] for pos in ego_trajectory]
    ego_y = [pos[1] for pos in ego_trajectory]
    plt.plot(ego_x, ego_y, 'b-', linewidth=2, label='Ego Vehicle')
    
    # Mark ego start and end
    plt.scatter(ego_x[0], ego_y[0], color='blue', s=100, marker='^', label='Ego Start')
    plt.scatter(ego_x[-1], ego_y[-1], color='blue', s=100, marker='s', label='Ego End')
    
    # Generate colors for agents
    colors = list(mcolors.TABLEAU_COLORS)
    
    # Create a set of intersecting track tokens for quick lookup
    intersection_tokens = {agent['track_token'] for agent in intersecting_agents}
    
    # Plot agent trajectories
    for i, (track_token, agent_data) in enumerate(agent_trajectories_dict.items()):
        color = colors[i % len(colors)]
        agent_x = [pos[0] for pos in agent_data['trajectory']]
        agent_y = [pos[1] for pos in agent_data['trajectory']]
        
        # Use solid line for intersecting agents, dashed for non-intersecting
        line_style = '-' if track_token in intersection_tokens else ':'
        plt.plot(agent_x, agent_y, color=color, linestyle=line_style, 
                 alpha=0.7, linewidth=1.5, label=f'Agent {track_token[:8]}')
        
        # Mark agent start and end
        plt.scatter(agent_x[0], agent_y[0], color=color, s=50, marker='^')
        plt.scatter(agent_x[-1], agent_y[-1], color=color, s=50, marker='s')
    
    # Highlight intersection points
    for agent in intersecting_agents:
        intersection = agent['intersection_point']
        
        # Draw intersection point with a star marker
        plt.scatter(intersection[0], intersection[1], color='red', s=200, marker='*', 
                   zorder=10, label='Intersection' if agent == intersecting_agents[0] else "")
        
        # Add text annotation with time to conflict
        plt.annotate(f"TTC: {agent['ego_time']- agent['agent_time']:.2f}",
                    xy=intersection, xytext=(10, 10),
                    textcoords='offset points', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    # Add grid, labels, and legend
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Vehicle Trajectories and Intersection Points')
    
    # Create a legend with only one entry per unique label
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    # Equal axis aspect ratio
    plt.axis('equal')
    
    # Add text explaining the plot elements
    plt.figtext(0.5, 0.01, 
               "Solid lines: Intersecting agents | Dotted lines: Non-intersecting agents\n"
               "Red stars: Trajectory intersection points", 
               ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('trajectory_intersections.png', dpi=300)
    plt.show()
    
    
    


def select_relevant_scenarios(
    save_dir: Optional[str] = None,
    sample_duration: int = 10,
):
    """
    Identify and save scenarios with intersecting agents from the NuPlan dataset.

    Args:
        save_dir (Optional[str]): Directory to save the log names and tokens.
        sample_duration (int): Time window to analyze trajectory intersections.

    """
    # Environment setup
    os.environ['NUPLAN_DATA_ROOT'] = os.path.expanduser('~') + '/nuplan-devkit/nuplan/dataset'
    os.environ['NUPLAN_MAPS_ROOT'] = os.path.expanduser('~') + '/nuplan-devkit/nuplan/dataset/maps'
    os.environ['NUPLAN_EXP_ROOT'] = os.path.expanduser('~') + '/nuplan-devkit/nuplan/exp'
    os.environ['NUPLAN_DB_FILES'] = os.path.expanduser('~') + '/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/mini'
    os.environ['NUPLAN_MAP_VERSION'] = "nuplan-maps-v1.0"

    CONFIG_PATH = '../../nuplan-devkit/nuplan/planning/script/config/simulation'
    CONFIG_NAME = 'default_simulation'
    COMMON_DIR = '../../nuplan-devkit/nuplan/planning/script/config/common'
    EXPERIMENT_DIR = '../../nuplan-devkit/nuplan/planning/script/experiments'

    SAVE_DIR = Path(os.getenv('NUPLAN_DATA_ROOT')) / 'my_planner_tmp'   

    EGO_CONTROLLER = 'two_stage_controller'
    OBSERVATION = 'box_observation'

    DATASET_PARAMS = [
        'scenario_builder=nuplan',
        'scenario_filter=one_continuous_log',
        'scenario_filter.ego_displacement_minimum_m=0.5'
    ]

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)

    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'group={SAVE_DIR}',
        f'experiment_name=my_planner',
        f'job_name=my_planner',
        'experiment=${experiment_name}/${job_name}',
        'worker=sequential',
        f'ego_controller={EGO_CONTROLLER}',
        f'observation={OBSERVATION}',
        f'hydra.searchpath=[{COMMON_DIR}, {EXPERIMENT_DIR}]',
        'output_dir=${group}/${experiment}',
        'gpu=true',
        *DATASET_PARAMS,
    ])

    scenario_builder = build_scenario_builder(cfg=cfg)
    scenario_filter = build_scenario_filter(cfg=cfg.scenario_filter)

    extracted_scenarios = []
    for db_file in scenario_builder._db_files:
        params = GetScenariosFromDbFileParams(
            data_root=scenario_builder._data_root,
            log_file_absolute_path=db_file,
            expand_scenarios=scenario_filter.expand_scenarios,
            map_root=scenario_builder._map_root,
            map_version=scenario_builder._map_version,
            scenario_mapping=scenario_builder._scenario_mapping,
            vehicle_parameters=scenario_builder._vehicle_parameters,
            filter_tokens=scenario_filter.scenario_tokens,
            filter_types=scenario_filter.scenario_types,
            filter_map_names=scenario_filter.map_names,
            sensor_root=scenario_builder._sensor_root,
            remove_invalid_goals=scenario_filter.remove_invalid_goals,
            include_cameras=scenario_builder._include_cameras,
            verbose=scenario_builder._verbose
        )
        scenarios = get_scenarios_from_db_file(params)
        for tag, val in scenarios.items():
            if any(skip_tag in tag for skip_tag in ["traffic_light", "stationary", "traffic_cone", "carpark", "unknown", "construction", "on_pickup_dropoff"]):
                continue
            extracted_scenarios += val
            
    
    
    print(f'Extracted {len(extracted_scenarios)} scenarios from dataset')

    scenarios_intersecting = {
        "scenarios": [],
        "intersecting_agents": [],
        "agent_trajectories_dicts": [],
        "ego_trajectories": []
    }

    for scenario in extracted_scenarios:
        if any(scenario._initial_lidar_token in s._lidarpc_tokens for s in scenarios_intersecting["scenarios"]):
            continue

        intersect, intersecting_agents, agent_trajectories_dict, ego_trajectory = analyze_trajectory_intersections(scenario, sample_duration)
        if intersect:
            if scenarios_intersecting["scenarios"] and scenarios_intersecting["scenarios"][-1]._log_name == scenario._log_name and scenarios_intersecting['intersecting_agents'][-1][0]["track_token"] == intersecting_agents[0]["track_token"]:
                continue
            scenarios_intersecting["scenarios"].append(scenario)
            scenarios_intersecting["intersecting_agents"].append(intersecting_agents)
            scenarios_intersecting["agent_trajectories_dicts"].append(agent_trajectories_dict)
            scenarios_intersecting["ego_trajectories"].append(ego_trajectory)

    print(f'Found {len(scenarios_intersecting["scenarios"])} scenarios where agent path is intersecting ego path')
    rel_scenarios = find_relevant_scenarios(scenarios_intersecting)
    print(f'Found {len(rel_scenarios)} relevant scenarios')
    log_names = [s._log_name for s in rel_scenarios]
    tokens = [s._initial_lidar_token for s in rel_scenarios]

    if save_dir:
        with open(Path(save_dir) / "log_names.json", "w") as file:
            json.dump(log_names, file)
        with open(Path(save_dir) / "tokens.json", "w") as file:
            json.dump(tokens, file)

if __name__ == '__main__':
    select_relevant_scenarios(save_dir="simulation")
    
    
    
    
