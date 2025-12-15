"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
from os import listdir
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_closed_loop_metrics_sim_logs(directory):
    ax = []
    ay = []

    v_avg_log = []
    a_avg_log = []
    j_avg_log = []
    delta_avg_log = []
    ddelta_avg_log = []
    d_min_log = []
    tc_log = []

    for file in listdir(directory):  # for scenario
        if not file[-3:] == 'pkl':
            continue

        with open(directory + '/' + file, 'rb') as f:
            scenario_data = pickle.load(f)

        x_log = []
        u_log = []
        
        map = scenario_data.pop('map', None)
        ref_lane = scenario_data.pop('ref_lane', None)
        scenario_name = scenario_data.pop('scenario_name', None)

        obs_x_log = np.empty((5, len(scenario_data)))
        obs_y_log = np.empty((5, len(scenario_data)))
        obs_psi_log = np.empty((5, len(scenario_data)))

        for t, lidar_pc in enumerate(scenario_data):    # for timesteps
            
            ego_box_xy = scenario_data[lidar_pc]['ego_state_xy']
            prediction_time = scenario_data[lidar_pc]['prediction_duration']
            x_full = scenario_data[lidar_pc]['x_full']
            u_full = scenario_data[lidar_pc]['u_full']
            obstacles = scenario_data[lidar_pc]['obstacles']
            obstacle_array = scenario_data[lidar_pc]['obstacle_array']
            x_log.append(x_full[0, :])
            u_log.append(u_full[0, :])
            ax.append(x_full[0, 4])
            ay.append(x_full[0, 3]**2*np.tan(x_full[0, 5])/3.089)
        
            tc_log.append(prediction_time)

            relevant_obstacles = [obs['obj_num'] for obs in obstacle_array]
            
            obs_idx = 0
            for agent in obstacles:
                if agent.metadata.track_token in relevant_obstacles:
                    obs_x_log[obs_idx][t] = agent.center.x
                    obs_y_log[obs_idx][t] = agent.center.y
                    obs_psi_log[obs_idx][t] = agent.center.heading
                    obs_idx += 1

        v_avg, a_avg, j_avg, delta_avg, ddelta_avg, d_min = calculate_metrics_cl(
            x_log, u_log, obs_x_log, obs_y_log, obs_psi_log,
        )
        
        v_avg_log.append(v_avg)
        a_avg_log.append(a_avg)
        j_avg_log.append(j_avg)
        delta_avg_log.append(delta_avg)
        ddelta_avg_log.append(ddelta_avg)
        d_min_log.append(d_min)

    return np.array(v_avg_log), np.array(a_avg_log), np.array(j_avg_log), np.array(delta_avg_log), np.array(ddelta_avg_log), \
        np.array(d_min_log), np.array(tc_log), ax, ay

def calculate_metrics_cl(x_log,u_log,obs_x_log, obs_y_log, obs_psi_log):

    X = np.array(x_log).reshape(len(x_log),len(x_log[0]))
    U = np.array(u_log).reshape(len(u_log),len(u_log[0]))
    v_avg = np.mean(np.abs(X[:,3]))
    a_avg = np.mean(np.abs(X[:,4]))
    delta_avg = np.mean(np.abs(X[:,5]))
    j_avg = np.mean(np.abs(U[:,0]))
    ddelta_avg = np.mean(np.abs(U[:,1]))


    # Vectorized computation of longitudinal distances
    delta_x = np.array(obs_x_log).T - X[:, 0][:, np.newaxis]
    delta_y = np.array(obs_y_log).T - X[:, 1][:, np.newaxis]
    
    longitudinal_distances = delta_x * np.cos(X[:, 2][:, np.newaxis]) + delta_y * np.sin(X[:, 2][:, np.newaxis])
    lateral_distances = delta_y * np.cos(X[:, 2][:, np.newaxis]) - delta_x * np.sin(X[:, 2][:, np.newaxis])
    
    valid_longitudinal_distances = np.abs(longitudinal_distances[np.abs(lateral_distances) < 1.0])
    
    # Find the minimum longitudinal distance (if any valid distances exist)
    if valid_longitudinal_distances.size > 0:
        d_min = np.min(valid_longitudinal_distances)
    else:
        d_min = 20

    return v_avg, a_avg, j_avg, delta_avg, ddelta_avg, d_min


def get_ground_truth_predictions(directory_gt):
    all_scenarios_gt = []
    files_gt=listdir(directory_gt)
    files_sorted_gt = sorted(
        files_gt,
        key=lambda fn: fn.split('_', 2)[-1]
    )


    for i, file in enumerate(files_sorted_gt):
        if not file[-3:] == 'pkl':
            continue

        with open(directory_gt + '/' + file, 'rb') as f:
            scenario_data = pickle.load(f)

        scenario_gt = {}
        scenario_gt['scenario_name'] = scenario_data["scenario_name"]

        for t, lidar_pc in enumerate(scenario_data): 
            if t>= 149:
                break# for timesteps
            obstacle_array = scenario_data[t]['obstacle_array']
            scenario_gt[t] = {}
            for obs in obstacle_array:
                x_gt= obs["x_gt"]
                y_gt= obs["y_gt"]
                traj_gt = np.column_stack((x_gt, y_gt))

                scenario_gt[t][obs["obj_num"]]=traj_gt

        all_scenarios_gt.append(scenario_gt)
    return all_scenarios_gt
                
                


def compute_ol_results_sim(directory, all_scenarios_gt):
    all_scenarios_results_ol = {}
    all_scenarios_results_ol['single_scenario_results'] = []
    all_scenarios_results_ol['mode_results'] = []
    files = [f for f in listdir(directory) if f.endswith('.pkl')]
    files_sorted = sorted(
        files,
        key=lambda fn: fn.split('_', 2)[-1]
    )


    M = 6   # number of modes

    # TOTAL accumulators (over all scenarios)
    total_sum_errors     = np.zeros(M, dtype=float)
    total_count_steps    = np.zeros(M, dtype=int)
    total_sum_fde        = np.zeros(M, dtype=float)
    total_count_vehicles = np.zeros(M, dtype=int)
    total_sum_tc         = np.zeros(M, dtype=float)
    total_count_tc       = np.zeros(M, dtype=int)
    total_sum_dist = 0
    
    
    sum_min_ADE_sc = 0
    sum_min_FDE_sc = 0
    sum_min_TC_sc = 0
    count_min_ADE_sc = 0
    count_min_FDE_sc = 0
    count_min_TC_sc = 0
    
    sum_min_ADE_t_o = 0
    sum_min_FDE_t_o = 0
    sum_min_TC_t_o = 0
    count_min_ADE_t_o = 0
    count_min_FDE_t_o = 0
    count_min_TC_t_o = 0
    
    for i, file in enumerate(files_sorted):

        with open(directory + '/' + file, 'rb') as f:
            scenario_data = pickle.load(f)

        scenario_results_ol = {}
        scenario_results_ol['scenario_name'] = scenario_data["scenario_name"]
        if  scenario_results_ol['scenario_name']!=all_scenarios_gt[i]['scenario_name']:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        scenario_results_ol['mode_results'] = []
        scenario_results_ol['overall_results'] = {}

        sum_errors     = np.zeros(M, dtype=float)
        count_steps    = np.zeros(M, dtype=int)
        sum_fde        = np.zeros(M, dtype=float)
        count_vehicles = np.zeros(M, dtype=int)
        
        sum_tc     = np.zeros(M, dtype=float)
        count_tc   = np.zeros(M, dtype=int)
        
        sum_dist = 0
        for t, lidar_pc in enumerate(scenario_data): 
            if t>= 149:
                break# for timesteps
            has_next_step =  t<148
            if has_next_step:
                next_obstacles = scenario_data[t+1]['obstacle_array']
                
            obstacle_array = scenario_data[t]['obstacle_array']
            observations = scenario_data[t]['obstacles']

     

            for o, obs in enumerate(obstacle_array):
                agent = next(agent for agent in observations if obs['obj_num'] == agent.metadata.track_token)
            
                if obs["obj_num"] not in all_scenarios_gt[i][t].keys():
                    continue
                else:
                    #print("found")
                    traj_gt = all_scenarios_gt[i][t][obs["obj_num"]][1:,:]# obs['pred_trajs_w'][0] #  
                    valid_mask = traj_gt[:, 0] != -1000

                if not valid_mask.any():
                    continue
                    
                if has_next_step:
                    next_obs = next(
                        (o for o in next_obstacles if o['obj_num'] == obs["obj_num"]),
                        None
                    )
                    if next_obs:
                        if obs["obj_num"] not in all_scenarios_gt[i][t+1].keys():
                            continue
                        else:
                            traj_gt_next = all_scenarios_gt[i][t+1][obs["obj_num"]][1:,:] # next_obs['pred_trajs_w'][0]  # 
                
                sum_errors_t_o     = np.zeros(M, dtype=float)
                count_steps_t_o    = np.zeros(M, dtype=int)
                sum_fde_t_o        = np.zeros(M, dtype=float)
                count_vehicles_t_o = np.zeros(M, dtype=int)

                sum_tc_t_o     = np.zeros(M, dtype=float)
                count_tc_t_o   = np.zeros(M, dtype=int)
                
                endpoints = np.zeros((M,2),  dtype=float)
                for m in range(6):
                    traj = obs['pred_trajs_w'][m][:50,:]

                    errors = np.linalg.norm(traj[valid_mask] - traj_gt[valid_mask], axis=1)
                    #print(errors)
                    sum_errors[m]  += errors.sum()
                    count_steps[m] += errors.shape[0]

                    fde = errors[-1]
                    sum_fde[m]+= fde
                    count_vehicles[m] += 1
                    
                    
                    sum_errors_t_o[m]  += errors.sum()
                    count_steps_t_o[m] += errors.shape[0]

                    sum_fde_t_o[m]+= fde
                    count_vehicles_t_o[m] += 1
                    
                    endpoints[m,:] = traj[-1,:]
                    
                    # Temporal Consistency: compare end-of-horizon at t to start-of-horizon at t+1
                    if has_next_step and next_obs is not None:
                        # predicted at t+1 for same mode
                        traj_next = next_obs['pred_trajs_w'][m][:50,:]
                        if traj[-1,0]==-1000 or traj_next[-2,0]==-1000:
                            continue
                        # numerator: distance between traj_pred[t][-1] and traj_pred_next[0]
                        num = np.linalg.norm(traj[-1] - traj_next[-2])

                        # denominator: distance GT curr pos → GT next pos
                        denom = np.linalg.norm(traj_gt[0] - traj_gt_next[0])
                        if denom > 0:
                            sum_tc[m]   += (num / denom)
                            count_tc[m] += 1
                            sum_tc_t_o[m]   += (num / denom)
                            count_tc_t_o[m] += 1
                            
                sum_dist += (calculate_pairwise_distances(endpoints)/6)/max(agent.velocity.magnitude(),1)
                
                
                
                
                ADE_t_o = sum_errors_t_o / np.maximum(count_steps_t_o, 1)
                FDE_t_o = sum_fde_t_o      / np.maximum(count_vehicles_t_o, 1)
                TC_t_o  = sum_tc_t_o       / np.maximum(count_tc_t_o,       1)

                m_minADE_t_o = np.argmin(ADE_t_o)
                m_minFDE_t_o = np.argmin(FDE_t_o)
                m_minTC_t_o =  np.argmin(TC_t_o)
                sum_min_ADE_t_o  += ADE_t_o[m_minADE_t_o]
                sum_min_FDE_t_o  += FDE_t_o[m_minFDE_t_o]
                sum_min_TC_t_o  += TC_t_o[m_minTC_t_o]
                count_min_ADE_t_o  += 1
                count_min_FDE_t_o += 1
                count_min_TC_t_o  += 1
                
                    
        
        DM = sum_dist / np.maximum(count_vehicles, 1)
        ADE = sum_errors   / np.maximum(count_steps, 1)
        FDE = sum_fde      / np.maximum(count_vehicles, 1)
        TC  = sum_tc       / np.maximum(count_tc,       1)
        minADE = sum_min_ADE_t_o/np.maximum(count_min_ADE_t_o,1)
        # if minADE>1.5:
        #     print(scenario_data["scenario_name"])
        minFDE = sum_min_FDE_t_o/np.maximum(count_min_FDE_t_o,1)
        minTC = sum_min_TC_t_o/np.maximum(count_min_TC_t_o,1)
        
        sum_min_ADE_sc  += minADE
        sum_min_FDE_sc  += minFDE 
        sum_min_TC_sc  += minTC
        count_min_ADE_sc  += 1
        count_min_FDE_sc += 1
        count_min_TC_sc  += 1
        #print(ADE)
        scenario_results_ol['overall_results']["minADE6"] = minADE#np.min(ADE)
        scenario_results_ol['overall_results']["minFDE6"] = minFDE#np.min(FDE)
        scenario_results_ol['overall_results']["minTC6"] = minTC #np.min(TC)
        #print(minADE)
        scenario_results_ol['overall_results']["ADE1"] = ADE[0]
        scenario_results_ol['overall_results']["FDE1"] = FDE[0]
        scenario_results_ol['overall_results']["TC1"] = TC[0]
        scenario_results_ol['overall_results']["DM"] = DM #total_DM
        for m in range(6):
            scenario_results_ol['mode_results'].append({
                'mode': m,
                'ADE': ADE[m],
                'FDE': FDE[m],
                'TC': TC[m]
            })

        #print(scenario_results_ol)
        all_scenarios_results_ol['single_scenario_results'].append(scenario_results_ol)  
        # accumulate into TOTALs
        total_sum_errors     += sum_errors
        total_count_steps    += count_steps
        total_sum_fde        += sum_fde
        total_count_vehicles += count_vehicles
        total_sum_tc         += sum_tc
        total_count_tc       += count_tc
        
        total_sum_dist += sum_dist
        
        
    
    # overall‐across‐scenarios metrics
    total_ADE = total_sum_errors     / np.maximum(total_count_steps,    1)
    total_FDE = total_sum_fde        / np.maximum(total_count_vehicles, 1)
    total_TC  = total_sum_tc         / np.maximum(total_count_tc,       1)
    # print("total_ADE", total_ADE)
    total_DM = total_sum_dist        / np.maximum(total_count_vehicles, 1)
    
    total_min_ADE6 = sum_min_ADE_sc     / np.maximum(count_min_ADE_sc,    1)
    total_min_FDE6 = sum_min_FDE_sc        / np.maximum(count_min_FDE_sc , 1)
    total_min_TC6  = sum_min_TC_sc        / np.maximum(count_min_TC_sc,       1)
    
    # print("total_ADE6", total_min_ADE6)
    # fill top‐level mode_results
    for m in range(M):
        all_scenarios_results_ol['mode_results'].append({
            'mode': m,
            'ADE':  total_ADE[m],
            'FDE':  total_FDE[m],
            'TC':   total_TC[m],
        })

    all_scenarios_results_ol.update({
        'minADE6': total_min_ADE6,
        'minFDE6': total_min_FDE6,
        'minTC6':  total_min_TC6,
        'ADE1':    total_ADE[0],
        'FDE1':    total_FDE[0],
        'TC1':     total_TC[0],
        'DM': total_DM
    })
    print(total_count_steps)
    return all_scenarios_results_ol
                
        
if __name__ == "__main__":

    # === From own simulation_logs ===
    predictors = ["constant_velocity", "autobot_mini", "autobot", "wayformer_mini", "wayformer", "MTR_mini", "MTR", "ground_truth"]
    planners = ["MPCC", "RBMPCC"]
    
    # === OL Sim results ===
    print("\n--- OL Sim Metrics ---")
    directory_gt = '../simulation_log/ground_truth/RBMPCC/'
    all_scenarios_gt = get_ground_truth_predictions(directory_gt)

    for predictor in predictors:
        for planner in planners:
            directory = f'../simulation_log/{predictor}/{planner}'
            try:
                res = compute_ol_results_sim(directory, all_scenarios_gt)
                for k in res.keys():
                    if k == "single_scenario_results":
                        continue
                    print(f"{predictor}/{planner:<10} -> {k}: {res[k]}")
            except Exception as e:
                print(f"{predictor}/{planner:<10} -> Error in OL results: {e}")

    print("\n--- Closed-loop Metrics ---")
    for predictor in predictors:
        for planner in planners:
            path = f"../simulation_log/{predictor}/{planner}"
            try:
                v, _, j, delta, ddelta, d_min, tc, ax, ay = calculate_closed_loop_metrics_sim_logs(path)
                print(f"{predictor}/{planner:<10} -> "
                      f"v_mean: {v.mean():.2f}  v_std: {v.std():.2f}  "
                      f"j_mean: {j.mean():.2f}  j_std: {j.std():.2f}  "
                      f"delta_mean: {delta.mean():.2f}  delta_std: {delta.std():.2f}  "
                      f"ddelta_mean: {ddelta.mean():.2f}  ddelta_std: {ddelta.std():.2f}  "
                      f"d_min_mean: {d_min.mean():.3f}  d_min_std: {d_min.std():.3f}  "
                      f"tc_mean: {tc.mean():.3f}  tc_std: {tc.std():.3f}  "
                      f"ax_mean: {ax.mean():.2f}  ax_std: {ax.std():.2f}  "
                      f"ay_mean: {ay.mean():.2f}  ay_std: {ay.std():.2f}")
            except Exception as e:
                print(f"{predictor}/{planner:<10} -> Error: {e}")


    # === From NuPlan simulation logs ===
    print("\n--- NuPlan TTC Metrics ---")
    metric_file = 'time_to_collision_within_bound.parquet'  # e.g., 'no_ego_at_fault_collisions.parquet'
    value = 'min_time_to_collision_stat_value'

    for planner in planners:
        for predictor in predictors:
            dir_path = f'../../nuplan-devkit/nuplan/dataset/eval_planner_{planner}_{predictor}_tmp/my_planner/my_planner/metrics/'
            try:
                df = pd.read_parquet(dir_path + metric_file)
                ttcmin = df[value].to_numpy()
                ttcmin = ttcmin[ttcmin != np.inf]
                print(f"{predictor}/{planner:<10} -> Average Closed-Loop Scores:\n{df.tail(1)}\n")
                print(f"{predictor}/{planner:<10} -> TTC mean: {ttcmin.mean():.3f}  std: {ttcmin.std():.3f}")
                
            except Exception as e:
                print(f"{predictor}/{planner:<10} -> Error loading nuplan scores: {e}")


