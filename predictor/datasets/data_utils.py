"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import os
from pathlib import Path
import compress_pickle as pickle
import time
import math
import numpy as np
from scipy.interpolate import interp1d
import torch
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from common_utils.time_tracking import timeit
# from scenarionet.common_utils import read_scenario, read_dataset_summary
from metadrive.scenario.scenario_description import ScenarioDescription
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
from predictor.datasets.types import object_type, polyline_type, track_type
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_submission_pb2
)
from typing import Optional, List

# import matplotlib
# # matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt

def is_ddp():
    return "WORLD_SIZE" in os.environ

@timeit
def first_fit_decreasing(files_to_indices, bucket_target_size=32):
    buckets_to_fill = []
    buckets_full = []

    for candidate_file, indices in files_to_indices.items():
        count = len(indices)
        placed = False
        # Try to place 'count' in an existing bucket
        for idx, bucket in enumerate(buckets_to_fill):
            if sum([len(indices) for file, indices in bucket]) + count <= bucket_target_size:
                bucket.append((candidate_file, files_to_indices[candidate_file]))
                placed = True
                if sum(len(indices) for file, indices in bucket) == bucket_target_size:
                    buckets_full.append(bucket)
                    del buckets_to_fill[idx]
                break
        # If it doesn't fit in any bucket, create a new one.
        if not placed:
            buckets_to_fill.append([(candidate_file, files_to_indices[candidate_file])])

    return buckets_full, buckets_to_fill

@timeit
def fill_rest_buckets(buckets_full, buckets_to_fill, bucket_target_size):
    samples_required= {}
    for idx, bucket in enumerate(buckets_to_fill):
        missing_samples = bucket_target_size - sum(len(indices) for file, indices in bucket)
        if missing_samples in samples_required.keys():
            samples_required[missing_samples].append(idx)
        else:
            samples_required[missing_samples] = [idx]

    buckets_to_fill_dict = {idx: bucket for idx, bucket in enumerate(buckets_to_fill)}
    leftovers = {}
    while len(buckets_to_fill) > 1:
        donor_bucket = buckets_to_fill[-1]
        last_file, indices = donor_bucket.pop()
        for i in range(len(indices)):
            num = len(indices)-i
            if num in samples_required.keys():
                receiver_bucket = buckets_to_fill_dict.pop(samples_required[num].pop(0))
                buckets_to_fill.remove(receiver_bucket)
                receiver_bucket.append((last_file, indices[:num]))
                buckets_full.append(receiver_bucket)
                indices = indices[num:]
                if not samples_required[num]:
                    _ = samples_required.pop(num)
        leftovers[last_file]= indices
        if not donor_bucket:
            _ = buckets_to_fill.pop()
    
    if buckets_to_fill:
        for file, indices in buckets_to_fill.pop():
            leftovers[file] = indices
    if sum([len(indices) for indices in leftovers.values()]) >= bucket_target_size:
        additional_buckets_full, buckets_to_fill = fill_into_buckets(leftovers, bucket_target_size)
        buckets_full += additional_buckets_full
    else:
        buckets_to_fill = [[(file, indices) for file, indices in leftovers.items()]]
    return buckets_full, buckets_to_fill

def fill_into_buckets(files_to_indices, bucket_target_size=32):
    files_to_indices = dict(sorted(files_to_indices.items(), key=lambda item: len(item[1]), reverse=True))
    buckets_full, buckets_to_fill = first_fit_decreasing(files_to_indices, bucket_target_size)
    buckets_full, buckets_to_fill = fill_rest_buckets(buckets_full, buckets_to_fill, bucket_target_size)
    samples_before = sum([len(indices) for indices in files_to_indices.values()])
    samples_bucketed = sum([len(indices) for bucket in buckets_full for file, indices in bucket])
    samples_disregarded = sum([len(indices) for file, indices in buckets_to_fill[-1]])
    assert samples_before - samples_disregarded == samples_bucketed
    return buckets_full, buckets_to_fill

def read_scenario(dataset_path, subset_path, scenario_file_name, compression=".pkl"):
    """Read a scenario file and return the Scenario Description instance.
    Note: We relaxed the assumption that it must be a pkl file to allow for compressions!

    Args:
        dataset_path: the path to the root folder of your dataset.
        mapping: the dict mapping return from read_dataset_summary.
        scenario_file_name: the file name to a scenario file

    Returns:
        The Scenario Description instance of that scenario.
    """
    file_path = Path(dataset_path, subset_path, scenario_file_name)
    # assert SD.is_scenario_file(file_path), "File: {} is not scenario file".format(file_path)
    with file_path.with_suffix(compression).open("rb") as f:
        # unpickler = CustomUnpickler(f)
        data = pickle.load(f)
    data = ScenarioDescription(data)
    return data


def generate_mask(current_index, total_length, interval):
    mask = []
    for i in range(total_length):
        # Check if the position is a multiple of the frequency starting from current_index
        if (i - current_index) % interval == 0:
            mask.append(1)
        else:
            mask.append(0)

    return np.array(mask)

def get_polyline_dir(polyline):
    # [x0-x0, x1-x0, x2-x1, ...] without making a rolled copy
    diff = np.diff(polyline, axis=0, prepend=polyline[0:1])
    # ‖diff‖ with shape (N, 1) to avoid [:, None] indexing
    norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    # clip in-place to avoid an extra array
    np.clip(norm, 1e-6, 1e9, out=norm)
    return diff / norm

def estimate_kalman_filter(history, prediction_horizon):
    """
    Predict the future position by running the kalman filter.

    :param history: 2d array of shape (length_of_history, 2)
    :param prediction_horizon: how many steps in the future to predict
    :return: the predicted position (x, y)

    Code taken from:
    On Exposing the Challenging Long Tail in Future Prediction of Traffic Actors
    """
    length_history = history.shape[0]
    z_x = history[:, 0]
    z_y = history[:, 1]
    v_x = 0
    v_y = 0
    for index in range(length_history - 1):
        v_x += z_x[index + 1] - z_x[index]
        v_y += z_y[index + 1] - z_y[index]
    v_x = v_x / (length_history - 1)
    v_y = v_y / (length_history - 1)
    x_x = np.zeros(length_history + 1, np.float32)
    x_y = np.zeros(length_history + 1, np.float32)
    P_x = np.zeros(length_history + 1, np.float32)
    P_y = np.zeros(length_history + 1, np.float32)
    P_vx = np.zeros(length_history + 1, np.float32)
    P_vy = np.zeros(length_history + 1, np.float32)

    # we initialize the uncertainty to one (unit gaussian)
    P_x[0] = 1.0
    P_y[0] = 1.0
    P_vx[0] = 1.0
    P_vy[0] = 1.0
    x_x[0] = z_x[0]
    x_y[0] = z_y[0]

    Q = 0.00001
    R = 0.0001
    K_x = np.zeros(length_history + 1, np.float32)
    K_y = np.zeros(length_history + 1, np.float32)
    K_vx = np.zeros(length_history + 1, np.float32)
    K_vy = np.zeros(length_history + 1, np.float32)
    for k in range(length_history - 1):
        x_x[k + 1] = x_x[k] + v_x
        x_y[k + 1] = x_y[k] + v_y
        P_x[k + 1] = P_x[k] + P_vx[k] + Q
        P_y[k + 1] = P_y[k] + P_vy[k] + Q
        P_vx[k + 1] = P_vx[k] + Q
        P_vy[k + 1] = P_vy[k] + Q
        K_x[k + 1] = P_x[k + 1] / (P_x[k + 1] + R)
        K_y[k + 1] = P_y[k + 1] / (P_y[k + 1] + R)
        x_x[k + 1] = x_x[k + 1] + K_x[k + 1] * (z_x[k + 1] - x_x[k + 1])
        x_y[k + 1] = x_y[k + 1] + K_y[k + 1] * (z_y[k + 1] - x_y[k + 1])
        P_x[k + 1] = P_x[k + 1] - K_x[k + 1] * P_x[k + 1]
        P_y[k + 1] = P_y[k + 1] - K_y[k + 1] * P_y[k + 1]
        K_vx[k + 1] = P_vx[k + 1] / (P_vx[k + 1] + R)
        K_vy[k + 1] = P_vy[k + 1] / (P_vy[k + 1] + R)
        P_vx[k + 1] = P_vx[k + 1] - K_vx[k + 1] * P_vx[k + 1]
        P_vy[k + 1] = P_vy[k + 1] - K_vy[k + 1] * P_vy[k + 1]

    k = k + 1
    x_x[k + 1] = x_x[k] + v_x * prediction_horizon
    x_y[k + 1] = x_y[k] + v_y * prediction_horizon
    P_x[k + 1] = P_x[k] + P_vx[k] * prediction_horizon * prediction_horizon + Q
    P_y[k + 1] = P_y[k] + P_vy[k] * prediction_horizon * prediction_horizon + Q
    P_vx[k + 1] = P_vx[k] + Q
    P_vy[k + 1] = P_vy[k] + Q
    return x_x[k + 1], x_y[k + 1]


def calculate_epe(pred, gt):
    diff_x = (gt[0] - pred[0]) * (gt[0] - pred[0])
    diff_y = (gt[1] - pred[1]) * (gt[1] - pred[1])
    epe = math.sqrt(diff_x + diff_y)
    return epe


def count_valid_steps_past(mask):
    reversed_mask = mask[::-1]  # Reverse the mask
    idx_of_first_zero = np.where(reversed_mask == 0)[0]  # Find the index of the first zero
    if len(idx_of_first_zero) == 0:
        return len(mask)  # If no zeros, return the length of the mask
    else:
        return idx_of_first_zero[0]  # Return the index of the first zero


def get_kalman_difficulty(output):
    """
    return the kalman difficulty at 2s, 4s, and 6s
    if the gt future is not valid up to the considered second, the difficulty is set to -1
    """
    for data_sample in output:
        # past trajectory of agent of interest
        past_trajectory = data_sample["obj_trajs"][0, :, :2]  # Time X (x,y)
        past_mask = data_sample["obj_trajs_mask"][0, :]
        valid_past = count_valid_steps_past(past_mask)
        past_trajectory_valid = past_trajectory[-valid_past:, :]  # Time(valid) X (x,y)

        # future gt trajectory of agent of interest
        gt_future = data_sample["obj_trajs_future_state"][0, :, :2]  # Time x (x, y)
        # Get last valid position
        valid_future = int(data_sample["center_gt_final_valid_idx"])

        kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s = -1, -1, -1
        try:
            if valid_future >= 19:
                # Get kalman future prediction at the horizon length, second argument is horizon length
                kalman_2s = estimate_kalman_filter(past_trajectory_valid, 20)  # (x,y)
                gt_future_2s = gt_future[19, :]
                kalman_difficulty_2s = calculate_epe(kalman_2s, gt_future_2s)

                if valid_future >= 39:
                    kalman_4s = estimate_kalman_filter(past_trajectory_valid, 40)  # (x,y)
                    gt_future_4s = gt_future[39, :]
                    kalman_difficulty_4s = calculate_epe(kalman_4s, gt_future_4s)

                    if valid_future >= 59:
                        kalman_6s = estimate_kalman_filter(past_trajectory_valid, 60)  # (x,y)
                        gt_future_6s = gt_future[59, :]
                        kalman_difficulty_6s = calculate_epe(kalman_6s, gt_future_6s)
        except:
            kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s = -1, -1, -1
        data_sample["kalman_difficulty"] = np.array([kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s])
    return

class TrajectoryType:
    STATIONARY = 0
    STRAIGHT_SLOW = 1
    STRAIGHT_FAST = 2
    STRAIGHT_RIGHT = 3
    STRAIGHT_LEFT = 4
    RIGHT_U_TURN = 5
    RIGHT_TURN = 6
    LEFT_U_TURN = 7
    LEFT_TURN = 8

# trajectory_correspondance = {0: "stationary", 1: "straight", 2: "straight_right",
#                              3: "straight_left", 4: "right_u_turn", 5: "right_turn",
#                              6: "left_u_turn", 7: "left_turn"}


def classify_track(start_point, end_point, start_velocity, end_velocity, start_heading, end_heading):
    # The classification strategy is taken from
    # waymo_open_dataset/metrics/motion_metrics_utils.cc#L28

    # Parameters for classification, taken from WOD
    kMaxSpeedForStationary = 2.0  # (m/s)
    kMaxDisplacementForStationary = 3.0  # (m)
    kMaxLateralDisplacementForStraight = 2.5  # (m)
    kMinLongitudinalDisplacementForUTurn = 0.0  # (m)
    kMaxAbsHeadingDiffForStraight = np.pi / 6.0  # (rad)

    kMaxSpeedForSlow = 11 # (m/s)
    kMaxLongitudinalDisplacementForStraightSlow = 45 # (m)

    x_delta = end_point[0] - start_point[0]
    y_delta = end_point[1] - start_point[1]

    final_displacement = np.hypot(x_delta, y_delta)
    heading_diff = end_heading - start_heading
    normalized_delta = np.array([x_delta, y_delta])
    rotation_matrix = np.array([[np.cos(-start_heading), -np.sin(-start_heading)],
                                [np.sin(-start_heading), np.cos(-start_heading)]])
    normalized_delta = np.dot(rotation_matrix, normalized_delta)
    start_speed = np.hypot(start_velocity[0], start_velocity[1])
    end_speed = np.hypot(end_velocity[0], end_velocity[1])
    max_speed = max(start_speed, end_speed)
    dx, dy = normalized_delta

    # Check for different trajectory types based on the computed parameters.
    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return TrajectoryType.STATIONARY
    if np.abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if np.abs(normalized_delta[1]) < kMaxLateralDisplacementForStraight:
            if max_speed < kMaxSpeedForSlow and final_displacement < kMaxLongitudinalDisplacementForStraightSlow:
                return TrajectoryType.STRAIGHT_SLOW 
            else: 
                return TrajectoryType.STRAIGHT_FAST
        return TrajectoryType.STRAIGHT_RIGHT if dy < 0 else TrajectoryType.STRAIGHT_LEFT
    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        return TrajectoryType.RIGHT_U_TURN if normalized_delta[
                                                  0] < kMinLongitudinalDisplacementForUTurn else TrajectoryType.RIGHT_TURN
    if dx < kMinLongitudinalDisplacementForUTurn:
        return TrajectoryType.LEFT_U_TURN
    return TrajectoryType.LEFT_TURN


# def interpolate_polyline(polyline, step=0.5):
#     if polyline.shape[0] == 1:
#         return polyline
#     polyline = polyline[:, :2]
#     distances = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1)))
#     distances = np.insert(distances, 0, 0)  # start with a distance of 0

#     # Create the new distance array
#     max_distance = distances[-1]
#     new_distances = np.arange(0, max_distance, step)

#     # Interpolate for x, y, z
#     new_polyline = []
#     for dim in range(polyline.shape[1]):
#         interp_func = interp1d(distances, polyline[:, dim], kind='linear')
#         new_polyline.append(interp_func(new_distances))

#     new_polyline = np.column_stack(new_polyline)
#     # add the third dimension back with zeros
#     new_polyline = np.concatenate((new_polyline, np.zeros((new_polyline.shape[0], 1))), axis=1)
#     return new_polyline

def interpolate_polyline(polyline, step=0.5):
    """
    Resample a polyline at ~`step` spacing (in meters), returning (M, 3) with z=0.
    Faster and lighter than interp1d; handles duplicate points robustly.
    """
    if polyline.shape[0] <= 1:
        return polyline
        # # Ensure 3D output
        # if polyline.shape[1] == 2:
        #     out = np.zeros((polyline.shape[0], 3), dtype=np.float32)
        #     out[:, :2] = polyline
        #     return out
        # return polyline.astype(np.float32, copy=False)

    # Work in XY only
    xy = np.asarray(polyline[:, :2], dtype=np.float32)

    # Differences and segment lengths
    d = np.diff(xy, axis=0)
    seg_len = np.hypot(d[:, 0], d[:, 1])

    # Drop zero-length segments to keep distances strictly increasing
    keep = seg_len > 1e-12
    if not np.all(keep):
        # Keep the first point plus any point that starts a non-zero segment
        idx = np.concatenate(([True], keep))
        xy = xy[idx]
        if xy.shape[0] <= 1:
            out = np.zeros((1, 3), dtype=np.float32)
            out[0, :2] = xy[0]
            return out
        d = np.diff(xy, axis=0)
        seg_len = np.hypot(d[:, 0], d[:, 1])

    # Cumulative distance (strictly increasing)
    n = xy.shape[0]
    dist = np.empty(n, dtype=np.float32)
    dist[0] = 0.0
    np.cumsum(seg_len, out=dist[1:])

    total = float(dist[-1])
    if total <= 1e-12:
        # All points collapsed — return a single point
        out = np.zeros((1, 3), dtype=np.float32)
        out[0, :2] = xy[0]
        return out

    # Choose number of samples so that the last one hits the endpoint exactly
    m = int(np.floor(total / step)) + 1
    new_dist = np.linspace(0.0, total, m, dtype=np.float32)

    # Fast 1D interpolation per axis (vectorized)
    x_new = np.interp(new_dist, dist, xy[:, 0]).astype(np.float32, copy=False)
    y_new = np.interp(new_dist, dist, xy[:, 1]).astype(np.float32, copy=False)

    # Stack and add z=0 column
    out = np.empty((m, 3), dtype=np.float32)
    out[:, 0] = x_new
    out[:, 1] = y_new
    out[:, 2] = 0.0
    return out

def get_heading(trajectory):
    # trajectory has shape (Time X (x,y))
    dx_ = np.diff(trajectory[:, 0])
    dy_ = np.diff(trajectory[:, 1])
    heading = np.arctan2(dy_, dx_)
    return heading


def get_trajectory_type(output):
    for data_sample in output:
        # Get last gt position, velocity and heading
        valid_end_point = int(data_sample["center_gt_final_valid_idx"])
        end_point = data_sample["obj_trajs_future_state"][0, valid_end_point, :2]  # (x,y)
        end_velocity = data_sample["obj_trajs_future_state"][0, valid_end_point, 2:]  # (vx, vy)
        # Get last heading, manually approximate it from the series of future position
        end_heading = get_heading(data_sample["obj_trajs_future_state"][0, :valid_end_point + 1, :2])[-1]

        # Get start position, velocity and heading.
        assert data_sample["obj_trajs_mask"][0, -1]  # Assumes that the start point is always valid
        start_point = data_sample["obj_trajs"][0, -1, :2]  # (x,y)
        start_velocity = data_sample["obj_trajs"][0, -1, -4:-2]  # (vx, vy)
        start_heading = 0.  # Initial heading is zero

        # Classify the trajectory
        try:
            trajectory_type = classify_track(start_point, end_point, start_velocity, end_velocity, start_heading,
                                             end_heading)
        except:
            trajectory_type = -1
        data_sample["trajectory_type"] = trajectory_type
    return

def densify_polyline(points: np.ndarray, step: float=0.5) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    out_pts = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        v = b - a
        dist = np.linalg.norm(v)
        if dist <= step:
            out_pts.append(b)
        else:
            unit = v / dist
            n = int(np.floor(dist / step))
            for i in range(1, n + 1):
                out_pts.append(a + unit * step * i)
            out_pts.append(b)
    return np.stack(out_pts, axis=0)

def motion_heading_alignment(positions, headings, observed) -> float:
    '''
    Calculate the alignment between position changes and measured headings
    1) by ratio between projected lateral/longitudenal motion
    2) by linear correlation between projected lateral/longitudenal motion
    '''
    observed = observed.astype(bool)
    x, y = positions[observed,0], positions[observed,1]
    psi = headings[observed][:-1]
    # build diffs
    dx = np.diff(x)
    dy = np.diff(y)
    # headings & laterals
    he = np.stack([np.cos(psi), np.sin(psi)], axis=1)
    la = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
    # projections
    lon = (dx*he[:,0] + dy*he[:,1])
    lat = (dx*la[:,0] + dy*la[:,1])
    R = np.sum(np.abs(lat)) / (np.sum(np.abs(lon))+1e-6)
    # correlation
    norms = np.hypot(dx, dy)
    move = norms > 1e-6
    m = np.stack([dx[move]/norms[move], dy[move]/norms[move]], axis=1)
    C = np.mean(np.einsum('ij,ij->i', m, he[move]))
    return R, C

def smooth_signal(signal, observed, s):
    from scipy.interpolate import make_splrep
    observed = observed.astype(bool)
    num_obs = np.sum(observed)
    t = np.arange(num_obs)
    spline = make_splrep(t, signal[observed], k=1, s=s)
    smooth_signal = spline(t)
    return smooth_signal

def signal_to_noise_ratio(positions: np.ndarray, observed: np.ndarray, s: float=1.0) -> float:
    '''
    Compute signal to noise ratio using cubic smoothing splines
    '''
    from scipy.interpolate import make_splrep
    observed = observed.astype(bool)
    num_obs = np.sum(observed)
    t = np.arange(num_obs)
    spl_x = make_splrep(t, positions[observed,0], k=3, s=s)
    spl_y = make_splrep(t, positions[observed,1], k=3, s=s)
    x_smooth = spl_x(t)
    y_smooth = spl_y(t)
    pos_smooth = np.stack([x_smooth, y_smooth], axis=1)
    residuals = positions[observed,:2] - pos_smooth
    signal_power = np.sum(np.diff(pos_smooth, axis=0), axis=0)**2
    noise_power = np.sum(np.diff(residuals, axis=0), axis=0)**2
    snr = signal_power/noise_power
    return snr


def plot_raw_data(scenario, current_idx=10, figsize=(12,12), save_path=None):
    n_colors = max(polyline_type.values()) + 1
    cmap = plt.cm.get_cmap('tab20', n_colors)
    fig, ax = plt.subplots(figsize=figsize)
    
    # --- plot map features ---
    for feat in scenario["map_features"].values():
        code = polyline_type.get(feat["type"], 7)
        color = cmap(code)
        # if it's a line
        if "polyline" in feat:
            poly = feat["polyline"][...,:2]
            ax.plot(poly[:,0], poly[:,1], color=color, linewidth=1)
        
        # if it's an area
        elif "polygon" in feat:
            coords = feat["polygon"][...,:2]
            patch = patches.Polygon(
                coords, closed=True,
                edgecolor=color,
                facecolor=color,
                alpha=0.4,
                linewidth=1
            )
            ax.add_patch(patch)

    # --- plot each track ---
    for tid, track in scenario["tracks"].items():
        st = track["state"]
        pos   = np.asarray(st["position"])   # (T,2)
        head  = np.asarray(st["heading"]).flatten()    # (T,)
        vel = np.asarray(st["velocity"]) # (T, 2)
        valid = np.asarray(st["valid"], bool).flatten() # (T,)
        w     = np.asarray(st["width"]).flatten()      # (T,)
        l     = np.asarray(st["length"]).flatten()     # (T,)

        # if the vehicle isn't present at current_idx, skip it entirely
        if not valid[current_idx]:
            continue
        cum_distance = np.sum(np.diff(pos[valid,:2], axis=0), axis=0)
        # print(tid, ": ", cum_distance, " ,", np.sum(valid))
        # if tid == "86d16e8363a25da6":
        #     breakpoint()

        if tid == scenario["metadata"]["sdc_id"] :
            color = "red"
        elif (np.abs(cum_distance) < np.sum(valid)/30).all():
            color = "blue"
        else:
            color = "black"

        x, y = pos[current_idx][:2]
        θ    = head[current_idx]
        length = l[current_idx]
        width  = w[current_idx]
        # car body
        rect = patches.Rectangle(
            (-length/2, -width/2), length, width,
            edgecolor=color,
            facecolor='none', lw=1.2
        )
        t = (patches.transforms.Affine2D()
                .rotate(θ)
                .translate(x, y)
                + ax.transData)
        rect.set_transform(t)
        ax.add_patch(rect)
        ax.text(x, y + width/2 + 0.3, str(tid), fontsize=8, ha='center', va='bottom', color=color)
        # heading arrow
        dx, dy = np.cos(θ)*length/2, np.sin(θ)*length/2
        ax.arrow(x, y, dx, dy,
                    head_width=width*0.3,
                    head_length=length*0.3,
                    fc='red' if tid==scenario["metadata"]["sdc_id"] else 'black',
                    ec='red' if tid==scenario["metadata"]["sdc_id"] else 'black')
        # 2) future trajectory (dashed) & little arrows
        future_idx = np.where((np.arange(len(valid)) > current_idx) & valid)[0]

        linear_vel = np.linalg.norm(vel,axis=1)
        vel_min, vel_max = np.min(linear_vel), np.max(linear_vel)
         
        import matplotlib.colors as mcolors
        colors = ["blue", "red"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=256)
        norm = mcolors.Normalize(vmin=vel_min, vmax=vel_max)
        if len(future_idx):
            # pts = pos[future_idx][...,:2]
            # dashed line
            # lc = LineCollection([pts], linestyles='--', linewidths=1)
            # ax.add_collection(lc)
            # tiny arrows at each waypoint
            for i in future_idx:
                v = linear_vel[i]
                rgba = cmap(norm(v))
                x, y = pos[i][...,:2]
                θ    = head[i]
                ax.arrow(x, y,
                         np.cos(θ)*0.5, np.sin(θ)*0.5,
                         head_width=0.2, head_length=0.3,
                         fc=rgba, ec=rgba)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f"Scenario from {scenario['metadata']['dataset']} raw {scenario['id']} @ frame {current_idx}")
    if save_path:
        save_path = Path(f"{save_path}/processed_scenario_{scenario['id']}_frame_{current_idx}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=1200)
        plt.close()
    else:
        matplotlib.use("TkAgg")
        plt.show()
    # plt.savefig(f"../data/processing/raw_{scenario['id']}_frame_{current_idx}.png", dpi=1200)
    # plt.close()

def plot_preprocessed_data(map_infos, track_infos, tracks_to_predict, metadata, current_idx=10, figsize=(12,12), save_path=None):
    """
    Draws map features and agent tracks from your preprocessed data.
    """
    # discrete colormap
    n_colors = max(polyline_type.values()) + 1
    cmap    = plt.cm.get_cmap('tab20', n_colors)
    fig, ax = plt.subplots(figsize=figsize)
    sdc_id = metadata["sdc_id"]
    scenario_id = metadata["scenario_id"]
    # 1) map features
    all_polylines = map_infos["all_polylines"]
    categories = [
        'lane', 'road_line', 'road_edge', 'sidewalk',
        'stop_sign', 'crosswalk', 'speed_bump', 'obstacles'
    ]
    for cat in categories:
        for feat in map_infos.get(cat, []):
            start, end = feat['polyline_index']
            coords = all_polylines[start:end, :2]  # drop z
            code   = polyline_type.get(feat['type'], 0)
            color  = cmap(code)

            if cat == 'crosswalk':
                ax.fill(coords[:,0], coords[:,1],
                        color=color, alpha=0.4, linewidth=1)
            else:
                ax.plot(coords[:,0], coords[:,1],
                        color=color, linewidth=1)

    # 2) agent tracks
    trajs      = track_infos['trajs']      # (num_agents, T, 10)
    object_ids = track_infos['object_id']
    num_agents, T, _ = trajs.shape
    draw_factor=1

    for i in range(num_agents):
        data = trajs[i]          # (T,10)

        # unpack & drop z
        pos3    = data[:, 0:3]             # x,y,z
        pos     = pos3[:, :2]              # x,y only
        length  = data[:, 3].reshape(-1)   # (T,)
        width   = data[:, 4].reshape(-1)
        vel = data[:, 7:9] # (T,2)
        heading = data[:, 6].reshape(-1)
        valid   = data[:, 9].astype(bool)  # (T,)
        tid     = object_ids[i].lower()

        cum_distance = np.sum(np.diff(pos[valid,:2], axis=0), axis=0)
        rms_noise = np.sqrt(np.mean(np.linalg.norm(np.diff(pos[valid,:2], axis=0), axis=1)**2))
        # print(tid, ": ", cum_distance, " ,", np.sum(valid), ", ", rms_noise)
        # skip agents not present now
        if not valid[current_idx]:
            continue

        # current vehicle rectangle + heading arrow
        x, y    = pos[current_idx]
        θ       = heading[current_idx]
        ℓ       = float(length[current_idx])
        w       = float(width[current_idx])


        if tid == sdc_id:
            color="red"
        elif tid in tracks_to_predict["track_id"]:
            color="blue"
        else:
            color="black"

        rect = patches.Rectangle(
            (-ℓ/2, -w/2), ℓ, w,
            edgecolor=color,
            facecolor='none', lw=1.2*draw_factor
        )
        t = (patches.transforms.Affine2D()
             .rotate(θ)
             .translate(x, y) + ax.transData)
        rect.set_transform(t)
        ax.add_patch(rect)
        ax.text(x, y + w/2 + 0.3, str(tid), fontsize=8, ha='center', va='bottom', color=color)
        dx, dy = np.cos(θ)*ℓ/2, np.sin(θ)*ℓ/2
        ax.arrow(
            x, y, dx, dy,
            head_width  = w * 0.3*draw_factor,
            head_length = ℓ * 0.3*draw_factor,
            fc          = rect.get_edgecolor(),
            ec          = rect.get_edgecolor()
        )

        # future trajectory
        fut_idx = np.where((np.arange(T) > current_idx) & valid)[0]

        linear_vel = np.linalg.norm(vel,axis=1)
        vel_min, vel_max = np.min(linear_vel), np.max(linear_vel)
        import matplotlib.colors as mcolors
        colors = ["blue", "red"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=256)
        norm = mcolors.Normalize(vmin=vel_min, vmax=vel_max)
        if fut_idx.size:
            pts = pos[fut_idx]
            # lc  = LineCollection(
            #     [pts], linestyles='--', linewidths=1, colors='gray'
            # )
            # ax.add_collection(lc)
            for j in fut_idx:
                v = linear_vel[j]
                rgba = cmap(norm(v))
                xj, yj = pos[j]
                θj      = heading[j]
                ax.arrow(
                    xj, yj,
                    np.cos(θj)*0.5, np.sin(θj)*0.5,
                    head_width  = 0.2*draw_factor,
                    head_length = 0.3*draw_factor,
                    fc          = rgba,
                    ec          = rgba
                )

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f"Scenario from {metadata['dataset']} preprocessed {scenario_id} @ frame {current_idx}")
    if save_path:
        save_path = Path(f"{save_path}/processed_scenario_{scenario_id}_frame_{current_idx}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=1200)
        plt.close()
    else:
        matplotlib.use("TkAgg")
        plt.show()


def plot_processed_data(data, current_idx=10, figsize=(12,12), save_path=None):
    """
    Draws map features and agent tracks from your processed data.
    """
    # discrete colormap
    n_colors = max(polyline_type.values()) + 1
    cmap    = plt.cm.get_cmap('tab20', n_colors)
    fig, ax = plt.subplots(figsize=figsize)
    # sdc_id = metadata["sdc_id"]
    scenario_id = data["scenario_id"]
    # 1) pull & convert map polylines
    mp = data['map_save']['traj_pos']
    if isinstance(mp, torch.Tensor):
        mp = mp.detach().cpu().numpy()
    # if mp is 2D, it's one long polyline; if 3D, iterate over the first dim
    if mp.ndim == 2:
        ax.plot(mp[:,0], mp[:,1], color='black', linewidth=1)
    else:
        for poly in mp:  # shape (L,2) each
            ax.plot(poly[:,0], poly[:,1],
                    color='black', linewidth=1)
            
    # 3) pull & convert agent arrays
    agents    = data['agent']
    pos     = agents['position'][...,:2]
    head    = agents['heading']
    shapes  = agents['shape'][...,:2]
    valid   = agents['valid_mask']
    ids      = agents['id']
    role    = agents["role"][...,0]
    to_predict = agents["train_mask"]

    # to numpy if torch
    def to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    pos    = to_np(pos)
    head   = to_np(head)
    shapes = to_np(shapes)
    role = to_np(role).astype(bool)
    to_predict = to_np(to_predict).astype(bool)
    valid  = to_np(valid).astype(bool)
    N, T = pos.shape[:2]

    # 4) plot each agent
    for i in range(N):
        if not valid[i, current_idx]:
            continue

        agent_role = role[i]
        agent_to_predict = to_predict[i]

        if agent_role:
            color = 'red'
        elif agent_to_predict:
            color = 'blue'
        else:
            color = 'black'

        # drop z if present
        pos2   = pos[i, :, :2]      # (T,2)
        θs     = head[i]            # (T,)
        length, width = shapes[i]   # two scalars
        # print(ids[i], ": ", head[i][current_idx])
        # current pose
        x, y = pos2[current_idx]
        θ    = θs[current_idx]

        # rectangle
        rect = patches.Rectangle(
            (-length/2, -width/2), length, width,
            edgecolor=color, facecolor='none', lw=1.2
        )
        tform = (patches.transforms.Affine2D()
                 .rotate(θ)
                 .translate(x, y)
                 + ax.transData)
        rect.set_transform(tform)
        ax.add_patch(rect)

        # heading arrow
        dx, dy = np.cos(θ)*length/2, np.sin(θ)*length/2
        ax.arrow(
            x, y, dx, dy,
            head_width  = width  * 0.2 *0.1,
            head_length = length * 0.3 *0.1,
            fc          = color,
            ec          = color
        )

        # future trajectory
        fut = np.where((np.arange(T) > current_idx) & valid[i])[0]
        if fut.size:
            # pts = pos2[fut]
            # lc  = LineCollection(
            #     [pts], linestyles='--',
            #     linewidths=1,
            #     colors = 'blue' if agent_id in predict_ids else 'gray'
            # )
            # ax.add_collection(lc)

            for j in fut:
                xj, yj = pos2[j]
                θj     = θs[j]
                ax.arrow(
                    xj, yj,
                    np.cos(θj)*0.5,
                    np.sin(θj)*0.5,
                    head_width  = width  * 0.3 *0.1,
                    head_length = length * 0.2 *0.1,
                    fc          = 'gray',
                    ec          = 'gray'
                )

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f"Processed Scenario @ frame {current_idx}")

    if save_path:
        save_path = Path(f"{save_path}/processed_scenario_{scenario_id[0]}_frame_{current_idx}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=1200)
        plt.close()
    else:
        matplotlib.use("TkAGG")
        plt.show()


def plot_tokenized_data(tokens_map, tokens_agent,
                        current_token_idx=2, scenario_id=0,
                        figsize=(12,12), save_path=None):
    """
    tokens_map:   tokens[0] dict
    tokens_agent: tokens[1] dict
    current_token_idx: which of the 14 agent‐token steps to highlight with a rectangle
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 1) map‐token polylines (unchanged)
    pos_map  = tokens_map["position"].cpu().numpy()       # [n_pl,2]
    ori_map  = tokens_map["orientation"].cpu().numpy()    # [n_pl]
    idx_map  = tokens_map["token_idx"].cpu().numpy()      # [n_pl]
    src      = tokens_map["token_traj_src"].cpu().numpy() # [n_token, 11*2]
    pl_type  = tokens_map["pl_type"].cpu().numpy()        # [n_pl]
    cmap_map = plt.cm.get_cmap("tab20")#, np.max(pl_type)+1)
    color_index=0
    for i in range(pos_map.shape[0]):
        color_index +=1
        pts_rel = src[idx_map[i]].reshape(-1,2)   # (11,2)
        θ = ori_map[i]
        R = np.array([[np.cos(θ), -np.sin(θ)],
                      [np.sin(θ),  np.cos(θ)]])
        pts = pts_rel.dot(R.T) + pos_map[i]
        ax.plot(pts[:,0], pts[:,1],
                color=cmap_map(color_index % cmap_map.N), linewidth=1)
                #color=cmap_map(pl_type[i]), linewidth=1)

    # 2) agent‐token rectangles + arrows
    gt_pos    = tokens_agent["gt_pos"].cpu().numpy()      # [n_ag,14,2]
    gt_head   = tokens_agent["gt_heading"].cpu().numpy()  # [n_ag,14]
    valid     = tokens_agent["valid_mask"].cpu().numpy()  # [n_ag,14]
    ag_shape  = tokens_agent["token_agent_shape"].cpu().numpy()  # [n_ag,2]
    ag_type   = tokens_agent["type"].cpu().numpy()        # [n_ag]
    cmap_ag   = plt.cm.get_cmap("tab10", np.max(ag_type)+1)
    n_ag, n_tok, _ = gt_pos.shape

    for i in range(n_ag):
        color = cmap_ag(ag_type[i])
        # 2a) rectangle at the CURRENT token
        if valid[i, current_token_idx]:
            w, ℓ  = ag_shape[i]
            x, y = gt_pos[i, current_token_idx]
            θ    = gt_head[i, current_token_idx]

            rect = patches.Rectangle(
                (-ℓ/2, -w/2), ℓ, w,
                edgecolor=color, facecolor='none', lw=1.2
            )
            t = (patches.transforms.Affine2D().rotate(θ).translate(x,y)
                 + ax.transData)
            rect.set_transform(t)
            ax.add_patch(rect)

        # 2b) heading arrows at _all_ valid token‐steps
        #    we'll make them small: length = 20% of vehicle length
        ℓ_i = ag_shape[i, 1]
        arrow_len = ℓ_i * 0.2
        head_w    = ag_shape[i,1] * 0.2
        head_l    = ℓ_i * 0.1

        for j in range(n_tok):
            if not valid[i,j]:
                continue
            xj, yj = gt_pos[i,j]
            θj      = gt_head[i,j]
            dx, dy  = np.cos(θj)*arrow_len, np.sin(θj)*arrow_len

            ax.arrow(
                xj, yj, dx, dy,
                head_width  = head_w,
                head_length = head_l,
                fc          = color,
                ec          = color,
                length_includes_head=True
            )

    ax.set_aspect('equal','box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f"Tokenized map + agent @ token‐step {current_token_idx}")
    if save_path:
        save_path = Path(f"{save_path}/tokenized_scenario_{scenario_id[0]}_token_{current_token_idx}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=1200)
        plt.close()
    else:
        matplotlib.use("TkAGG")
        plt.show()

def plot_scenario(
    scenario: scenario_pb2.Scenario,
    current_idx: int = 10,
    figsize: tuple = (12, 12),
    save_path: Optional[str] = None
):
    """
    Draws map features and agent tracks from a WOMD Scenario proto.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # 1) map edges
    for mf in scenario.map_features:
        if not mf.HasField('road_edge'):
            continue
        xs = [pt.x for pt in mf.road_edge.polyline]
        ys = [pt.y for pt in mf.road_edge.polyline]
        ax.plot(xs, ys, color='black', linewidth=1)

    # 2) Build agent arrays
    N = len(scenario.tracks)
    # assume all tracks have same number of states
    T = len(scenario.tracks[0].states)
    pos   = np.zeros((N, T, 3))
    head  = np.zeros((N, T))
    valid = np.zeros((N, T), dtype=bool)
    length = np.zeros(N)
    width  = np.zeros(N)
    # Roles / predict flags: you’ll need to tag these yourself,
    # e.g. based on object_type or a separate list
    role = np.zeros(N, dtype=bool)
    to_predict = np.zeros(N, dtype=bool)

    for i, track in enumerate(scenario.tracks):
        length[i] = track.states[0].length
        width[i]  = track.states[0].width
        for t, state in enumerate(track.states):
            pos[i, t, :]   = (state.center_x, state.center_y, state.center_z)
            head[i, t]     = state.heading
            valid[i, t]    = state.valid
        # Example: mark ego by object_type == 1
        role[i] = (track.object_type == 1)
        # Example: mark future to predict if any invalid in history
        to_predict[i] = not all(valid[i, :current_idx+1])

    # 3) Plot each agent at current frame
    for i in range(N):
        if not valid[i, current_idx]:
            continue

        # choose color
        if role[i]:
            color = 'red'
        elif to_predict[i]:
            color = 'blue'
        else:
            color = 'black'

        x, y = pos[i, current_idx, :2]
        θ    = head[i, current_idx]
        L, W  = length[i], width[i]

        # bounding box
        rect = patches.Rectangle(
            (-L/2, -W/2), L, W,
            edgecolor=color, facecolor='none', lw=1.2
        )
        tform = (patches.transforms.Affine2D()
                 .rotate(θ)
                 .translate(x, y)
                 + ax.transData)
        rect.set_transform(tform)
        ax.add_patch(rect)

        # heading arrow
        dx, dy = np.cos(θ)*L/2, np.sin(θ)*L/2
        ax.arrow(x, y, dx, dy,
                 head_width  = W*0.02,
                 head_length = L*0.03,
                 fc=color, ec=color)
        
        # agent ID label at the center of the patch
        ax.text(
            x, y,                 # position
            str(i),               # text = agent index
            color=color,          # same color as box/arrow
            fontsize=6,           # tweak as needed
            ha='center', va='center',
            backgroundcolor='white',  # optional: white bg for legibility
            alpha=0.3
        )

        # future short arrows
        fut = np.where((np.arange(T) > current_idx) & valid[i])[0]
        for t in fut:
            xt, yt = pos[i, t, :2]
            θt      = head[i, t]
            ax.arrow(xt, yt,
                     np.cos(θt)*0.5,
                     np.sin(θt)*0.5,
                     head_width  = W*0.015,
                     head_length = L*0.015,
                     fc='gray', ec='gray')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f"Scenario {scenario.scenario_id} @ frame {current_idx}")

    if save_path:
        out = Path(save_path) / f"simulated_scenario_{scenario.scenario_id}_frame_{current_idx}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300)
        plt.close()
    else:
        matplotlib.use("TkAgg")
        plt.show()


def plot_multiple_cur_and_split(cur_pos_list, split_polyline_list, figsize=(12,12)):
    """
    Overlay all original polylines and their fixed-window splits to form a map.
    
    Args:
        cur_pos_list:        list of (N,2) arrays or tensors
        split_polyline_list: list of (M,L,2) or (M,L,3) arrays or tensors
    """
    fig, ax = plt.subplots(1,2, figsize=figsize)
    cmap = plt.cm.get_cmap('tab20')
    color_idx = 0
    # Plot all original polylines in light gray
    for cur_pos in cur_pos_list:
        color = cmap(color_idx % cmap.N)
        color_idx += 1
        pts = np.asarray(cur_pos)
        ax[0].plot(pts[..., 0], pts[..., 1],
                    marker='o', linestyle='-', color=color, markersize=2, linewidth=0.5)
    
    # Plot all split windows with colored markers/arrows

    for split_polyline in split_polyline_list:

        
        sp = np.asarray(split_polyline)
        # sp shape: (num_windows, points_per_window, 2 or 3)
        for window in sp:
            # Choose a color per semantic polyline
            color = cmap(color_idx % cmap.N)
            color_idx += 1
            pts = window[:, :2]
            ax[1].plot(pts[..., 0], pts[..., 1],
                    marker='o', linestyle='-', color=color, markersize=2, linewidth=0.5)
            # # If headings are present, draw small arrows
            # if window.shape[1] == 3:
            #     heads = window[:, 2]
            #     dx = np.cos(heads)
            #     dy = np.sin(heads)
            #     ax[1].quiver(pts[..., 0], pts[..., 1], dx, dy,
            #               angles='xy', scale_units='xy', scale=5, width=0.002,
            #               color=color, alpha=0.8)
    
    for i in [0,1]:
        ax[i].set_aspect('equal', 'box')
        ax[i].set_xlabel('X')
        ax[i].set_ylabel('Y')
        ax[i].set_title('Overlayed Map: Raw Polylines & Tokenized Chunks')
    plt.tight_layout()
    plt.show()

def plot_all_scenario_rollouts(
    scenario: scenario_pb2.Scenario,
    scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
    rollout_indices: Optional[List[int]] = None,
    figsize: tuple = (12,12),
    scenario_id: Optional[str] = None,
    save_path: Optional[str] = None,
    current_idx: Optional[int] = 10,
):
    """
    Overlay every rollout from every scenario in `scenario_rollouts`,
    coloring each scenario uniquely.

    Args:
      scenario_rollouts: list returned by get_scenario_rollouts()
      rollout_indices: if provided, only plot these rollout‐IDs (same for all scenarios)
      figsize: figure size
      save_path: directory to save a PNG (one file) instead of plt.show()
    """
    num_rollouts = len(scenario_rollouts.joint_scenes)
    if rollout_indices is None:
        rollout_indices = list(range(num_rollouts))
        suffix = "all"
    else:
        suffix=str(rollout_indices)

    cmap = plt.cm.get_cmap('tab20', num_rollouts)

    fig, ax = plt.subplots(figsize=figsize)
    seen = set()

    # 1) map edges
    for mf in scenario.map_features:
        if not mf.HasField('road_edge'):
            continue
        xs = [pt.x for pt in mf.road_edge.polyline]
        ys = [pt.y for pt in mf.road_edge.polyline]
        ax.plot(xs, ys, color='black', linewidth=1)

    # 2) Build agent arrays
    N = len(scenario.tracks)
    # assume all tracks have same number of states
    T = len(scenario.tracks[0].states)
    pos   = np.zeros((N, T, 3))
    head  = np.zeros((N, T))
    valid = np.zeros((N, T), dtype=bool)
    length = np.zeros(N)
    width  = np.zeros(N)
    # Roles / predict flags: you’ll need to tag these yourself,
    # e.g. based on object_type or a separate list
    role = np.zeros(N, dtype=bool)
    to_predict = np.zeros(N, dtype=bool)

    for i, track in enumerate(scenario.tracks):
        length[i] = track.states[0].length
        width[i]  = track.states[0].width
        for t, state in enumerate(track.states):
            pos[i, t, :]   = (state.center_x, state.center_y, state.center_z)
            head[i, t]     = state.heading
            valid[i, t]    = state.valid
        # Example: mark ego by object_type == 1
        role[i] = (track.object_type == 1)
        # Example: mark future to predict if any invalid in history
        to_predict[i] = not all(valid[i, :current_idx+1])

    # 3) Plot each agent at current frame
    for i in range(N):
        if not valid[i, current_idx]:
            continue

        # choose color
        if role[i]:
            color = 'red'
        elif to_predict[i]:
            color = 'blue'
        else:
            color = 'black'

        x, y = pos[i, current_idx, :2]
        θ    = head[i, current_idx]
        L, W  = length[i], width[i]

        # bounding box
        rect = patches.Rectangle(
            (-L/2, -W/2), L, W,
            edgecolor=color, facecolor='none', lw=1.2
        )
        tform = (patches.transforms.Affine2D()
                 .rotate(θ)
                 .translate(x, y)
                 + ax.transData)
        rect.set_transform(tform)
        ax.add_patch(rect)

        # heading arrow
        dx, dy = np.cos(θ)*L/2, np.sin(θ)*L/2
        ax.arrow(x, y, dx, dy,
                 head_width  = W*0.02,
                 head_length = L*0.03,
                 fc=color, ec=color)
        
        # agent ID label at the center of the patch
        ax.text(
            x, y,                 # position
            str(i),               # text = agent index
            color=color,          # same color as box/arrow
            fontsize=3,           # tweak as needed
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.1),  # optional: white bg for legibility
            alpha=0.5
        )

    # 2) rollouts
    for r in rollout_indices:
        if r >= len(scenario_rollouts.joint_scenes):
            continue
        color = cmap(r)
        js = scenario_rollouts.joint_scenes[r]

        # label the first trajectory of this rollout (across all scenarios)
        do_label = (r not in seen)
        for traj in js.simulated_trajectories:
            xs, ys = traj.center_x, traj.center_y
            ax.plot(
                xs, ys,
                linewidth=1.0,
                alpha=0.6,
                color=color,
                label=(f"rollout {r}" if do_label else "_nolegend_")
            )
        if do_label:
            seen.add(r)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title("All Scenarios – colored by rollout index")
    #ax.legend(title="Rollout index", loc="best", fontsize="small")

    if save_path:
        out = Path(save_path) / f"rollouts_scenario_{scenario_id}_{suffix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=1200)
        plt.close()
    else:
        matplotlib.use("TkAGG")
        plt.show()