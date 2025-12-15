"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import numpy as np
import torch
from scipy.interpolate import interp1d
from typing import Optional

def get_polylines_from_polygon(polygon: np.ndarray) -> np.ndarray:
    # polygon: [4, 3]
    l1 = np.linalg.norm(polygon[1, :2] - polygon[0, :2])
    l2 = np.linalg.norm(polygon[2, :2] - polygon[1, :2])

    def _pl_interp_start_end(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        length = np.linalg.norm(start - end)
        unit_vec = (end - start) / length
        pl = []
        for i in range(int(length) + 1):  # 4.5 -> 5 [0,1,2,3,4]
            x, y, z = start + unit_vec * i
            pl.append([x, y, z])
        pl.append([end[0], end[1], end[2]])
        return np.array(pl)

    if l1 > l2:
        pl1 = _pl_interp_start_end(polygon[0], polygon[1])
        pl2 = _pl_interp_start_end(polygon[2], polygon[3])
    else:
        pl1 = _pl_interp_start_end(polygon[0], polygon[3])
        pl2 = _pl_interp_start_end(polygon[2], polygon[1])
    return np.concatenate([pl1, pl1[::-1], pl2, pl2[::-1]], axis=0)


def _interpolating_polyline(polylines, distance=0.5, split_distance=5):
    # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
    # correct polylines
    diffs = np.diff(polylines, axis=0)
    mask = np.concatenate(([True], np.any(diffs !=0, axis=1)))
    polylines = polylines[mask]
    # Compute heading between consecutive points
    dx = np.diff(polylines[:, 0])
    dy = np.diff(polylines[:, 1])
    heading = np.arctan2(dy, dx)
    # Append last heading to maintain same length
    heading = np.concatenate([heading, heading[-1:]])

    dist_along_path_list = []
    polylines_list = []
    euclidean_dists = np.linalg.norm(polylines[1:, :2] - polylines[:-1, :2], axis=-1)
    euclidean_dists = np.concatenate([[0], euclidean_dists])
    heading_diff = np.minimum(2*np.pi - np.abs(np.diff(heading)), np.abs(np.diff(heading)))
    heading_diff = np.concatenate([heading_diff, heading_diff[-1:]])
    breakpoints = np.where(euclidean_dists > 3)[0]
    breakpoints = np.where(heading_diff>np.pi/8)[0]
    #(heading_diff>np.pi/4)
    breakpoints = np.concatenate([[-1], breakpoints, [polylines.shape[0]]])
    for i in range(1, breakpoints.shape[0]):
        start = breakpoints[i - 1] + 1
        end = breakpoints[i] +1
        dist_along_path_list.append(
            np.cumsum(euclidean_dists[start:end]) - euclidean_dists[start]
        )
        polylines_list.append(polylines[start:end])
    # if breakpoints.size>2:
    #     breakpoint()
    #     plot_polylines(polylines, polylines_list)
    multi_polylines_list = []
    for idx in range(len(dist_along_path_list)):
        if len(dist_along_path_list[idx]) < 2:
            continue
        dist_along_path = dist_along_path_list[idx]
        polylines_cur = polylines_list[idx]
        # Create interpolation functions for x and y coordinates
        fxy = interp1d(dist_along_path, polylines_cur, axis=0)
        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
        new_dist_along_path = np.concatenate(
            [new_dist_along_path, dist_along_path[[-1]]]
        )

        # Combine the new x and y coordinates into a single array
        new_polylines = fxy(new_dist_along_path)
        polyline_size = int(split_distance / distance)
        if new_polylines.shape[0] >= (polyline_size + 1):
            padding_size = (
                new_polylines.shape[0] - (polyline_size + 1)
            ) % polyline_size
            final_index = (
                new_polylines.shape[0] - (polyline_size + 1)
            ) // polyline_size + 1
        else:
            padding_size = new_polylines.shape[0]
            final_index = 0
        multi_polylines = None
        new_polylines = torch.from_numpy(new_polylines)
        new_heading = torch.atan2(
            new_polylines[1:, 1] - new_polylines[:-1, 1],
            new_polylines[1:, 0] - new_polylines[:-1, 0],
        )
        new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
        new_polylines = torch.cat([new_polylines, new_heading], -1)
        if new_polylines.shape[0] >= (polyline_size + 1):
            multi_polylines = new_polylines.unfold(
                dimension=0, size=polyline_size + 1, step=polyline_size
            )
            multi_polylines = multi_polylines.transpose(1, 2)
            multi_polylines = multi_polylines[:, ::5, :]
        if padding_size >= 3:
            last_polyline = new_polylines[final_index * polyline_size :]
            last_polyline = last_polyline[
                torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()
            ]
            if multi_polylines is not None:
                multi_polylines = torch.cat(
                    [multi_polylines, last_polyline.unsqueeze(0)], dim=0
                )
            else:
                multi_polylines = last_polyline.unsqueeze(0)
        if multi_polylines is None:
            continue
        multi_polylines_list.append(multi_polylines)
    if len(multi_polylines_list) > 0:
        multi_polylines_list = torch.cat(multi_polylines_list, dim=0).to(torch.float32)
    else:
        multi_polylines_list = None
    return multi_polylines_list

import matplotlib.pyplot as plt

def plot_polylines(polyline, chunks):
    plt.figure(figsize=(12,12))

    x_full, y_full = polyline[:,0], polyline[:,1]
    plt.plot(x_full, y_full,
             color='black', linewidth=2,
             label='Original polyline')

    cmap = plt.get_cmap('tab10')
    for idx, chunk in enumerate(chunks):
        x_c, y_c = chunk[:,0], chunk[:,1]
        plt.plot(x_c, y_c,
                 marker='o', linestyle='-',
                 color=cmap(idx % cmap.N),
                 label=f'Chunk {idx+1} ({chunk.shape[0]} pts)')
    # Decorations
    plt.title("polyline interpolation")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis('equal')  # preserve aspect ratio
    plt.show()


# from scipy.spatial.distance import euclidean
# import math

def _interpolating_polyline_original(polylines, distance=0.5, split_distance=5):
    # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter

    # Compute heading between consecutive points
    dx = np.diff(polylines[:, 0])
    dy = np.diff(polylines[:, 1])
    heading = np.arctan2(dy, dx)
    # Append last heading to maintain same length
    heading = np.concatenate([heading, heading[-1:]])

    dist_along_path_list = [[0]]
    polylines_list = [[polylines[0]]]
    for i in range(1, polylines.shape[0]):
        euclidean_dist = euclidean(polylines[i, :2], polylines[i - 1, :2])
        heading_diff = min(abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1])),
                           abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1]) + math.pi))
        if heading_diff > math.pi / 4 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > math.pi / 8 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > 0.1 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif euclidean_dist > 10:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        else:
            dist_along_path_list[-1].append(dist_along_path_list[-1][-1] + euclidean_dist)
            polylines_list[-1].append(polylines[i])
    # plt.plot(polylines[:, 0], polylines[:, 1])
    # plt.savefig('tmp.jpg')
    new_x_list = []
    new_y_list = []
    multi_polylines_list = []
    for idx in range(len(dist_along_path_list)):
        if len(dist_along_path_list[idx]) < 2:
            continue
        dist_along_path = np.array(dist_along_path_list[idx])
        polylines_cur = np.array(polylines_list[idx])
        # Create interpolation functions for x and y coordinates
        fx = interp1d(dist_along_path, polylines_cur[:, 0])
        fy = interp1d(dist_along_path, polylines_cur[:, 1])
        # fyaw = interp1d(dist_along_path, heading)

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
        new_dist_along_path = np.concatenate([new_dist_along_path, dist_along_path[[-1]]])
        # Use the interpolation functions to generate new x and y coordinates
        new_x = fx(new_dist_along_path)
        new_y = fy(new_dist_along_path)
        # new_yaw = fyaw(new_dist_along_path)
        new_x_list.append(new_x)
        new_y_list.append(new_y)

        # Combine the new x and y coordinates into a single array
        new_polylines = np.vstack((new_x, new_y)).T
        polyline_size = int(split_distance / distance)
        if new_polylines.shape[0] >= (polyline_size + 1):
            padding_size = (new_polylines.shape[0] - (polyline_size + 1)) % polyline_size
            final_index = (new_polylines.shape[0] - (polyline_size + 1)) // polyline_size + 1
        else:
            padding_size = new_polylines.shape[0]
            final_index = 0
        multi_polylines = None
        new_polylines = torch.from_numpy(new_polylines)
        new_heading = torch.atan2(new_polylines[1:, 1] - new_polylines[:-1, 1],
                                  new_polylines[1:, 0] - new_polylines[:-1, 0])
        new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
        new_polylines = torch.cat([new_polylines, new_heading], -1)
        if new_polylines.shape[0] >= (polyline_size + 1):
            multi_polylines = new_polylines.unfold(dimension=0, size=polyline_size + 1, step=polyline_size)
            multi_polylines = multi_polylines.transpose(1, 2)
            multi_polylines = multi_polylines[:, ::5, :]
        if padding_size >= 3:
            last_polyline = new_polylines[final_index * polyline_size:]
            last_polyline = last_polyline[torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()]
            if multi_polylines is not None:
                multi_polylines = torch.cat([multi_polylines, last_polyline.unsqueeze(0)], dim=0)
            else:
                multi_polylines = last_polyline.unsqueeze(0)
        if multi_polylines is None:
            continue
        multi_polylines_list.append(multi_polylines)
    if len(multi_polylines_list) > 0:
        multi_polylines_list = torch.cat(multi_polylines_list, dim=0)
    else:
        multi_polylines_list = None
    return multi_polylines_list