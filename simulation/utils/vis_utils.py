"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Polygon, Rectangle


def get_polyline_arc_length(points: np.ndarray) -> np.ndarray:
    """Calculate cumulative distance from the start for each vertex in a sequence."""
    steps = points[1:] - points[:-1]
    segment_lengths = np.hypot(steps[:, 0], steps[:, 1])
    total_lengths = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    return total_lengths


def interpolate_centerline(xy: np.ndarray, n_points: int):
    """Generate an interpolated set of evenly spaced points along a 2D centerline."""
    distances = get_polyline_arc_length(xy)
    target_positions = np.linspace(0, distances[-1], n_points)
    
    interpolated = np.empty((n_points, 2), dtype=xy.dtype)
    for dim in range(2):
        interpolated[:, dim] = np.interp(target_positions, distances, xy[:, dim])
    
    return interpolated


def plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location: np.ndarray,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
    alpha: float = 1.0,
    label: str = None,
    zorder: int = 50,
    fill: bool = True,
) -> None:
    """Render a rectangular bounding box at a specified location and orientation."""

    length, width = bbox_size
    radius = np.hypot(length, width)
    angle_offset = math.atan2(width, length)

    offset_x = (radius / 2) * math.cos(heading + angle_offset)
    offset_y = (radius / 2) * math.sin(heading + angle_offset)

    corner_x = cur_location[0] - offset_x
    corner_y = cur_location[1] - offset_y

    box = Rectangle(
        xy=(corner_x, corner_y),
        width=length,
        height=width,
        angle=np.degrees(heading),
        facecolor=color if fill else "none",
        edgecolor=color,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(box)

    if length > 1.0:
        tip_vector = 0.25 * length * np.array([math.cos(heading), math.sin(heading)])
        ax.arrow(
            cur_location[0],
            cur_location[1],
            tip_vector[0],
            tip_vector[1],
            color="white",
            zorder=zorder + 1,
            head_width=0.5,
        )


def plot_box(
    ax: plt.Axes,
    cur_location: np.ndarray,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
    alpha=1.0,
    label=None,
    zorder=50,
    fill=True,
    **kwargs,
) -> None:
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        xy=(pivot_x, pivot_y),
        width=bbox_length,
        height=bbox_width,
        angle=np.degrees(heading),
        fc=color if fill else "none",
        ec="dimgrey",
        alpha=alpha,
        label=label,
        zorder=zorder,
        **kwargs,
    )
    ax.add_patch(vehicle_bounding_box)


def plot_polygon(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    alpha=1.0,
    zorder=50,
    label=None,
) -> None:
    ax.add_patch(
        Polygon(
            np.stack([x, y], axis=1),
            closed=True,
            fc=color,
            ec="dimgrey",
            alpha=alpha,
            zorder=zorder,
            label=label,
        )
    )

def plot_polyline(
    ax,
    polylines: List[np.ndarray],
    cmap: str = "spring",
    linewidth: int = 3,
    arrow: bool = True,
    reverse: bool = False,
    alpha: float = 0.5,
    zorder: int = 100,
    color_change: bool = True,
    color=None,
    linestyle: str = "-",
    label=None,
) -> None:
    """Draw multiple polylines with customizable appearance and optional direction arrows."""

    # Normalize color input
    if isinstance(color, str):
        color = [color for _ in polylines]

    for idx, path in enumerate(polylines):
        smoothed = interpolate_centerline(path, 50)

        # Draw arrow at the end
        if arrow:
            end_point = smoothed[-1]
            vec = smoothed[-1] - smoothed[-2]
            vec /= np.linalg.norm(vec)
            col = plt.cm.get_cmap(cmap)(0) if color_change else color[idx]
            ax.quiver(
                end_point[0],
                end_point[1],
                vec[0],
                vec[1],
                scale_units="xy",
                scale=0.25,
                minlength=0.5,
                alpha=alpha,
                zorder=zorder - 1,
                color=col,
            )

        # Gradient color mode
        if color_change:
            distances = get_polyline_arc_length(smoothed)
            segments = np.concatenate(
                [smoothed[:-1, None], smoothed[1:, None]], axis=1
            )
            gradient = LineCollection(
                segments,
                cmap=cmap,
                norm=plt.Normalize(distances.min(), distances.max()),
                zorder=zorder,
                alpha=alpha,
                label=label,
            )
            gradient.set_array(distances[::-1] if reverse else distances)
            gradient.set_linewidth(linewidth)
            ax.add_collection(gradient)

        # Single-color mode
        else:
            ax.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                color=color[idx],
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
                label=label,
            )

def plot_direction(ax, anchors, dir_vecs, zorder=1):
    for anchor, dir_vec in zip(anchors, dir_vecs):
        if np.linalg.norm(dir_vec) == 0:
            continue
        vec = dir_vec / np.linalg.norm(dir_vec)
        ax.arrow(
            anchor[0],
            anchor[1],
            vec[0],
            vec[1],
            color="black",
            zorder=zorder,
            head_width=0.2,
            head_length=0.2,
        )


def plot_trajectory_with_angle(ax, traj):
    if traj.shape[-1] > 3:
        angle_phase_num = traj.shape[-1] - 2
        phase = 2 * np.pi * np.arange(angle_phase_num) / angle_phase_num
        xn = traj[..., -3:]  # (N, 3)
        angles = -np.arctan2(
            np.sum(np.sin(phase) * xn, axis=-1), np.sum(np.cos(phase) * xn, axis=-1)
        )
    else:
        angles = traj[..., -1]

    ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=2)
    for p, angle in zip(traj, angles):
        ax.arrow(
            p[0],
            p[1],
            np.cos(angle) * 0.5,
            np.sin(angle) * 0.5,
            color="black",
            zorder=1,
            head_width=0.3,
            head_length=0.2,
        )
    ax.axis("equal")


def plot_crosswalk(ax, edge1, edge2):
    polygon = np.concatenate([edge1, edge2[::-1]])
    ax.add_patch(
        Polygon(
            polygon, closed=True, fc="k", alpha=0.3, hatch="///", ec="w", linewidth=2
        )
    )


def plot_sdc(
    ax,
    center,
    heading,
    width,
    length,
    steer=0.0,
    color="pink",
    fill=True,
    wheel=True,
    **kwargs,
):
    vec_heading = np.array([np.cos(heading), np.sin(heading)])
    vec_tan = np.array([np.sin(heading), -np.cos(heading)])

    front_left_wheel = center + 1.419 * vec_heading + 0.35 * width * vec_tan
    front_right_wheel = center + 1.419 * vec_heading - 0.35 * width * vec_tan
    wheel_heading = heading + steer
    wheel_size = (0.8, 0.3)

    plot_box(
        ax, center, heading, color=color, fill=fill, bbox_size=(length, width), **kwargs
    )

    if wheel:
        plot_box(
            ax,
            front_left_wheel,
            wheel_heading,
            color="k",
            fill=True,
            bbox_size=wheel_size,
            **kwargs,
        )
        plot_box(
            ax,
            front_right_wheel,
            wheel_heading,
            color="k",
            fill=True,
            bbox_size=wheel_size,
            **kwargs,
        )


def plot_lane_area(ax, left_bound, right_bound, fc="silver", alpha=1.0, ec=None):
    polygon = np.concatenate([left_bound, right_bound[::-1]])
    ax.add_patch(
        Polygon(polygon, closed=True, fc=fc, alpha=alpha, ec=None, linewidth=2)
    )

