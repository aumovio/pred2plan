"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# input
# ego: (16,3)
# agents: (16,n,3)
# map: (150,n,3)

# visualize all of the agents and ego in the map, the first dimension is the time step,
# the second dimension is the number of agents, the third dimension is the x,y,theta of the agent
# visualize ego and other in different colors, visualize past and future in different colors,past is the first 4 time steps, future is the last 12 time steps
# visualize the map, the first dimension is the lane number, the second dimension is the x,y,theta of the lane
# you can discard the last dimension of all the elements

def check_loaded_data(data, index=0):
    agents = np.concatenate([data['obj_trajs'][..., :2], data['obj_trajs_future_state'][..., :2]], axis=-2)
    map = data['map_polylines']

    agents = agents[index]
    map = map[index]
    ego_index = data['track_index_to_predict'][index]
    ego_agent = agents[ego_index]

    fig, ax = plt.subplots()

    def draw_line_with_mask(point1, point2, color, line_width=4):
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    # Function to draw lines with a validity check

    # Plot the map with mask check
    for lane in map:
        if lane[0, -3] in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if lane[i, -3] > 0:
                draw_line_with_mask(lane[i, :2], lane[i, -2:], color='grey', line_width=1)

    # Function to draw trajectories
    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    # Draw trajectories for other agents
    for i in range(agents.shape[0]):
        draw_trajectory(agents[i], line_width=2)
    draw_trajectory(ego_agent, line_width=2, ego=True)
    # Set labels, limits, and other properties
    vis_range = 100
    # ax.legend()
    ax.set_xlim(-vis_range + 30, vis_range + 30)
    ax.set_ylim(-vis_range, vis_range)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    # As defined in the common_utils.py file
    # traj_type = { 0: "stationary", 1: "straight", 2: "straight_right",
    #         3: "straight_left", 4: "right_u_turn", 5: "right_turn",
    #         6: "left_u_turn", 7: "left_turn" }
    #
    # kalman_2s, kalman_4s, kalman_6s = list(data["kalman_difficulty"][index])
    #
    # plt.title("%s -- Idx: %d -- Type: %s  -- kalman@(2s,4s,6s): %.1f %.1f %.1f" % (1, index, traj_type[data["trajectory_type"][0]], kalman_2s, kalman_4s, kalman_6s))
    # # Return the axes object
    # plt.show()

    # Return the PIL image
    return plt
    # return ax


def concatenate_images(images, rows, cols):
    # Determine individual image size
    width, height = images[0].size

    # Create a new image with the total size
    total_width = width * cols
    total_height = height * rows
    new_im = Image.new('RGB', (total_width, total_height))

    # Paste each image into the new image
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        new_im.paste(image, (col * width, row * height))

    return new_im


def concatenate_varying(image_list, column_counts):
    if not image_list or not column_counts:
        return None

    # Assume all images have the same size, so we use the first one to calculate ratios
    original_width, original_height = image_list[0].size
    total_height = original_height * column_counts[0]  # Total height is based on the first column

    columns = []  # To store each column of images

    start_idx = 0  # Starting index for slicing image_list

    for count in column_counts:
        # Calculate new height for the current column, maintaining aspect ratio
        new_height = total_height // count
        scale_factor = new_height / original_height
        new_width = int(original_width * scale_factor)

        column_images = []
        for i in range(start_idx, start_idx + count):
            # Resize image proportionally
            resized_image = image_list[i].resize((new_width, new_height), Image.Resampling.LANCZOS)
            column_images.append(resized_image)

        # Update start index for the next batch of images
        start_idx += count

        # Create a column image by vertically stacking the resized images
        column = Image.new('RGB', (new_width, total_height))
        y_offset = 0
        for img in column_images:
            column.paste(img, (0, y_offset))
            y_offset += img.height

        columns.append(column)

    # Calculate the total width for the new image
    total_width = sum(column.width for column in columns)

    # Create the final image to concatenate all column images
    final_image = Image.new('RGB', (total_width, total_height))
    x_offset = 0
    for column in columns:
        final_image.paste(column, (x_offset, 0))
        x_offset += column.width

    return final_image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D
from matplotlib.patches import FancyArrowPatch


def visualize_prediction(batch, prediction, draw_index: int = 0, bounds: float = 60.0):
    """
    NuPlan-style visualization of a single sample.

    - Centers the scene around a chosen "target" agent.
    - Map lanes in light gray.
    - Past/future ground truth trajectories for all agents.
    - Multi-modal predicted trajectories for the target agent, colored by probability.
    """

    input_dict = batch["input_dict"]

    # --- Extract data for this sample ----------------------------------------------------
    map_lanes = input_dict["map_polylines"][draw_index].cpu().numpy()          # (num_lanes, num_pts, 30)
    map_mask = input_dict["map_polylines_mask"][draw_index].cpu().numpy()      # (num_lanes, num_pts)

    past_traj = input_dict["obj_trajs"][draw_index].cpu().numpy()              # (num_agents, T_past, 29)
    future_traj = input_dict["obj_trajs_future_state"][draw_index].cpu().numpy()   # (num_agents, T_fut, 4)
    past_traj_mask = input_dict["obj_trajs_mask"][draw_index].cpu().numpy()        # (num_agents, T_past)
    future_traj_mask = input_dict["obj_trajs_future_mask"][draw_index].cpu().numpy()   # (num_agents, T_fut)

    scenario_id = input_dict["scenario_id"][draw_index]

    pred_future_prob = prediction["predicted_probability"][draw_index].detach().cpu().numpy()   # (num_modes,)
    pred_future_traj = prediction["predicted_trajectory"][draw_index].detach().cpu().numpy()    # (num_modes, T_fut, 5)

    # Take x/y from first two dims everywhere
    map_xy = map_lanes[..., :2]            # (num_lanes, num_pts, 2)
    past_xy = past_traj[..., :2]           # (num_agents, T_past, 2)
    future_xy = future_traj[..., :2]       # (num_agents, T_fut, 2)
    pred_future_xy = pred_future_traj[..., :2]  # (num_modes, T_fut, 2)

    # Lane type: you previously used the last 20 dims of the first point
    map_type = map_lanes[..., 0, -20:]     # (num_lanes, 20)

    # --- Choose center/target agent index -----------------------------------------------
    center_idx = 0
    if "center_objects" in input_dict:
        center_val = input_dict["center_objects"][draw_index]
        center_idx = int(center_val.item()) if hasattr(center_val, "item") else int(center_val)

    # last valid past state of the center agent
    center_mask = past_traj_mask[center_idx].astype(bool)      # (T_past,)
    if center_mask.any():
        center_last_idx = np.where(center_mask)[0][-1]
    else:
        center_last_idx = past_traj_mask.shape[1] - 1

    center_xy_world = past_xy[center_idx, center_last_idx]     # (2,)

    # --- Recenter everything -------------------------------------------------------------
    map_xy_centered = map_xy - center_xy_world[None, None, :]
    past_xy_centered = past_xy - center_xy_world[None, None, :]
    future_xy_centered = future_xy - center_xy_world[None, None, :]
    pred_future_xy_centered = pred_future_xy - center_xy_world[None, None, :]

    # --- Create fig/axes in NuPlan-ish style --------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.axis("equal")
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    ax.axis("off")

    ax.text(
        0.02,
        0.98,
        f"{scenario_id}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # --- Helper: gradient polyline like in NuPlan ---------------------------------------
    def plot_polyline(polyline_xy, cmap_name="Blues", linewidth=2.0, alpha=1.0, zorder=3):
        if len(polyline_xy) < 2:
            return
        t = np.linspace(0.0, 1.0, len(polyline_xy))
        pts = polyline_xy.reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = plt.Normalize(t.min(), t.max())
        lc = LineCollection(
            seg,
            cmap=cmap_name,
            norm=norm,
            array=t,
            linewidth=linewidth,
            alpha=alpha,
        )
        lc.set_zorder(zorder)
        ax.add_collection(lc)

    # --- Draw map -----------------------------------------------------------------------
    for lane_idx, lane_xy in enumerate(map_xy_centered):
        lane_type_onehot = map_type[lane_idx]
        lane_type_idx = np.argmax(lane_type_onehot)

        # keep your original filter
        if lane_type_idx in [1, 2, 3]:
            mask = map_mask[lane_idx].astype(bool)
            valid_xy = lane_xy[mask]
            if len(valid_xy) < 2:
                continue

            ax.plot(
                valid_xy[:, 0],
                valid_xy[:, 1],
                color="lightgray",
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
                zorder=2,
            )
        else:
            mask = map_mask[lane_idx].astype(bool)
            valid_xy = lane_xy[mask]
            if len(valid_xy) < 2:
                continue

            ax.plot(
                valid_xy[:, 0],
                valid_xy[:, 1],
                color="lightgray",
                alpha=0.9,
                linewidth=2.0,
                linestyle="-",
                zorder=0,
            )

    # --- Draw past trajectory (center agent only) --------------------------------------
    num_agents = past_xy_centered.shape[0]

    xy = past_xy_centered[center_idx]                 # (T_past, 2)
    mask = past_traj_mask[center_idx].astype(bool)    # (T_past,)
    valid_xy = xy[mask]
    if len(valid_xy) >= 2:
        plot_polyline(
            valid_xy,
            cmap_name="autumn",
            linewidth=2.5,
            alpha=0.9,
            zorder=5,
        )

    # --- Draw predicted future trajectories (target only, multi-modal) ------------------
    orig = cm.get_cmap("Greens")
    blues_dark = LinearSegmentedColormap.from_list(
        "greens_dark", 
        orig(np.linspace(0.5, 1.0, 256))  # skip lightest 20%
    )


    if pred_future_xy_centered.ndim == 3:
        probs = pred_future_prob.astype(float)  # (num_modes,)
        if probs.max() > 0:
            probs_norm = probs / probs.max()
        else:
            probs_norm = probs

        for mode_idx, traj_xy in enumerate(pred_future_xy_centered):
            if len(traj_xy) < 2:
                continue

            p = probs_norm[mode_idx]
            color = blues_dark(p)

            alpha = 0.6

            ax.plot(
                traj_xy[:, 0],
                traj_xy[:, 1],
                color=color,
                linewidth=2.0,
                alpha=alpha,
                zorder=6,
            )

    # --- Draw future GT trajectory (center agent only) ---------------------------------
    xy = future_xy_centered[center_idx]                  # (T_fut, 2)
    mask = future_traj_mask[center_idx].astype(bool)     # (T_fut,)
    valid_xy = xy[mask]
    if len(valid_xy) >= 2:
        plot_polyline(
            valid_xy,
            cmap_name="autumn_r",
            linewidth=2.5,
            alpha=0.9,
            zorder=7,
        )


    # --- Draw rotated bounding boxes at last past step for each agent -------------------
    num_agents = past_traj.shape[0]

    for agent_idx in range(num_agents):
        mask = past_traj_mask[agent_idx].astype(bool)
        if mask.any():
            last_idx = np.where(mask)[0][-1]
        else:
            last_idx = past_traj_mask.shape[1] - 1

        # state in *world* frame (un-centered)
        last_state = past_traj[agent_idx, last_idx]              # (29,)
        # position in centered frame (after ego-translation)
        last_xy_centered = past_xy_centered[agent_idx, last_idx] # (2,)

        center_x, center_y = last_xy_centered

        # assuming: [3]=width, [4]=length, [5]=height
        width, length, _ = last_state[3:6]
        heading_sin = last_state[23]
        heading_cos = last_state[24]
        heading = np.arctan2(heading_sin, heading_cos)  # radians

        if agent_idx == center_idx:
            ec = "#ff7f0e"   # orange
            lw = 1.5
        else:
            ec = "#1f77b4"   # blue-ish
            lw = 0.8

        # local rect centered at (0,0) in its own frame
        rect = patches.Rectangle(
            (-width / 2.0, -length / 2.0),
            width,
            length,
            linewidth=lw,
            edgecolor=ec,
            facecolor="none",
            linestyle="-",
            zorder=8,
            transform=Affine2D()
            .rotate(heading)           
            .translate(center_x, center_y)
            + ax.transData,
        )
        arrow_len = 1.2 * length   # "forward" length

        dx = arrow_len * np.cos(heading)
        dy = arrow_len * np.sin(heading)

        arrow = FancyArrowPatch(
            (center_x, center_y),
            (center_x + dx, center_y + dy),
            arrowstyle="-",
            mutation_scale=8,      # arrow head size
            linewidth=0.8,
            color=ec,              # match rectangle color
            alpha=0.9,             # you can also tune arrow alpha separately
            zorder=9,
        )
        ax.add_patch(arrow)
        ax.add_patch(rect)

    # --- Colorbar for prediction probabilities ------------------------------------------
    if pred_future_prob is not None and len(pred_future_prob) > 0:
        sm = plt.cm.ScalarMappable(cmap=blues_dark, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.016, pad=0.04)
        cbar.set_label("Predicted mode probability")

    plt.tight_layout(pad=0.1)
    return fig