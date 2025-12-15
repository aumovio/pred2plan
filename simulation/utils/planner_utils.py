"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import numpy as np
from math import inf as math_inf
from casadi import diagcat, vertcat, cos, sin, tan, inf
from planner.planning_utils.spline import Spline
from planner.planning_utils.obstacles import Obstacle
from planner.planning_utils.ScenarioSelection.DrivingCorridorSelector import DrivingCorridor
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObjectType
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from shapely.geometry import LineString, Point, Polygon, CAP_STYLE, JOIN_STYLE
from shapely.ops import unary_union,  nearest_points
from typing import List, Tuple, Dict
import yaml
import logging 
logger = logging.getLogger(__name__)



from tuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import (
    get_current_roadblock_candidates,
    BreadthFirstSearchRoadBlock,
    remove_route_loops
)

def route_roadblock_correction(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
) -> List[str]:
    """
    Applies several methods to correct route roadblocks.
    :param ego_state: class containing ego state
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param search_depth_backward: depth of forward BFS search, defaults to 15
    :param search_depth_forward:  depth of backward BFS search, defaults to 30
    :return: list of roadblock id's of corrected route
    """

    starting_block, starting_block_candidates = get_current_roadblock_candidates(
        ego_state, map_api, route_roadblock_dict
    )
    starting_block_ids = [roadblock.id for roadblock in starting_block_candidates]

    route_roadblocks = list(route_roadblock_dict.values())
    route_roadblock_ids = list(route_roadblock_dict.keys())

    # Fix 1: when agent starts off-route
    if starting_block.id not in route_roadblock_ids:
        # Backward search if current roadblock not in route
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[0], map_api, forward_search=False
        )
        (path, path_id), path_found = graph_search.search(
            starting_block_ids, max_depth=search_depth_backward
        )

        if path_found:
            route_roadblocks[:0] = path[:-1]
            route_roadblock_ids[:0] = path_id[:-1]

        else:
            # Forward search to any route roadblock
            graph_search = BreadthFirstSearchRoadBlock(
                starting_block.id, map_api, forward_search=True
            )
            (path, path_id), path_found = graph_search.search(
                route_roadblock_ids[:3], max_depth=search_depth_forward
            )

            if path_found:
                end_roadblock_idx = np.argmax(
                    np.array(route_roadblock_ids) == path_id[-1]
                )

                route_roadblocks = route_roadblocks[end_roadblock_idx + 1 :]
                route_roadblock_ids = route_roadblock_ids[end_roadblock_idx + 1 :]

                route_roadblocks[:0] = path
                route_roadblock_ids[:0] = path_id

            else: # in case of complete failure, just take longest viable path
                route_roadblocks = path
                route_roadblock_ids = path_id

    # Fix 2: check if roadblocks are linked, search for links if not
    roadblocks_to_append = {}
    for i in range(len(route_roadblocks) - 1):
        next_incoming_block_ids = [
            _roadblock.id for _roadblock in route_roadblocks[i + 1].incoming_edges
        ]
        is_incoming = route_roadblock_ids[i] in next_incoming_block_ids

        if is_incoming:
            continue

        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[i], map_api, forward_search=True
        )
        (path, path_id), path_found = graph_search.search(
            route_roadblock_ids[i + 1], max_depth=search_depth_forward
        )

        if path_found and path and len(path) >= 3:
            path, path_id = path[1:-1], path_id[1:-1]
            roadblocks_to_append[i] = (path, path_id)

    # append missing intermediate roadblocks
    offset = 1
    for i, (path, path_id) in roadblocks_to_append.items():
        route_roadblocks[i + offset : i + offset] = path
        route_roadblock_ids[i + offset : i + offset] = path_id
        offset += len(path)

    # Fix 3: cut route-loops
    route_roadblocks, route_roadblock_ids = remove_route_loops(
        route_roadblocks, route_roadblock_ids
    )

    return route_roadblock_ids


def get_control_points_path(ref_lane: InterpolatedPath, s_max) -> np.ndarray:
    """ Converts given Referencepath to Bezier Curve Path
    :param ref_lane: path from initial ego position to path goal
    :param i_max   : 
    :return cp_p: controlpoint path
    """
    
    #s_max = params["s_final"] 
    i_max = len(ref_lane._path)-1
    for i, p in enumerate(ref_lane._path):
        if p.progress > s_max:
            i_max=i
            break
         
        
    deg = 7
    spline = Spline(degree=deg)
    xy = np.array([[point.x, point.y] for point in ref_lane.get_sampled_path()[:i_max]]) #-xy0
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    s = np.block([0, np.cumsum(ds)])
    print(s, s.shape)
    xy = np.column_stack([
        np.interp(np.linspace(0, s[-1], 101), s, xy[:, 0]),
        np.interp(np.linspace(0, s[-1], 101), s, xy[:, 1]),
    ])
    #t0 = spline.inital_guess(xy)
    t_inf, converged, res = spline.borgespastva(xy, k=deg, maxiter=20)
    cp_p = spline.find_control_points(xy, t=t_inf) #+xy0
    c_p = spline.from_control_points(cp_p,N=1001)
    
    phi = spline.phi_from_control_points(cp_p,N=1001)
    c_p = np.hstack((spline.from_control_points(cp_p, N=1001),spline.phi_from_control_points(cp_p, N=1001).reshape(-1,1)))
    print(spline.pathlength(cp_p , np.linspace(0, 1, 1001))[-1], s_max)
    print(cp_p)
    return cp_p, c_p


def shift_movement(dt, x0, u0, l=3.25):
    """ Shifts ego position x0 to next position """
    f = vertcat(
        x0[3]*cos(x0[2]),    #x
        x0[3]*sin(x0[2]),    #y
        x0[3]*tan(x0[5])/l,  #psi
        x0[4],  #vx
        u0[0],  #ax
        u0[1],  #delta
        u0[2]   #s
    )
    st = x0 + dt*f
    return st


# def _ray_hit_nearest_forward(p, n, boundary_ls: LineString, ray_len: float):
#     """Cast a ray from p along unit vector n, return nearest forward intersection Point or None."""
#     x0, y0 = p
#     x1, y1 = x0 + ray_len * n[0], y0 + ray_len * n[1]
#     ray = LineString([(x0, y0), (x1, y1)])
#     inter = boundary_ls.intersection(ray)
#     if inter.is_empty:
#         return None
#     pts = []
#     gt = inter.geom_type
#     if gt == "Point":
#         pts = [inter]
#     elif gt == "MultiPoint":
#         pts = list(inter.geoms)
#     else:
#         # touch along a segment: take endpoints as candidates
#         geoms = [inter] if gt == "LineString" else getattr(inter, "geoms", [])
#         for g in geoms:
#             if hasattr(g, "coords"):
#                 c = list(g.coords)
#                 if c:
#                     pts.append(Point(c[0])); pts.append(Point(c[-1]))
#     # pick the nearest *forward* hit (positive projection along n)
#     best_pt, best_t = None, float("inf")
#     for pt in pts:
#         dx, dy = pt.x - x0, pt.y - y0
#         t = dx * n[0] + dy * n[1]
#         if t > 1e-6 and t < best_t:
#             best_t, best_pt = t, pt
#     return best_pt

# def get_road_boundaries(ego_path, drivable_polygon, buffer, c_p):
#     """
#     Loop-proof version:
#       - For each centerline sample (x,y,psi) in c_p, cast a left and a right ray to the polygon boundary.
#       - Distances are the nearest forward ray hits minus `buffer`.
#       - Returns: right_boundary(LineString), left_boundary(LineString),
#                  roadbound_dist_right(np.ndarray), roadbound_dist_left(np.ndarray)
#     """
#     # Build a boundary LineString to intersect with (exterior only; add interiors if you want islands as edges)
#     exterior = LineString(drivable_polygon.exterior.coords)
#     boundary_ls = exterior
#     # If you want to consider holes as possible boundaries too, uncomment:
#     # boundary_ls = unary_union([exterior] + [LineString(r.coords) for r in drivable_polygon.interiors])

#     # Long enough ray to always reach the boundary
#     minx, miny, maxx, maxy = drivable_polygon.bounds
#     diag = np.hypot(maxx - minx, maxy - miny)
#     ray_len = max(10.0, 2.0 * diag)

#     M = len(c_p)
#     left_hits, right_hits = [], []
#     roadbound_dist_left  = np.zeros(M, dtype=float)
#     roadbound_dist_right = np.zeros(M, dtype=float)

#     for i, (x, y, psi) in enumerate(c_p):
#         # Left/right unit normals from heading psi
#         nL = (-np.sin(psi),  np.cos(psi))
#         nR = ( np.sin(psi), -np.cos(psi))

#         pL = _ray_hit_nearest_forward((x, y), nL, boundary_ls, ray_len)
#         pR = _ray_hit_nearest_forward((x, y), nR, boundary_ls, ray_len)

#         # Fallback: if a ray misses (numerics/tangent), use nearest boundary point
#         if pL is None:
#             pL = boundary_ls.interpolate(boundary_ls.project(Point(x, y)))
#         if pR is None:
#             pR = boundary_ls.interpolate(boundary_ls.project(Point(x, y)))

#         dL = max(0.0, np.hypot(pL.x - x, pL.y - y) - buffer)
#         dR = max(0.0, np.hypot(pR.x - x, pR.y - y) - buffer)

#         left_hits.append(pL)
#         right_hits.append(pR)
#         roadbound_dist_left[i]  = dL
#         roadbound_dist_right[i] = dR

#     # --- your existing post-processing, preserved ---
#     threshold = 1.75  # minimal total width
#     sum_width = roadbound_dist_left + roadbound_dist_right
#     idx = np.where(sum_width < threshold)[0]
#     if idx.size:
#         diff = threshold - sum_width[idx]
#         roadbound_dist_left[idx]  += diff / 2.0
#         roadbound_dist_right[idx] += diff / 2.0

#     roadbound_dist_left  = np.maximum(0.7, roadbound_dist_left)
#     roadbound_dist_right = np.maximum(0.7, roadbound_dist_right)

#     # Optional: your caps/cleanups
#     roadbound_dist_left  = cap_errorenous_roadbound_dist(roadbound_dist_left)
#     roadbound_dist_right = cap_errorenous_roadbound_dist(roadbound_dist_right)

#     # Build sampled boundary polylines for visualization (no interior chords)
#     left_boundary  = LineString([(p.x, p.y) for p in left_hits])
#     right_boundary = LineString([(p.x, p.y) for p in right_hits])

#     return right_boundary, left_boundary, roadbound_dist_right, roadbound_dist_left


def get_road_boundaries(ego_path: InterpolatedPath, drivable_polygon, buffer, c_p):
    DEFAULT_LANEWIDTH = 3.5
    # get beginning and end of road boundary linesegment
    s0 = ego_path.get_start_progress()
    s_1 = ego_path.get_end_progress()
    s_m = (s_1-s0)/2
    _, path_x0, path_y0, path_psi0 = ego_path.get_state_at_progress(s0)
    _, path_x_1, path_y_1, path_psi_1 = ego_path.get_state_at_progress(s_1)
    _, path_xm, path_ym, path_psim = ego_path.get_state_at_progress(s_m)

    exterior_coords = np.array(drivable_polygon.exterior.coords)
    #point_coords = np.array( nearest_points(drivable_polygon.boundary, Point(path_x0,path_y0))[0].xy)
    # squared_distances = np.sum((exterior_coords - point_coords.T) ** 2, axis=1)
    # c1 = np.argmin(squared_distances)

    # Initialize variables
    min_distance = float("inf")
    closest_edge = None
    closest_indices = None
    point = Point(path_x0,path_y0)

    for i in range(len(exterior_coords) - 1):
        edge = LineString([exterior_coords[i], exterior_coords[i + 1]])  # Create edge as a LineString
        distance = point.distance(edge)  # Compute distance to the edge

        if distance < min_distance:
            min_distance = distance
            closest_edge = (exterior_coords[i], exterior_coords[i + 1])  # Store the closest edge
            closest_indices = (i, i + 1) 
            
    c1 = closest_indices[0]

    # exterior_coords = np.array(drivable_polygon.exterior.coords)
    # point_coords = np.array( nearest_points(drivable_polygon.boundary, Point(path_x_1, path_y_1))[0].xy)
    # squared_distances = np.sum((exterior_coords - point_coords.T) ** 2, axis=1)
    #c2 = np.argmin(squared_distances)
    min_distance = float("inf")
    closest_edge = None
    closest_indices = None
    point = Point(path_x_1,path_y_1)

    for i in range(len(exterior_coords) - 1):
        edge = LineString([exterior_coords[i], exterior_coords[i + 1]])  # Create edge as a LineString
        distance = point.distance(edge)  # Compute distance to the edge

        if distance < min_distance:
            min_distance = distance
            closest_edge = (exterior_coords[i], exterior_coords[i + 1])  # Store the closest edge
            closest_indices = (i, i + 1) 
            
    c2 = closest_indices[0]
    
    c1, c2 = (min(c1, c2), max(c1, c2))
    right_boundary = LineString(drivable_polygon.exterior.coords[c1:c2+1])
    left_boundary = LineString(drivable_polygon.exterior.coords[c2:]+drivable_polygon.exterior.coords[:c1+1])

    #check which one is left / right
    xy1 = np.array(nearest_points(right_boundary, Point(path_xm,path_ym))[0].xy)
    xy2 = np.array(nearest_points(left_boundary, Point(path_xm,path_ym))[0].xy)
    d_perp = np.array([[-np.sin(path_psim)], [np.cos(path_psim)]])
    midpoint = np.array([[path_xm], [path_ym]])
    d1 = np.dot((xy1 - midpoint).T, d_perp)
    d2 = np.dot((xy2 - midpoint).T, d_perp)

    if d1 > d2:
        left_boundary , right_boundary  = right_boundary , left_boundary 
    else:
        left_boundary , right_boundary  = left_boundary , right_boundary 

    # get dist to roadboundaries
    roadbound_dist_left, roadbound_dist_right = [], []
    for point_coords in c_p:
        point = Point(point_coords[:2])
        wl = point.distance(left_boundary) - buffer
        wr = point.distance(right_boundary) - buffer
        roadbound_dist_left.append(wl)
        roadbound_dist_right.append(wr)
        
        
    roadbound_dist_right = replace_negatives_with_closest_positive(np.array(roadbound_dist_right))
    roadbound_dist_left = replace_negatives_with_closest_positive(np.array(roadbound_dist_left))
    
    
    threshold = DEFAULT_LANEWIDTH/2.0 #1.0*2
    sum_width= roadbound_dist_right+roadbound_dist_left
    indices = np.where(sum_width< threshold)[0]
    diff = threshold - sum_width[indices]
    roadbound_dist_right[indices] += diff/2
    roadbound_dist_left[indices] += diff/2
    
    roadbound_dist_left = np.maximum(0.7, roadbound_dist_left)
    roadbound_dist_right = np.maximum(0.7, roadbound_dist_right)
    
    #fix probable mistakes:
    roadbound_dist_left = cap_errorenous_roadbound_dist(roadbound_dist_left)
    roadbound_dist_right = cap_errorenous_roadbound_dist(roadbound_dist_right)
    # indices = np.where( roadbound_dist_left< 0.7)[0]
    # print(indices, roadbound_dist_left[indices])
    # roadbound_dist_left[indices] =0.7
    return right_boundary, left_boundary, roadbound_dist_right, roadbound_dist_left


def cap_errorenous_roadbound_dist(array, threshold1=23, threshold2=7.75, cap_value=1.75, percentage=0.5):
    if np.mean(array > threshold1) >= percentage:
        print("Probably a mistake in the lane width calculation")
        array[array > threshold2] = cap_value
    return array


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


def replace_negatives_with_closest_positive(arr):

    subset = arr[:40]

    pos_indices = np.where(subset > 0)[0]
    neg_indices = np.where(subset < 0)[0]

    # Proceed only if negatives are fewer than positives
    if len(neg_indices) < len(pos_indices) and len(pos_indices) > 0:
        for neg_idx in neg_indices:
            # Find index of closest positive by minimal distance
            closest_pos_idx = pos_indices[np.argmin(np.abs(pos_indices - neg_idx))]
            arr[neg_idx] = arr[closest_pos_idx]

    return arr


def fix_angle_jumps(angles):
    diffs = np.diff(angles)
    wrapped_diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    fixed_angles = np.cumsum(np.concatenate(([angles[0]], wrapped_diffs)))
    return fixed_angles

def smooth_outliers(arr, threshold):
    arr = arr.copy()
    # Compute differences with previous and next elements
    diff_prev = np.abs(arr[1:-1] - arr[:-2])
    diff_next = np.abs(arr[1:-1] - arr[2:])
    diff_neighbors = np.abs(arr[:-2] - arr[2:])

    # Mask where the middle element differs a lot from neighbors, but neighbors are close to each other
    mask = ((diff_prev > threshold) | (diff_next > threshold)) & (diff_neighbors < threshold)

    # Compute corrected values
    corrected_values = (arr[:-2][mask] + arr[2:][mask]) / 2

    # Apply corrections
    arr[1:-1][mask] = corrected_values

    return arr



def wrap_diff(d):
    return (d + np.pi) % (2*np.pi) - np.pi

def clamp_orientation(angles, max_step):
    result = np.empty_like(angles)
    result[0] = angles[0]
    for i in range(1, len(angles)):
        d = wrap_diff(angles[i] - result[i-1])
        if np.abs(d) > max_step:
            result[i] = result[i-1]  # clamp to previous value
        else:
            result[i] = angles[i]

    return result

def _get_starting_edge_candidates(
        ego_state: EgoState,
        route_roadblocks: list[RoadBlockGraphEdgeMapObject]
) -> LaneGraphEdgeMapObject:
    """
    Return a list of candidate starting edges, ordered by:
    1) whether they contain the ego position
    2) distance to ego footprint
    """
    assert route_roadblocks is not None, "_route_raodblocks has not yet been initialized. Please call the initialize() function first!"
    assert len(route_roadblocks) >= 2, "_route_roadblocks should have at least 2 elements!"

    candidates = []

    for block in route_roadblocks[:2]:  # first and second roadblock
        for edge in block.interior_edges:
            on_lane = edge.contains_point(ego_state.center)
            dist = edge.polygon.distance(ego_state.car_footprint.geometry)
            # Lower is better: first by "on lane", then by distance
            score = (0 if on_lane else 1, dist)
            candidates.append((score, edge))

    assert candidates, "No interior edges found in first two roadblocks!"
    candidates.sort(key=lambda x: x[0])

    # Return only edges, ordered by score
    return [edge for _, edge in candidates]

def _breadth_first_search(ego_state: EgoState,  _route_roadblocks: List[RoadBlockGraphEdgeMapObject], _candidate_lane_edge_ids: List[str]) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
    """
    Performs iterative breadth first search to find a route to the target roadblock.
    :param ego_state: current ego state
    :return:
        - A route starting from the given start edge
        - A bool indicating if the route is successfully found. Successful means that there exists a path
          from the start edge to an edge contained in the end roadblock. If unsuccessful a longes route is given
    """
    assert(
        _route_roadblocks is not None
    ), "_route_roadblock has not yet been initialized. Please call the intitialize() function first!"
    assert(
        _candidate_lane_edge_ids is not None
    ), "_candidate_lane_edge_ids has not yet been initialized. Please call the intitialize() function first!"

    starting_edges = _get_starting_edge_candidates(ego_state, _route_roadblocks)
    best_route: List[LaneGraphEdgeMapObject] = []
    best_found: bool = False

    for starting_edge in starting_edges:
        graph_search = BreadthFirstSearch(starting_edge, _candidate_lane_edge_ids)
        offset = 1 if starting_edge.get_roadblock_id() == _route_roadblocks[1].id else 0
        route_plan, path_found = graph_search.search(_route_roadblocks[-1], len(_route_roadblocks[offset:]))

        if path_found:
            best_route = route_plan
            best_found = True
            break   
        elif len(route_plan) > len(best_route):
            best_route = route_plan
            best_found = False

    if not best_found:
        logger.warning(
            "EvaluationPlanner could not find a valid path to the target roadblock. Using longest route found instead."
        )

    return best_route, best_found

def oriented_bbox_from_polygon(polygon: Polygon):
    """
    Returns (length, width, orientation_rad) for the polygon's minimum rotated rectangle.
    - length  = longer side of the rectangle
    - width   = shorter side of the rectangle
    - orientation_rad = angle of the *length* side w.r.t. +x axis, normalized to [-π/2, π/2)
    """
    if polygon.is_empty or polygon.area == 0:
        return 0.0, 0.0, 0.0, 0.0

    rect = polygon.minimum_rotated_rectangle
    center = rect.centroid
    center = [center.x, center.y]
    coords = list(rect.exterior.coords)[:-1]  # 4 unique corner coords
    if len(coords) < 4:
        return 0.0, 0.0, 0.0
 
    def edge_len_vec(p0, p1):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return np.hypot(dx, dy), (dx, dy)
 
    edges = [edge_len_vec(coords[i], coords[(i+1) % 4]) for i in range(4)]
    lengths_only = [L for L, _ in edges]
 
    L = max(lengths_only)
    W = min(lengths_only)
 
    # orientation from the vector of the *long* side
    i_long = lengths_only.index(L)
    dx, dy = edges[i_long][1]
    angle = np.arctan2(dy, dx)  # radians, range [-π, π]
 
    # normalize to [-π/2, π/2)
    if angle >= np.pi/2:
        angle -= np.pi
    elif angle < -np.pi/2:
        angle += np.pi
 
    return center, L, W, angle,

def compute_boundaries(cp, wL, wR):
    cp = np.asarray(cp)
    x, y, psi = cp[:,0], cp[:,1], cp[:,2]
 
    wL = np.broadcast_to(wL, x.shape)
    wR = np.broadcast_to(wR, x.shape)
 
    psi = np.unwrap(psi)
    nLx = -np.sin(psi)
    nLy =  np.cos(psi)
 
    BL = np.column_stack([x + wL * nLx, y + wL * nLy])
    BR = np.column_stack([x - wR * nLx, y - wR * nLy])
    return BL, BR

def line_to_centered_polygon(coords: np.ndarray,
                             width: float,
                             cap_style: str = "round",
                             join_style: str = "round") -> Polygon:
    """
    Build a corridor polygon centered on a reference line.
 
    Parameters
    ----------
    coords : (M,2) array_like
        Polyline coordinates [[x0,y0],[x1,y1],...].
    width : float
        Total corridor width. The line will be centered, so each side gets width/2.
        (Make sure your coordinate units make sense for this width.)
    cap_style : {"round","flat","square"}
        End-cap style for the corridor.
    join_style : {"round","mitre","bevel"}
        Corner join style at polyline vertices.
 
    Returns
    -------
    shapely.geometry.Polygon
        The corridor polygon (possibly a MultiPolygon if the line self-intersects).
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
        raise ValueError("coords must be an (M,2) array with M >= 2")
 
    cap = {"round": CAP_STYLE.round, "flat": CAP_STYLE.flat, "square": CAP_STYLE.square}[cap_style]
    join = {"round": JOIN_STYLE.round, "mitre": JOIN_STYLE.mitre, "bevel": JOIN_STYLE.bevel}[join_style]
 
    line = LineString(coords)
    # Buffer by half the desired width so the line sits in the center.
    poly = line.buffer(width / 2.0, cap_style=cap, join_style=join)
    return poly