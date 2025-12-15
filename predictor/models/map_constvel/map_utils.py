"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import numpy as np
import math
from copy import deepcopy
from shapely import LineString, Point, Polygon
from collections import deque
from statistics import median
from typing import Callable, Iterable, List, Dict, Any


def search_end_states(
    associated_lanes: List[Dict[str, Any]],
    max_distance: float,
    map_data,
    max_paths: int = 1000,
    include_start_neighbour = True,
    include_end_neighbour = True,
) -> List[Dict[str, Any]]:
    """
    Collect all possible end states (position, heading, curvature) reachable within
    `max_distance` starting from the associated lanes AND their closest neighbors.
    Also include end states obtained by projecting each found endpoint onto the
    closest adjacent neighbor lanes. Returns UNIQUE end states.

    Each returned dict contains:
      - lane_idx   : int
      - s          : float (arclength on lane, meters)
      - point      : shapely.geometry.Point
      - xy         : (x, y) convenience tuple
      - heading    : float (radians)
      - curvature  : float (1/m)
      - distance   : float (distance traveled along the path to reach this end)
      - path       : List[int] (ordered lane indices from enumerate_successor_paths)
      - source     : 'path_end' | 'neighbor_of_end'
    """
    # Ensure lanes exist
    if 'converted_lanes' not in map_data:
        map_data = convert_map(map_data)
    lanes = map_data['converted_lanes']

    if not associated_lanes:
        return []

    # ---- Build start seeds (like in search_paths) -----------------------------
    seeds: Dict[int, Dict[str, Any]] = {}
    for rec in associated_lanes:
        idx = int(rec['lane_idx'])
        s   = float(rec['s_on_lane'])
        cp  = rec['closest_point']
        d   = float(rec.get('distance', 0.0))
        L   = lanes[idx].length
        s   = max(0.0, min(L, s))
        if (idx not in seeds) or (d < seeds[idx]['dist']):
            seeds[idx] = {'s': s, 'point': cp, 'dist': d}

    # Add closest neighbors (of associated lanes only) as extra seeds
    if include_start_neighbour:
        initial_idxs = list(seeds.keys())
        for idx in initial_idxs:
            cp = seeds[idx]['point']
            for nx in (find_neighbour(idx, map_data, get_closest=True) or []):
                if nx in seeds:
                    continue
                Lnx = lanes[nx].length
                s_nx = max(0.0, min(Lnx, float(lanes[nx].project(cp))))
                seeds[nx] = {'s': s_nx, 'point': cp, 'dist': seeds[idx]['dist']}

    # ---- Helpers --------------------------------------------------------------
    S_BIN = 0.05  # 5 cm bin for uniqueness on arclength
    def s_key(l_idx: int, s_val: float) -> tuple:
        return (int(l_idx), int(round(float(s_val) / S_BIN)))

    def make_end_state(l_idx: int, s_val: float, dist: float, path: List[int], source: str) -> Dict[str, Any]:
        line = lanes[l_idx]
        s_c  = max(0.0, min(line.length, float(s_val)))
        pt   = line.interpolate(s_c)
        hd   = lane_heading_at_s(line, s_c)       # radians
        kv   = lane_curvature_at_s(line, s_c)     # 1/m
        return {
            'lane_idx': int(l_idx),
            's': s_c,
            'point': pt,
            'xy': (pt.x, pt.y),
            'heading': hd,
            'curvature': kv,
            'distance': float(dist),
            'path': list(path),
            'source': source
        }

    # ---- Enumerate forward paths; collect unique end states -------------------
    unique_ends: Dict[tuple, Dict[str, Any]] = {}   # key -> end_state (with path)
    total_cap = max_paths

    for lane_idx, info in seeds.items():
        paths = enumerate_successor_paths(
            start_idx=lane_idx,
            start_s=info['s'],
            max_distance=max_distance,
            map_data=map_data,
            max_paths=max_paths
        )
        for p in paths:
            end_lane = p['end']['lane_idx']
            end_s    = p['end']['s']
            key      = s_key(end_lane, end_s)
            es       = make_end_state(end_lane, end_s, p['distance'], p['path'], source='path_end')

            # Deduplicate by (lane_idx, s bin) — keep the one with larger traveled distance
            if (key not in unique_ends) or (es['distance'] > unique_ends[key]['distance']):
                unique_ends[key] = es

            if len(unique_ends) >= total_cap:
                break
        if len(unique_ends) >= total_cap:
            break

    # ---- For each end, add neighbor-of-end states (keep ORIGINAL path) -------
    if include_end_neighbour:
        neighbor_additions: Dict[tuple, Dict[str, Any]] = {}
        for es in list(unique_ends.values()):
            l_idx = es['lane_idx']
            pt    = es['point']
            for nx in (find_neighbour(l_idx, map_data, get_closest=True) or []):
                line_n = lanes[nx]
                s_n    = line_n.project(pt)
                key_n  = s_key(nx, s_n)
                # Use the ORIGINAL path and distance from the parent end state
                es_n   = make_end_state(nx, s_n, es['distance'], es['path'], source='neighbor_of_end')

                # Keep the variant with larger traveled distance
                if (key_n not in neighbor_additions) or (es_n['distance'] > neighbor_additions[key_n]['distance']):
                    neighbor_additions[key_n] = es_n

                if len(unique_ends) + len(neighbor_additions) >= total_cap:
                    break
            if len(unique_ends) + len(neighbor_additions) >= total_cap:
                break

        # Merge neighbor additions while respecting uniqueness rule
        for k, es_n in neighbor_additions.items():
            if (k not in unique_ends) or (es_n['distance'] > unique_ends[k]['distance']):
                unique_ends[k] = es_n

    # ---- Final sort & return --------------------------------------------------
    all_ends = list(unique_ends.values())
    
    # ---- collapse to at most ONE end state per lane_idx -----------------
    def _source_rank(src: str) -> int:
            # higher is better
            return 1 if src == 'path_end' else 0

    best_per_lane: Dict[int, Dict[str, Any]] = {}
    for es in all_ends:
        k = es['lane_idx']
        cur = best_per_lane.get(k)
        if cur is None:
            best_per_lane[k] = es
            continue

        # Compare by (source_rank desc, distance desc, s desc)
        cand_key = (_source_rank(es['source']), es['distance'], es['s'])
        cur_key  = (_source_rank(cur['source']), cur['distance'], cur['s'])
        if cand_key > cur_key:
            best_per_lane[k] = es

    out = list(best_per_lane.values())
    
    out.sort(key=lambda e: (-e['distance'], e['lane_idx'], e['s']))
    return out

def search_paths( 
    associated_lanes: List[Dict[str, Any]],  # output from associate_vehicle_to_lanes
    max_distance: float,
    map_data,                                # expects map_data['converted_lanes']
    max_paths: int = 1000
) -> List[Dict[str, Any]]:
    """
    Build start states from the associated lane(s) and their adjacent neighbors,
    then enumerate forward paths up to `max_distance`. Deduplicate by the ordered
    sequence of lane indices ('path'), keeping the variant that reaches the
    largest traveled distance.

    Returns: list of path dicts as produced by `enumerate_successor_paths`.
    """
    # Ensure lanes exist
    if 'converted_lanes' not in map_data:
        map_data = convert_map(map_data)  # safe no-op if already present
    lanes = map_data['converted_lanes']

    if not associated_lanes:
        return []

    # ---- 1) Build unique start seeds: {lane_idx -> {'s': float, 'point': Point, 'dist': float}} ----
    # For a lane seen multiple times in `associated_lanes`, keep the entry with the smallest vehicle-to-lane distance.
    seeds: Dict[int, Dict[str, Any]] = {}
    for rec in associated_lanes:
        idx = rec['lane_idx']
        s   = float(rec['s_on_lane'])
        cp  = rec['closest_point']  # shapely Point near the vehicle
        d   = float(rec.get('distance', 0.0))
        # Clamp s to lane length just in case
        L = lanes[idx].length
        s = max(0.0, min(L, s))
        if (idx not in seeds) or (d < seeds[idx]['dist']):
            seeds[idx] = {'s': s, 'point': cp, 'dist': d}

    # ---- 2) Add closest adjacent neighbors of each associated lane as extra seeds ----
    # For each associated lane, ask for its closest neighbors (left/right). Project the same nearby point onto them.
    initial_seed_idxs = list(seeds.keys())  # only neighbors of associated lanes, not neighbors-of-neighbors
    for idx in initial_seed_idxs:
        cp = seeds[idx]['point']
        neigh_idxs = find_neighbour(idx, map_data, get_closest=True) or []
        for nx in neigh_idxs:
            if nx in seeds:
                continue  # already seeded
            # project the same nearby point to get a sensible start s on the neighbor lane
            s_nx = lanes[nx].project(cp)
            Lnx  = lanes[nx].length
            s_nx = max(0.0, min(Lnx, float(s_nx)))
            seeds[nx] = {'s': s_nx, 'point': cp, 'dist': seeds[idx]['dist']}

    # ---- 3) Enumerate forward paths from every seed ----
    unique_by_signature: Dict[tuple, Dict[str, Any]] = {}
    total_cap = max_paths  # overall cap on unique results to avoid explosion

    for lane_idx, info in seeds.items():
        start_s = info['s']
        paths = enumerate_successor_paths(
            start_idx=lane_idx,
            start_s=start_s,
            max_distance=max_distance,
            map_data=map_data,
            max_paths=max_paths  # per-seed cap
        )

        for p in paths:
            sig = tuple(p['path'])  # use lane sequence as the uniqueness key
            # Keep the variant with the largest traveled distance (ties keep existing)
            if (sig not in unique_by_signature) or (p['distance'] > unique_by_signature[sig]['distance']):
                unique_by_signature[sig] = p

            if len(unique_by_signature) >= total_cap:
                break  # hit global cap
        if len(unique_by_signature) >= total_cap:
            break

    # ---- 4) Return unique paths (optionally sorted) ----
    out = list(unique_by_signature.values())
    # Sort: longer traveled distance first, then fewer lane transitions
    out.sort(key=lambda r: (-r['distance'], len(r['path'])))
    return out
    

def enumerate_successor_paths(
    start_idx: int,
    start_s: float,
    max_distance: float,
    map_data,  
    max_paths: int = 1000
) -> List[Dict[str, Any]]:
    """
    Walk forward along successors up to `max_distance` from (start_idx, start_s).

    Returns a list of path dicts. Each dict contains:
      - path: [lane_idx0, lane_idx1, ...]
      - distance: total distance actually traversed (<= max_distance)
      - end: {'lane_idx': last_idx, 's': arclength on last lane where we stop}
      - segments: [{'lane_idx', 's_start', 's_end', 'length'} ...]  # exact per-lane spans

    Notes:
      * If the cutoff happens inside the first lane, you'll get a single path
        ending at (start_idx, start_s + max_distance).
      * If successors branch, you get one entry per branch that reaches the cutoff
        or a dead-end first.
      * Loops are pruned with a simple dominance check so the search can’t blow up.
    """
    lanes = map_data['converted_lanes']
    lengths = [ls.length for ls in lanes]
    L0 = lengths[start_idx]
    if start_s < 0.0: start_s = 0.0
    if start_s > L0:   start_s = L0

    # If we end inside the starting lane, return immediately.
    remaining_on_first = max(0.0, L0 - start_s)

    if max_distance <= remaining_on_first:
        end_s = start_s + max_distance
        return [{
            'path': [start_idx],
            'distance': max_distance,
            'end': {'lane_idx': start_idx, 's': end_s},
            'segments': [{
                'lane_idx': start_idx,
                's_start': start_s,
                's_end': end_s,
                'length': max_distance
            }]
        }]
    # Otherwise we will at least traverse to the end of start_idx
    outputs = []
    q = deque()
    # State tuple:
    # (path, lane_idx, s_on_lane, remaining_distance, cum_distance, segments)
    q.append((
        [start_idx],
        start_idx,
        start_s,
        max_distance,
        0.0,
        [{
            'lane_idx': start_idx,
            's_start': start_s,
            's_end': L0,
            'length': remaining_on_first
        }]
    ))

    # Best-remaining-distance seen per lane to prune dominated states
    # (entering a lane with less remaining than before is never better)
    best_remaining = {(start_idx): max_distance}

    while q and len(outputs) < max_paths:
        path, lane_idx, s, rem, cum, segments = q.popleft()
        L = lengths[lane_idx]
        spent_here = max(0.0, L - s)
        rem2 = rem - spent_here
        cum2 = cum + spent_here

        if rem2 <= 0.0:
            # We actually ended inside this lane segment; trim the last segment precisely
            last = segments[-1].copy()
            last['s_end'] = s + rem
            last['length'] = rem
            outputs.append({
                'path': path,
                'distance': cum + rem,
                'end': {'lane_idx': lane_idx, 's': s + rem},
                'segments': segments[:-1] + [last]
            })
            continue

        # Move to successors
        next_ids = find_next(lane_idx, map_data)
        if next_ids.size==0: # no successor
            # Dead end at end of this lane
            outputs.append({
                'path': path,
                'distance': cum2,
                'end': {'lane_idx': lane_idx, 's': L},
                'segments': segments
            })
            continue
        for nx in next_ids:
            # Prune dominated expansions into the same lane
            if rem2 <= best_remaining.get(nx, -1.0) - 1e-6:
                continue
            best_remaining[nx] = rem2
            Lnx = lengths[nx]
            if rem2 <= Lnx:
                # We stop inside the successor lane
                outputs.append({
                    'path': path + [nx],
                    'distance': cum2 + rem2,
                    'end': {'lane_idx': nx, 's': rem2},
                    'segments': segments + [{
                        'lane_idx': nx,
                        's_start': 0.0,
                        's_end': rem2,
                        'length': rem2
                    }]
                })
            else:
                # Traverse the whole successor and keep going
                q.append((
                    path + [nx],
                    nx,
                    0.0,
                    rem2,
                    cum2,
                    segments + [{
                        'lane_idx': nx,
                        's_start': 0.0,
                        's_end': Lnx,
                        'length': Lnx
                    }]
                ))
    return outputs


def associate_vehicle_to_lanes_incrementally(vehicle_xy,
                                             vehicle_heading,
                                             map_data,
                                             max_distance=6.0,
                                             start_distance = 2.0,
                                             max_heading_diff_rad= np.pi/4,
                                             eps=0.2
                                            ):
    results = []
    while len(results) == 0 and start_distance<= max_distance:
        results += associate_vehicle_to_lanes(vehicle_xy=vehicle_xy,
                                              vehicle_heading = vehicle_heading,
                                              map_data = map_data,
                                              max_distance = start_distance,
                                              max_heading_diff_rad = max_heading_diff_rad,
                                              eps=eps)
        
        start_distance = start_distance+2
        
    return results

def associate_vehicle_to_lanes(
    vehicle_xy,
    vehicle_heading,
    map_data,
    max_distance=2.0,
    max_heading_diff_rad= np.pi/4,
    eps=0.2
):
    """
    Parameters
    ----------
    vehicle_xy : tuple (x, y) in the same CRS/units as the lanes (meters).
    vehicle_heading_deg : float, vehicle yaw in degrees (0° = +x, CCW positive).
    lanes : iterable of shapely LineString (you can pass (id, line) tuples too).
    max_distance : max perpendicular distance (meters) from vehicle to lane.
    max_heading_diff_deg : max absolute heading difference (degrees).
    eps : arclength step when estimating lane tangent.

    Returns
    -------
    list of dicts, one per matching lane:
      {
        "lane": <LineString or lane record>,
        "distance": <float meters>,
        "heading_diff_deg": <float degrees>,
        "closest_point": <shapely Point>,
        "s_on_lane": <float arclength position>
      }
    Sorted by distance, then by heading difference.
    """
    lanes = map_data['converted_lanes']
    x, y = vehicle_xy
    v_pt = Point(x, y)

    results = []
    for idx, lane in enumerate(lanes):
        # If the caller passed (id, LineString), keep the pair; otherwise the object itself.
        line = lane[1] if isinstance(lane, (list, tuple)) and isinstance(lane[1], LineString) else lane
        

        if not isinstance(line, LineString):
            continue  # skip invalid entries
        
        dist = v_pt.distance(line)
        if dist > max_distance:
            continue

        # Heading check
        s = line.project(v_pt)
        closest = line.interpolate(s)
        lane_heading = lane_heading_at_s(line, s, eps=eps)
        diff = angle_between_angles(vehicle_heading, lane_heading)

        if diff <= max_heading_diff_rad:
            results.append({
                "lane": lane,
                "lane_idx": idx,
                "distance": dist,
                "heading_diff_deg": diff,
                "closest_point": closest,
                "s_on_lane": s
            })

    # Sort best matches first
    results.sort(key=lambda r: (r["distance"], r["heading_diff_deg"]))
    
    
    if len(results) > 1: # check if candidates are successor and predecessor => take only the successor
        rel = np.array(map_data['lane_relationships'])
        cand_idxs = [r['lane_idx'] for r in results]

        # Mark any candidate that is a successor of another candidate
        successors = set()
        for i in range(len(cand_idxs)):
            a = cand_idxs[i]
            for j in range(len(cand_idxs)):
                if i == j:
                    continue
                b = cand_idxs[j]
                # If b is successor of a, prefer b
                if rel[a, b] == 2:
                    successors.add(b)
                # (The reverse check is redundant for collecting successors,
                #  but harmless if you like symmetry.)
                # if rel[b, a] == 2:
                #     successors.add(a)

        if successors:
            # Keep only successors; maintain previous sorting
            results = [r for r in results if r['lane_idx'] in successors]
    
    return results


def closest_adjacent_lanes_unlabeled(
    current_idx: int,
    map_data,                  
    candidates: list | None = None,    # optional (e.g., from find_neighbour)
    max_lateral: float = 6.0,          # meters; band to consider "adjacent"
    min_overlap: float = 0.25,         # fraction of samples that must be valid
    sign_consistency: float = 0.7,     # how consistently on the same side
    n_samples: int = 8,               # samples along reference lane
    eps: float = 0.2,                  # step for tangent estimation
    return_indices_only: bool = True   # if True, return [idx,...]; else return dicts
):
    """
    Returns a list with up to TWO entries: the closest adjacent lane on each side,
    sorted by absolute lateral distance (nearest first). No left/right labels.

    Each returned entry is either:
      • idx (int)                if return_indices_only=True, or
      • {'idx', 'lat_dist', 'overlap'}  if return_indices_only=False
    """
    lanes = map_data['converted_lanes']
    ref = lanes[current_idx]
    L = ref.length
    
    if L == 0.0:
        return []

    # Candidate pool (drop self)
    if candidates is None:
        cand_indices = [i for i in range(len(lanes)) if i != current_idx]
    else:
        cand_indices = sorted({i for i in candidates if i != current_idx})

    # Samples along the reference lane (avoid endpoints)
    ss = [(i + 1) * L / (n_samples + 1) for i in range(n_samples)]
    ref_pts   = [ref.interpolate(s) for s in ss]
    ref_tan   = [lane_heading_at_s(ref, s, eps, return_tangent=True) for s in ss]
    ref_normL = [(-ty, tx) for (tx, ty) in ref_tan]  # left normal

    # Evaluate candidates
    left_cands  = []  # (idx, med_abs_lat, overlap)
    right_cands = []
    for j in cand_indices:
        line = lanes[j]
        lats, signs = [], []
        for k, p in enumerate(ref_pts):
            nx, ny = ref_normL[k]
            sc = line.project(p)
            q  = line.interpolate(sc)
            dx, dy = (q.x - p.x), (q.y - p.y)
            lat = dx*nx + dy*ny  # + = left of ref, - = right
            if abs(lat) <= max_lateral:
                lats.append(lat)
                signs.append(1 if lat >= 0.0 else -1)

        if not lats:
            continue

        overlap = len(lats) / n_samples
        if overlap < min_overlap:
            continue

        # Require consistent side (avoid crossers/zigzags)
        sign_ratio = abs(sum(signs)) / len(signs)
        if sign_ratio < sign_consistency:
            continue
        med_lat = median(lats)

        # If median is exactly 0 (rare), break tie by majority sign
        side_left = (med_lat > 0) or (med_lat == 0 and sum(signs) > 0)
        entry = (j, abs(med_lat), overlap)
        if side_left:
            left_cands.append(entry)
        else:
            right_cands.append(entry)

    # Pick nearest per side
    picks = []
    if left_cands:
        picks.append(min(left_cands, key=lambda t: t[1]))
    if right_cands:
        picks.append(min(right_cands, key=lambda t: t[1]))

    # Sort by proximity and format output
    picks.sort(key=lambda t: t[1])
    if return_indices_only:
        return [idx for (idx, _, _) in picks]
    else:
        return [{'idx': idx, 'lat_dist': d, 'overlap': ov} for (idx, d, ov) in picks]

############################ utils functions ########################################
def convert_map(map_data, update_adjacent=True):
    '''
    Add lanes as line strings 
    '''
    converted_lane_pts = [LineString(lane_pts.cpu()) for lane_pts in map_data['center_pts'].contiguous()]
    
    new_map_data = deepcopy(map_data)
    new_map_data['converted_lanes'] = converted_lane_pts
    
    if update_adjacent:
        new_map_data = refresh_adjacent_annotations(new_map_data)
    
    return new_map_data


def lane_heading_at_s(line: LineString, 
                      s: float, 
                      eps: float = 0.2,
                      return_tangent = False):
    """
    Estimate the lane's local heading (deg) at the point on the line closest to p.
    Uses a forward difference along the line; falls back to backward if near the end.
    `eps` is the step in the line's arclength units (same as the line coordinates).
    """
    #s = line.project(p)  # arclength position (meters if your coords are meters)
    L = line.length
    if L == 0.0:
        return 0.0

    s1 = min(L, max(0.0, s))
    # Try forward step; if at the end, step backward.
    if s1 + eps <= L:
        pa = line.interpolate(s1)
        pb = line.interpolate(s1 + eps)
    elif s1 - eps >= 0.0:
        pa = line.interpolate(s1 - eps)
        pb = line.interpolate(s1)
    else:
        # Degenerate short line: use tiny symmetric step if possible
        half = min(eps * 0.5, L * 0.5)
        pa = line.interpolate(max(0.0, s1 - half))
        pb = line.interpolate(min(L, s1 + half))

    dx = pb.x - pa.x
    dy = pb.y - pa.y
    
    if return_tangent:
        return dx, dy
    else:
        if dx == 0.0 and dy == 0.0:
            return 0.0
        return math.atan2(dy, dx)  # [-pi,pi] bearing, x-axis = 0°, increasing CCW


def lane_curvature_at_s(line: LineString, s: float, ds: float = 0.5, signed: bool = True) -> float:
    """
    Curvature κ(s) for a lane centerline (LineString) at arclength s.
    Uses a 3-point circumcircle approximation around s with step ±ds.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Lane centerline in metric coordinates.
    s : float
        Arclength position along `line` (meters).
    ds : float
        Half-window for sampling around `s` (meters). Typical: 0.3–1.0 m.
    signed : bool
        If True, left turns (CCW) are positive; otherwise return |κ|.

    Returns
    -------
    float
        Curvature in 1/m. Zero if geometry is too short/degenerate.
    """
    L = line.length
    if L <= 0.0:
        return 0.0

    # Clamp s to [0, L]
    s = max(0.0, min(L, s))

    # Use as much window as available near ends
    h1 = min(ds, s)
    h2 = min(ds, L - s)

    # If one side has almost no room, try to expand symmetrically if possible
    if h1 < 1e-6 or h2 < 1e-6:
        # fall back to the largest symmetric window that fits
        h = min(ds, s, L - s)
        h1, h2 = h, h
        if h <= 1e-6:
            return 0.0  # can't form a triangle

    p0 = line.interpolate(s - h1)
    p1 = line.interpolate(s)
    p2 = line.interpolate(s + h2)

    # Triangle side lengths
    ax, ay = p1.x - p0.x, p1.y - p0.y
    bx, by = p2.x - p1.x, p2.y - p1.y
    cx, cy = p2.x - p0.x, p2.y - p0.y

    a = math.hypot(ax, ay)
    b = math.hypot(bx, by)
    c = math.hypot(cx, cy)

    if a <= 1e-9 or b <= 1e-9 or c <= 1e-9:
        return 0.0

    # Twice the signed triangle area using cross((p1-p0), (p2-p0))
    cross = ax * cy - ay * cx
    # κ = 4*Area / (a*b*c) = 2*|cross| / (a*b*c)
    k = (2.0 * cross) / (a * b * c)

    return k if signed else abs(k)
    

def angle_between_vectors(v1, v2):
    return np.arccos(v1@v2.T / (np.linalg.norm(v1) * (np.linalg.norm(v2))))

def angle_between_angles(h1, h2):
    s_h1, c_h1 = np.sin(h1), np.cos(h1)
    s_h2, c_h2 = np.sin(h2), np.cos(h2)
    return np.arccos(np.clip(c_h1*c_h2 + s_h1*s_h2, -1., 1.))


def find_next(start_idx, map_data):
    return np.where(np.array(map_data['lane_relationships'])[start_idx] == 2)[0]

def find_left(start_idx, map_data, get_closest = True):
    if get_closest:
        return closest_adjacent_lanes_unlabeled(current_idx=start_idx,
                                                map_data=map_data,
                                                candidates = np.where(np.array(map_data['lane_relationships'])[start_idx] == 3)[0])
    else:
        return np.where(np.array(map_data['lane_relationships'])[start_idx] == 3)[0]

def find_right(start_idx, map_data, get_closest = True):
    if get_closest:
        return closest_adjacent_lanes_unlabeled(current_idx=start_idx,
                                                map_data=map_data,
                                                candidates = np.where(np.array(map_data['lane_relationships'])[start_idx] == 4)[0])
    else:
        return np.where(np.array(map_data['lane_relationships'])[start_idx] == 4)[0]

def find_neighbour(start_idx, map_data, get_closest = True):
    if get_closest:
        return closest_adjacent_lanes_unlabeled(current_idx=start_idx,
                                                map_data=map_data,
                                                candidates = np.where(np.array(map_data['lane_relationships'])[start_idx] >= 3)[0])
    else:
        return np.where(np.array(map_data['lane_relationships'])[start_idx] >= 3)[0]
    
    
    
# --- small helper: unit left-normal at arclength s ----------------------------
def _unit_left_normal_at_s(line: LineString, s: float, eps: float = 0.2):
    # reuse your lane_heading_at_s to get the tangent vector
    tx, ty = lane_heading_at_s(line, s, eps, return_tangent=True)
    n = math.hypot(tx, ty)
    if n == 0.0:
        return (0.0, 1.0)
    return (-ty / n, tx / n)

def _closest_adjacent_one_side(
    current_idx: int,
    map_data,
    side: str,                        # 'left' or 'right'
    candidates: list | None = None,   # optional prefilter of indices
    max_lateral: float = 6.0,
    min_overlap: float = 0.25,
    sign_consistency: float = 0.7,
    n_samples: int = 8,
    eps: float = 0.2
) -> int | None:
    """
    Geometry-only picker that returns the closest adjacent lane on the given side.
    Side is determined by signed lateral distance using the reference lane's left normal.
    """
    assert side in ("left", "right")
    lanes = map_data['converted_lanes']
    ref = lanes[current_idx]
    L = ref.length
    if L == 0.0:
        return None

    # Candidate pool
    if candidates is None:
        cand_indices = [k for k in range(len(lanes)) if k != current_idx]
    else:
        cand_indices = sorted({int(k) for k in candidates if int(k) != current_idx})

    # Sample along the reference lane (avoid exact endpoints)
    ss = [(i + 1) * L / (n_samples + 1) for i in range(n_samples)]
    ref_pts = [ref.interpolate(s) for s in ss]
    # Left normals from your existing tangent util
    ref_tan = [lane_heading_at_s(ref, s, eps=eps, return_tangent=True) for s in ss]
    ref_normL = [(-ty, tx) for (tx, ty) in ref_tan]

    best_idx, best_abs_med_lat = None, float('inf')

    for j in cand_indices:
        line = lanes[j]
        lats, signs = [], []
        for k, p in enumerate(ref_pts):
            nx, ny = ref_normL[k]
            sc = line.project(p)
            q  = line.interpolate(sc)
            lat = (q.x - p.x) * nx + (q.y - p.y) * ny   # + => left of ref; - => right
            if abs(lat) <= max_lateral:
                lats.append(lat)
                signs.append(1 if lat >= 0.0 else -1)

        if not lats:
            continue

        overlap = len(lats) / n_samples
        if overlap < min_overlap:
            continue

        # require consistent side
        sign_ratio = abs(sum(signs)) / len(signs)
        if sign_ratio < sign_consistency:
            continue

        med_lat = median(lats)
        want_left = (side == "left")
        # If median is exactly 0 (rare), treat as right by convention
        is_left = (med_lat > 0.0)
        if (want_left and not is_left) or ((not want_left) and is_left and med_lat != 0.0):
            continue

        abs_med = abs(med_lat)
        if abs_med < best_abs_med_lat:
            best_abs_med_lat = abs_med
            best_idx = j

    return best_idx



def _indices_between(i: int, j: int) -> list[int]:
    """Inclusive integer band between i and j (order-agnostic)."""
    lo, hi = (i, j) if i <= j else (j, i)
    return list(range(lo, hi + 1))

# --- main updater: refresh adjacency in the M×M matrix ------------------------
def refresh_adjacent_annotations(
    map_data,
    strategy: str = "augment",    # "augment" (default) or "replace"
    max_lateral: float = 6.0,
    min_overlap: float = 0.25,
    sign_consistency: float = 0.7,
    n_samples: int = 8,
    eps: float = 0.2
):
    """
    Fix left/right adjacency in map_data['lane_relationships'] using geometry,
    but ONLY for rows (lanes) that already have at least one left or right neighbor.

    Parameters
    ----------
    strategy : "augment" | "replace"
        - "augment": add the closest missing neighbor on each side; keep existing.
        - "replace": clear existing left/right entries and keep only the closest.

    Notes
    -----
    - Successor (code==2) entries are untouched.
    - We DO NOT enforce symmetry (to respect your constraint about only
      modifying lanes that already have left/right). If you want symmetry,
      you can run a pass after this to mirror updates.
    """
    # ensure converted lanes exist
    if 'converted_lanes' not in map_data:
        map_data = convert_map(map_data)

    rel = np.array(map_data['lane_relationships'], copy=True)
    M = rel.shape[0]
    assert rel.shape[0] == rel.shape[1], "lane_relationships must be MxM"

    for i in range(M):
        row = rel[i]
        has_left  = np.any(row == 3)
        has_right = np.any(row == 4)
        if not (has_left or has_right):
            continue  # only process lanes that already have left or right
        
        ### update left ###
        left_ann = np.where(row == 3)[0]
        candL = sorted({k for j in left_ann for k in _indices_between(i, int(j)) if k != i})
        li = _closest_adjacent_one_side(
            current_idx=i,
            map_data=map_data,
            side="left",
            candidates=candL,
            max_lateral=max_lateral,
            min_overlap=min_overlap,
            sign_consistency=sign_consistency,
            n_samples=n_samples,
            eps=eps
        )
        
        if strategy == "replace":
            rel[i, row == 3] = 0
        if li is not None:
            rel[i, li] = 3

        right_ann = np.where(row == 4)[0]
        candR = sorted({k for j in right_ann for k in _indices_between(i, int(j)) if k != i})
        ri = _closest_adjacent_one_side(
            current_idx=i,
            map_data=map_data,
            side="right",
            candidates=candR,
            max_lateral=max_lateral,
            min_overlap=min_overlap,
            sign_consistency=sign_consistency,
            n_samples=n_samples,
            eps=eps
        )

        if strategy == "replace":
            rel[i, row == 4] = 0
        if ri is not None:
            rel[i, ri] = 4

    # write back
    map_data['lane_relationships'] = rel
    return map_data