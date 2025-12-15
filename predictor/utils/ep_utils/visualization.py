import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from scipy.stats.distributions import chi2
from scipy.special import softmax
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import unbatch
from typing import Optional, Dict, Union
from utils.ep_utils import Spline, basis_function_b
from tqdm.auto import tqdm
import seaborn as sns
from matplotlib.collections import LineCollection
import torch.nn.functional as F
from copy import deepcopy

def track_type_color_gradient(track_type, length, a_range=(0.2, 0.6)):
    if track_type == 3: # focal agent
        return plt.cm.Reds(np.linspace(a_range[0], a_range[1], length)), 2 # red
    elif track_type == 2: # scored agent
        return plt.cm.Blues(np.linspace(a_range[0], a_range[1], length)), 1 # orange
    else:
        return plt.cm.Greys(np.linspace(a_range[0], a_range[1], length)), 0 # grey

def ego_color_gradient(length, a_range=(0.2, 0.6)):
    return plt.cm.Greens(np.linspace(a_range[0], a_range[1], length)) 


def prediction_mode_color(length, k, K=6):
    K_range = np.linspace(0., 1., K+1)
    return plt.cm.viridis(np.linspace(K_range[k], K_range[k+1], length)) 

# def obj_type_color(obj_type):
#     if track_type == 3: # focal agent
#         return sns.color_palette('colorblind')[3] # red
#     elif track_type == 2: # scored agent
#         return sns.color_palette('colorblind')[1] # orange
#     else:
#         return sns.color_palette('colorblind')[7] # grey

def draw_confidence_ellipse(mean, cov, ax, **kwarg):
    '''
    Draw the covariance ellipse with 2-Sigma confidence.
    mean: mean of multivariate Normal Distribution
    cov: covariance of multivariate Normal Distribution
    '''
    
    d,u = np.linalg.eigh(cov)
    
    confidence = chi2.ppf(0.95,2)
    height, width = 2*np.sqrt(confidence*d)
    angle = np.degrees(np.arctan2(u[1,-1], u[0,-1]))
    
    #ax.scatter(*mean, s=100, marker='x', **kwarg)
    
    ellipse = Ellipse(
        xy=mean, 
        width=width, 
        height=height, 
        angle = angle,
        fill = False,
        **kwarg
    )
    
    ax.add_artist(ellipse)

def visualise_scenario(data: Union[Batch, HeteroData],
                       prediction = None,
                       in_global_frame = False,
                       draw_raw_data = False):
    if isinstance(data, HeteroData):
        B = data.batch_size
    else:
        B = 1
    
    data = data.cpu()
    if prediction is not None:
        dummy_pred_dict_list = [dict() for _ in range(B)]
        for k, v in prediction.items():
            v_list = unbatch(v, batch=data['agent']['batch'])
            for dummy_pred_dict, unbatched_v in zip(dummy_pred_dict_list, v_list):
                dummy_pred_dict[k]=unbatched_v.cpu().detach()

    fig, axs = plt.subplots(B, 1, figsize=(8, 9*B))
    
    # for d in dummy_pred_dict_list:
    #     for k, v in d.items():
    #         print(k, v.shape)
    
    for batch_idx in tqdm(range(B)):
        visualise_one_scenario(ax = axs[batch_idx],
                               data = data.get_example(batch_idx),
                               prediction = dummy_pred_dict_list[batch_idx] if prediction else None,
                               in_global_frame = in_global_frame,
                               draw_raw_data = draw_raw_data)
        
def visualise_one_scenario(ax,
                           data: HeteroData,
                           prediction = None,
                           in_global_frame = False,
                           draw_raw_data = False,
                           draw_gt = True,
                           radius = 50 ):

    path_deg = data['map']['map_lane_boundary_cps'].shape[-2]-1
    hist_deg = data['agent']['cps_mean'].shape[-2]-1
    
    origin, R = data['origin'], data['ro_mat']
    
        
    draw_map(ax=ax, data=data, in_global_frame=in_global_frame)
    draw_track(ax=ax, data=data, in_global_frame=in_global_frame, draw_raw_data=draw_raw_data, draw_gt = draw_gt)
    
    
    if prediction is not None:
        draw_prediction(ax=ax, data=data, prediction=prediction, in_global_frame=in_global_frame)
        
    if in_global_frame:
        ax.set_xlim([origin[0]-radius, origin[0]+radius])
        ax.set_ylim([origin[1]-radius, origin[1]+radius])
    else:
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
    ax.set_aspect(1)
    ax.set_xlabel('Scenario: ' + data['scenario_id'], fontsize=10)
                           
        
def draw_map(ax,
             data: HeteroData,
             in_global_frame = False):
    
    path_deg = data['map']['map_lane_boundary_cps'].shape[-2]-1
    M = data['map']['num_nodes']
    origin, R = data['origin'], data['ro_mat']  
                           
    map_cps = data['map']['map_lane_boundary_cps'].view(M, 3, -1)
    tau = np.linspace(0., 1., 20)
    map_points = (map_cps@ basis_function_b(tau, n=path_deg, delta_t=1, return_kron=True).T).view(M, 3, -1, 2)
    
    if in_global_frame:
        map_points = map_points@R.T + origin 
        
    # Plot Map
    for lane_type, lane_points in zip(data['map']['map_lane_type'], map_points):
        #ax.plot(*lane_points[-1].T, c='green', zorder=-1)
        ax.plot(*lane_points[0].T, c='grey', zorder=-1, linewidth=0.5)
        ax.plot(*lane_points[1].T, c='grey', zorder=-1, linewidth=0.5)
        
        
def draw_track(ax,
               data: HeteroData,
               in_global_frame = False,
               draw_raw_data = False,
               draw_gt = True):
    A, T_hist, T_fut = data['agent']['num_nodes'], data['agent']['x'].shape[1], data['agent']['y'].shape[1]
    origin, R = data['origin'], data['ro_mat']
    
    if draw_raw_data:
        x = data['agent']['x'][:, :, :2]
        
    else:
        hist_deg = data['agent']['cps_mean'].shape[-2]-1
        tau = np.linspace(0., 1., T)
        x_cps = data['agent']['cps_mean'].view(A, -1)
        x = (x_cps@ basis_function_b(tau, n=hist_deg, delta_t=4.9, return_kron=True).T).view(A, T_hist, -1)
        
    if in_global_frame:
        x = x@R.T + origin 
        if data['agent']['y'] is not None:
            y = data['agent']['y'][:, :, :2]@R.T + origin 
        
    # Plot Track 
    for obj_idx, (obj_type, track_type, x_mask, y_mask, hist_points, future_points) in enumerate(zip(data['agent']['object_type'], data['agent']['track_category'], data['agent']['timestep_x_mask'], data['agent']['timestep_y_mask'], x, y)):
        c_hist, zorder = track_type_color_gradient(track_type, T_hist, (0.2, 0.4))
        c_fut, _ = track_type_color_gradient(track_type, T_fut, (0.4, 0.6))
        if obj_idx == data['agent']['av_index']:
            c_hist = ego_color_gradient(T_hist, (0.2, 0.4))
            c_fut = ego_color_gradient(T_fut, (0.4, 0.6))
        
        if draw_raw_data:
            ax.scatter(*hist_points[x_mask].T, c=c_hist[x_mask], marker='.', zorder=zorder)
            ax.scatter(*hist_points[x_mask][[-1]].T, c=c_hist[x_mask][[-1]], marker='o', zorder=5, edgecolor='black')
        else:
            ax.scatter(*hist_points.T, c = c_hist, marker='.', zorder=zorder)
            ax.scatter(*hist_points[[-1]].T, c=c_hist[[-1]], marker='o', zorder=5, edgecolor='black')
        
        if data['agent']['y'] is not None and draw_gt:
            ax.plot(*future_points[y_mask].T, '--', c='black', linewidth=2, zorder=10) 
            ax.scatter(*future_points[y_mask].T, c=c_fut[y_mask], marker='.', zorder=zorder)



def draw_prediction(ax,
                    data: HeteroData,
                    prediction,
                    in_global_frame = False,
                    target = ['focal', 'scored']):
    pred_traj_local = prediction['loc_traj'] # [A,K,60,2]
    prob = F.softmax(prediction['pi'], dim=-1) # [A,K]
    A, K, T_fut = pred_traj_local.shape[:3]

    global_origin, global_R = data['origin'], data['ro_mat']
    agent_origin, agent_R = data['agent']['agent_origin'], data['agent']['agent_R']

    pred_traj = pred_traj_local@agent_R.unsqueeze(1).mT + agent_origin.unsqueeze(1).unsqueeze(2)

    if in_global_frame:
        pred_traj = pred_traj@global_R.T + global_origin

    
    mask = data['agent']['timestep_x_mask'][:, -1]
    # if 'focal' in target:
    #     mask = mask | data['agent']['track_category'] == 3

    # if 'scored' in target:
    #     mask = mask | data['agent']['track_category'] == 2
    pred_traj = pred_traj[mask]
    #print(pred_traj.shape, mask.shape)
    prob = prob[mask]
    #print(pred_traj.shape)
    for k in range(K):
        if k == 5:
            c = prediction_mode_color(T_fut, k, K)
            #c = sns.color_palette('colorblind')[k]
            for a in range(pred_traj.shape[0]):
                c_a = deepcopy(c)
                #print(c.shape)
                c_a[:, -1] =  c[:, -1] * prob[a,k].numpy()
                #print(pred_traj[a,k].shape, c_a.shape)
                ax.scatter(*pred_traj[a,k].T, c = c_a, marker='.', zorder=5)

            #ax.plot(*pred_traj[a,k].T, c = c, alpha= float(prob[a,k].numpy())+ 0.3, zorder=5)