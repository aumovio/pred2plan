"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import casadi as ca
import numpy as np
import time

from functools import partial
import traceback

from . import DrivingCorridorSelector
from . import  AdaptiveDecisionPostponing as adp
from . import Collisionrisk as cr

global dcm, F_iou, F_intersect, dc_helper
def init_scenario_selector(path,w_l, w_r, num_hard_obs, params, obs_num=3, modes=6):
    N, dt,M,s_final,v_max,v_min, a_max, a_min= params["N"],params["dt"],params["M"],params["s_final"],params["v_max"],0, params["a_max"], params["a_min"]
    global dcm
    dcm = []
    for i in range(modes):
        dcm.append(DrivingCorridorSelector.DrivingCorridorExtractor(N=N, dt=dt, v_max=v_max, v_min=v_min, a_max=a_max, a_min=a_min, obs_num=3,  path_ref=path, w_l=w_l, w_r=-w_r, num_hard_obs=num_hard_obs, M=M, s_final=s_final))

    global F_iou, F_intersect
    F_iou, F_intersect = init_casadi_functions(N)

    global f_backwardsprop_lin_eq
    f_backwardsprop_lin_eq = adp.init_backwardsprop_lin_eq_ca(params) 

    global dc_helper
    dc_helper = DrivingCorridorSelector.DrivingCorridor(lane_id=0, mode_id=0, N=N, dt=dt, v_max=v_max, v_min=0, a_max=a_max, a_min=a_min,f=0)   #self.dc.append() #0i%2
    dc_helper.init_casadi_function(num_hard_obs)

    global f_risk
    f_risk =cr.init_collision_risk(params) 


def init_casadi_functions(N=40):

    #iou
    dc1= ca.MX.sym('dc1',4)
    dc2= ca.MX.sym('dc2',4)
    # Unpack the coordinates
    x_min1, x_max1, y_min1, y_max1 = dc1[0], dc1[1], dc1[2], dc1[3]
    x_min2, x_max2, y_min2, y_max2 = dc2[0], dc2[1], dc2[2], dc2[3]
    
    # Calculate the intersection coordinates
    x_min_inter = ca.fmax(x_min1, x_min2)
    y_min_inter = ca.fmax(y_min1, y_min2)
    x_max_inter = ca.fmin(x_max1, x_max2)
    y_max_inter = ca.fmin(y_max1, y_max2)
    
    # Calculate the width and height of the intersection
    inter_width = ca.fmax(0, x_max_inter - x_min_inter)
    inter_height = ca.fmax(0, y_max_inter - y_min_inter)
    
    # Calculate the area of the intersection
    inter_area = inter_width * inter_height
    
    # Calculate the area of both boxes
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    # Calculate the union area
    union_area = area1 + area2 - inter_area
    
    # Calculate IoU
    iou = ca.if_else(union_area > 0 , inter_area / union_area, 0) #inter_area / union_area if union_area > 0 else 0
    f_iou= ca.Function('f_iou', [dc1, dc2],[iou]) #,
    F_iou=f_iou.map(N)


    #intersection
    
    dc1= ca.MX.sym('dc1',4)
    dc2= ca.MX.sym('dc2',4)
    #intersection = ca.MX.sym('intersect',4)
    # Unpack the coordinates
    x_min1, x_max1, y_min1, y_max1 = dc1[0], dc1[1], dc1[2], dc1[3]
    x_min2, x_max2, y_min2, y_max2 = dc2[0], dc2[1], dc2[2], dc2[3]
    
    # Calculate the intersection coordinates
    x_min_inter = ca.fmax(x_min1, x_min2)
    y_min_inter = ca.fmax(y_min1, y_min2)
    x_max_inter = ca.fmin(x_max1, x_max2)
    y_max_inter = ca.fmin(y_max1, y_max2)
    
    
    intersection = ca.vertcat(x_min_inter, x_max_inter,y_min_inter, y_max_inter)
    f_intersect= ca.Function('f_interect', [dc1, dc2],[intersection]) #,
    F_intersect=f_intersect.map(N)

    return F_iou, F_intersect

def cluster_scenarios(params, dcs,groups=[], modes=[], ms=np.arange(6), N=40, trhld=0.92**40):
    for i, m in enumerate(ms): #range(len(dcs)): 

        found= False
        for k,group in enumerate(groups):
           ioup = np.prod(F_iou(group.T, dcs[i].T))
           if  ioup >  trhld: #or (len(modes) >= params["S"] and ioup > trhld/4): 
               groups[k]= F_intersect(group.T, dcs[i].T).T
               #print(k,i, modes)
               modes[k].append(m)
               found = True
               break
        if not found:
            new_group = dcs[i]
            groups.append(new_group)
            modes.append([m]) #i

    rm_k=0
    for k,group in enumerate(groups[::-1]): #temporary
        if len(modes) <= params["S"]:
            break
        kk = -k-1 + rm_k
        if -kk>len(modes):
            break
        if len(modes[kk]) <= 2: #
            for j,group1 in enumerate(groups[kk-1::-1]):
                jj= -j-1+kk
                ioup = np.prod(F_iou(group.T, group1.T))
                #print(kk,jj,ioup)
                if  ioup > trhld/10:
                   groups[jj]= F_intersect(group.T, group1.T).T
                   #print(k,i, modes)
                   modes[jj].append(modes[kk][0])
                   del groups[kk]
                   del modes[kk]
                   rm_k+=1
                   break
                         
    return groups, modes
        
def get_dc_constraints(group,lane_id):
    #print("lane_id",lane_id)
    e_c_min = group[:,2]
    e_c_max = group[:,3]
    s_ref_min = dcm[0].correct_arclength_2_ref_lane(group[:,0], lane_id)  
    s_ref_max = dcm[0].correct_arclength_2_ref_lane(group[:,1], lane_id)  
    x_min = dcm[0].lut_x(s_ref_min)
    y_min = dcm[0].lut_y(s_ref_min)
    psi_min = dcm[0].lut_phi(s_ref_min)
    x_max = dcm[0].lut_x(s_ref_max)
    y_max = dcm[0].lut_y(s_ref_max)
    psi_max = dcm[0].lut_phi(s_ref_max)
    #min: mx x +my y +c > 0
    mx_min = np.sin(psi_min +np.pi/2)
    my_min = -np.cos(psi_min +np.pi/2)
    c_min = -np.sin(psi_min +np.pi/2)*x_min + np.cos(psi_min +np.pi/2)*y_min 
    #max: mx x +my y +c < 0
    mx_max = np.sin(psi_max +np.pi/2)
    my_max = -np.cos(psi_max +np.pi/2)
    c_max = -np.sin(psi_max +np.pi/2)*x_max + np.cos(psi_max +np.pi/2)*y_max
    return e_c_min,e_c_max, mx_min, my_min, c_min, mx_max, my_max, c_max

def extract_dc_m( m, z0 ,ObstacleArray, v_max):
    # Perform a simple computation: return (index^2) + z0
    return dcm[m].extract_dc_m(z0,m, ObstacleArray,v_max)

def sum_elements_at_indices(arr, indices):
    return sum(arr[index] for index in indices if 0 <= index < len(arr))

def add_lat_dc(dc, k_lc1, k_lc2, lane_id, current_lane_id,lane_width=3.5 ):
        # for each dc add lateral dc
        if k_lc1!=-1: # or k_lc2+1<dc.shape[0]:
            dc[:k_lc1,2] = current_lane_id*lane_width-lane_width/2
            dc[:k_lc1,3] = current_lane_id*lane_width+lane_width/2
            dc[k_lc1:k_lc2,2] = min(current_lane_id*lane_width-lane_width/2, lane_id*lane_width-lane_width/2)
            dc[k_lc1:k_lc2,3] = max(current_lane_id*lane_width+lane_width/2, lane_id*lane_width+lane_width/2)
            if k_lc2+1<dc.shape[0]:
                dc[k_lc2:,2] = lane_id*lane_width-lane_width/2
                dc[k_lc2: ,3] = lane_id*lane_width+lane_width/2
        else:
            dc[:,2] = current_lane_id*lane_width-lane_width/2
            dc[:,3] = current_lane_id*lane_width+lane_width/2
    
        return dc

    
def scenario_selection(z0, ObstacleArray, params, x0_, v_max, pool=None): 
    t_ = time.time()
    S = params["S"]
    m=len(dcm)
    N = dcm[0].N
    try:
        # Use partial to fix z0, ObstacleArray, and dcm arguments
        extract_dc_m_partial = partial(extract_dc_m, z0=z0, ObstacleArray=ObstacleArray, v_max=v_max)
        
        if pool is not None:
            dcs = pool.map(extract_dc_m_partial, np.arange(m))
        else:
            dcs = [extract_dc_m_partial(i) for i in range(m)]
        reach_v = dcs[0][0][:,2:4]
        # add lateral corridor
        dcs1 = [np.zeros((N, 4)) for _ in range(m)]
        for i in range(m):
            dcs1[i][:,:4]= add_lat_dc(dcs[i][0], dcs[i][2], dcs[i][3], dcs[i][4], dcs[i][5])
        groups, modes = cluster_scenarios(params,dcs1, [], [])#

        # backwards reachible set
        dc_frs = ca.DM.zeros((N,4))
        brset1,brset2 = -ca.inf*ca.DM.ones((N,6)), ca.inf*ca.DM.ones((N,6)) #np.array([[-ca.inf,-ca.inf,0,0,0,0]]), np.array([[ca.inf,ca.inf,0,0,0,0]])
        cost1, cost2 = 0,0
        idx_lower = 0
        idx_upper = 0
        for i, group in enumerate(groups):
            dc_frs[:,:2] = group[:,:2]
            dc_frs[:,2:4] =reach_v
            brset = dc_helper.f_brs(dc_frs)
            a1minreq = 2*(brset[1,0]-brset[0,0])/(params["dt"]**2) - 2*z0[3]/params["dt"] #maybe brset0-z0
            a1maxreq = 2*(brset[1,1]-brset[0,1])/(params["dt"]**2) - 2*z0[3]/params["dt"] 

            if brset[-1,1] > brset1[-1,1]:
                if brset[-1,1] -5 < brset1[-1,1]:
                    cost1 = dc_helper.f_cost(brset[:,:4])
                    cost2 = dc_helper.f_cost(brset1[:,:4])
                    if cost1 < cost2:
                        continue
                brset1[:,:4] = brset
                brset1[:,4:] = group[:,2:]
                idx_upper=i
            if brset[-1,1] < brset2[-1,1]:
                if brset[-1,1] +5 > brset2[-1,1]:
                    cost1 = dc_helper.f_cost(brset[:,:4])
                    cost2 = dc_helper.f_cost(brset2[:,:4])
                    if cost1 < cost2:
                        continue
                brset2[:,:4] = brset
                brset2[:,4:] = group[:,2:]
                idx_lower=i

        #check max decision postponing
        if x0_ is None:
            N_tree=N+1+(N-1)*(S-1)
            x0_ =  np.tile(z0.T,(N_tree,1)) 
        t_b = adp.get_decision_postponing(modes, ObstacleArray,params,modes[idx_lower][0], x0_, f_risk)  # , relevant_m, x_full, f_riskmuss lösung for most relevant objects finden
        t_b = min(t_b,20)
        max_b = adp.get_max_branching_k(brset1,brset2, params, f_backwardsprop_lin_eq)

        if t_b > max_b : #or 
            dcs_ = [dcs[i] for m in modes[idx_upper]] #[dcs[i] for m in modes[0]]
            dcs1 = [np.zeros((N, 4)) for _ in range(len(modes[idx_upper]))] #range(len(modes[0]))]
            for i in range(len(modes[idx_upper])): #range(len(modes[0])):
                dcs1[i][:,:4]= add_lat_dc(dcs_[i][7], dcs_[i][9], dcs_[i][10], dcs_[i][11], dcs_[i][5])
            len_groups = len(groups)
            groups, modes = cluster_scenarios(params, dcs1,groups[:idx_upper]+groups[idx_upper+1:], modes[:idx_upper]+modes[idx_upper+1:], modes[idx_upper])
            if len(groups)> len_groups-1:
                dc_frs[:,:2] = groups[-1][:,:2] 
                brset1[:,:4] = dc_helper.f_brs(dc_frs)
                brset1[:,4:] = groups[-1][:,2:]
                max_b = adp.get_max_branching_k(brset1,brset2, params, f_backwardsprop_lin_eq)
                if t_b > max_b:
                    t_b = max_b

        # get constraints
        p_m =np.full(len(modes), 1/len(modes))
        o=0
        for j in range(len(modes)):
            p_m[j] = sum_elements_at_indices(ObstacleArray[o].probs,modes[j])   
        p_m = np.array(p_m).flatten()  
        i_sorted = np.argsort(p_m)[::-1]
    
    
        current_lane_id = dcm[0].get_current_lane_id(z0)

        dc_cons = ca.DM.zeros((8,(N+1)*S))
        dc_cons[0,0::N+1] = float(groups[0][0,2])
        dc_cons[1,0::N+1] = float(groups[0][0,3])
        for  i in range(S):
            if len(groups)>i:
                group = groups[i_sorted[i]]
            else:
                idx=min(i,len(modes)-1)
                group = groups[i_sorted[idx]]
            e_c_min,e_c_max, mx_min, my_min, c_min, mx_max, my_max, c_max = get_dc_constraints(group,current_lane_id)
            dc_cons[0,(N+1)*i+1:(N+1)*(i+1)] = e_c_min
            dc_cons[1,(N+1)*i+1:(N+1)*(i+1)] = e_c_max
            if z0[-1]>0.5: #error handling behind path
                dc_cons[2,(N+1)*i+1:(N+1)*(i+1)] = mx_min
                dc_cons[3,(N+1)*i+1:(N+1)*(i+1)] = my_min
                dc_cons[4,(N+1)*i+1:(N+1)*(i+1)] = c_min
            dc_cons[5,(N+1)*i+1:(N+1)*(i+1)] = mx_max
            dc_cons[6,(N+1)*i+1:(N+1)*(i+1)] = my_max
            dc_cons[7,(N+1)*i+1:(N+1)*(i+1)] = c_max
        dc_cons = dc_cons.T

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        t_b = 10
        modes= [[0], [1], [2], [3], [4], [5]]
        p_m = np.full(len(modes), 1/len(modes))
        i_sorted = np.arange(len(modes))
        dc_cons = None
        p_b =np.full(S, 1/S)

    """Put selected Scenarios in obstacleArraySCMPCC"""    
    p_b =np.full(S, 1/S)
    for o, obs in enumerate(ObstacleArray):
        obs_x_predk_m =[]
        obs_y_predk_m =[]
        obs_psi_predk_m =[]
        obs_v_predk_m =[]
        obs_stdx_predk_m =[]
        obs_stdy_predk_m =[]
        obs.S =S
        for j in range(S):
            if len(modes)>=S:
                s = modes[i_sorted[j]][0] #modes[j][0]
                #if o==0:
                    #print("selected modes",modes[i_sorted[j]])
                p_b[j] = p_m[i_sorted[j]] #sum_elements_at_indices(ObstacleArray[o].probs,modes[j])   
            else:
                idx=min(j,len(modes)-1)
                s = i_sorted[idx] #j
                p_b[j] =  p_m[i_sorted[idx]]#sum_elements_at_indices(ObstacleArray[o].probs, [j])  
            if len(ObstacleArray[o].x_preds)< s+1:
                    s = len(ObstacleArray[o].x_preds)-1
                    #print(">Warning: Less predictions then scenarios ")
            obs_x_predk_m.append(ObstacleArray[o].x_preds[s])
            obs_y_predk_m.append(ObstacleArray[o].y_preds[s])
            obs_psi_predk_m.append(ObstacleArray[o].psi_preds[s])
            obs_v_predk_m.append(ObstacleArray[o].v_preds[s])
            obs_stdx_predk_m.append(ObstacleArray[o].stdx_preds[s])
            obs_stdy_predk_m.append(ObstacleArray[o].stdy_preds[s])
        obs.get_prediction(obs_x_predk_m, obs_y_predk_m, obs_psi_predk_m, N, obs_v_predk_m, obs_stdx_predk_m, obs_stdy_predk_m, p_b/np.sum(p_b)) 
    return  modes, dc_cons, ObstacleArray, t_b #




 