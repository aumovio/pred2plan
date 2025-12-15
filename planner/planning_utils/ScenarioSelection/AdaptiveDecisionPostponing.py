"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import casadi as ca
import numpy as np
import time
from . import Collisionrisk as cr
import scipy.linalg as linalg

   
"""Max Decision Postponing"""

def get_max_branching_k(brset1,brset2, params, f_backwardsprop_lin_eq):

    d1=brset1
    d2=brset2

    if d1[-1,1]>d2[-1,1]:
        du = d1
        dl = d2
    else:
        du = d1
        dl = d2

    k_i  = get_dc_intersect_k(du, dl, params["N"])


    # get m, c at k_i
    lin_con = backwardsprop_lin_eq(du, dl, k_i, params,f_backwardsprop_lin_eq)


    # check for each k_i-l whether intersection between all sets ( reachable_v, reachable_x,..)
    k_b = backwardsprop_get_intersect_k(du, dl, k_i, lin_con, params)-1

    # check lateral:
    k_b =  get_dc_intersect_k_bw(du[:,4:],dl[:,4:], k_b)
    return k_b

def get_dc_intersect_k(du, dl, N):

    for k in range(N):
        x_min1, x_max1 = du[k,0],du[k,1]
        x_min2, x_max2 = dl[k,0],dl[k,1]
        # Calculate the intersection bounds
        # print((x_min1, x_min2))
        # print((x_max1, x_max2))
        x_min_inter = ca.fmax(x_min1, x_min2)
        x_max_inter = ca.fmin(x_max1, x_max2)
    
        #print(x_min_inter, x_max_inter, v_min_inter,v_max_inter  )
        # Check if there is a valid intersection
        if (x_min_inter > x_max_inter):
            return k-1
    return N-2


def get_dc_intersect_k_bw(du, dl, N):

    for k in range(N, 0, -1):
        x_min1, x_max1 = du[k,0],du[k,1]
        x_min2, x_max2 = dl[k,0],dl[k,1]
        # Calculate the intersection bounds
        # print((x_min1, x_min2))
        # print((x_max1, x_max2))
        x_min_inter = ca.fmax(x_min1, x_min2)
        x_max_inter = ca.fmin(x_max1, x_max2)
    
        #print(x_min_inter, x_max_inter, v_min_inter,v_max_inter  )
        # Check if there is a valid intersection
        if (x_min_inter < x_max_inter):
            return k
    return 0
    
def compute_x_kl1( x_k1, u_max, l, params):
    dt = params["dt"]

    A = ca.DM([
        [1, dt],
        [0, 1]
    ])

    B = ca.DM([
            [0.5*dt**2],
            [dt]
            ])
    
    A_l1 = ca.vertcat(ca.horzcat(1, dt*(l+1)),ca.horzcat(0,1))
    # Compute the sum of A^i * B for i = 0 to l
    sum_Ai_B =(l + 1) *ca.DM([
    [0.5 * dt**2],
    [dt]
    ])
    sum_Ai_B[0] = (l + 1) *sum_Ai_B[0]

    # Compute x_{k-l}
    x_kl = ca.mtimes(ca.inv(A_l1), (x_k1 - ca.mtimes(u_max, sum_Ai_B)))
    # print(x_kl[1] )
    return x_kl



def backwardsprop_lin_eq(du, dl, k_i, params,f_backwardsprop_lin_eq):
    # N = 40
    dt = params["dt"]

    #get max and min
    v_u0 = np.clip(np.diff(du[k_i:,0].T)/dt,0,params["v_max"])
    v_l1 =np.clip(np.diff(dl[k_i:,1].T)/dt,0,params["v_max"])
    #print(v_u0, v_l1)
    k_max = np.argmax(v_u0) + k_i 
    k_min = np.argmin(v_l1) + k_i 

    k= k_max 
    l_u= k_max-k_i
    x_values_u =du[k,0:2]

    v_values_u =  ca.vertcat(ca.fmin(ca.fmax((du[k+1,0]-du[k,0])/dt,0),15.3), (du[k+1,0]-du[k,1])/dt)
    
    k= k_min 
    l_l= k_min-k_i
    x_values_l = dl[k,0:2]
    v_values_l = ca.vertcat((dl[k+1,1]-dl[k,0])/dt, ca.fmin(ca.fmax((dl[k+1,1]-dl[k,1])/dt,0),15.3))# vmink= (xmink+1-xmink)/dt-0.5*params["a_max"]*dt

    mk1_max, ck1_max, mk1_min, ck1_min, x_l1, x_l2,x_l3,x_l4 = f_backwardsprop_lin_eq( x_values_u, v_values_u,x_values_l, v_values_l, l_u, l_l)
    

    return  mk1_max, ck1_max, mk1_min, ck1_min, x_l1, x_l2,x_l3,x_l4


def init_backwardsprop_lin_eq_ca(params):

    x_values_u = ca.MX.sym('xu',2)
    v_values_u =  ca.MX.sym('vu',2)
    x_values_l = ca.MX.sym('xl',2)
    v_values_l = ca.MX.sym('vl',2)
    l_u = ca.MX.sym("l_u")
    l_l = ca.MX.sym("l_l")

    #v_values = [float(du[k+1,0]-d1[k,0])/dt, float(du[k+1,0]-du[k,1])/dt]
    x1_k1 = ca.vertcat(x_values_u[0],v_values_u[0])
    x_l1 = compute_x_kl1( x1_k1, params["a_max"], l_u, params)
    x2_k1 = ca.vertcat(x_values_u[1],v_values_u[1])
    x_l2 = compute_x_kl1( x2_k1, params["a_max"],l_u, params)
    # Given points
    x1, y1 = x_l1[0,0],x_l1[1,0]
    x2, y2 = x_l2[0,0],x_l2[1,0]

    #print("over",x1,y1,x2,y2)
    # Calculate the slope (m) and intercept (b) of the line
    mk1_max = (y2 - y1) / (x2 - x1)  # Slope formula
    ck1_max = y1 - mk1_max * x1             # Intercept formula

    #print("mk1_max,ck1_max",mk1_max,ck1_max)
    # below


    #v_values = [float(dl[k+1,1]-dl[k,0])/dt, float(dl[k+1,1]-dl[k,1])/dt]
    x3_k1 = ca.vertcat(x_values_l[0],v_values_l[0])
    x_l3 = compute_x_kl1( x3_k1, params["a_min"], l_l, params)
    x4_k1 = ca.vertcat(x_values_l[1],v_values_l[1])
    x_l4 = compute_x_kl1( x4_k1, params["a_min"], l_l, params)
    # Given points
    x1, y1 = x_l3[0,0],x_l3[1,0]
    x2, y2 = x_l4[0,0],x_l4[1,0]
    #print("lower",x1,y1,x2,y2)
    # Calculate the slope (m) and intercept (b) of the line
    mk1_min = (y2 - y1) / (x2 - x1)  # Slope formula
    ck1_min = y1 - mk1_min * x1             # Intercept formula
    #print("mk1_min,ck1_min",mk1_min,ck1_min)

    f_backwardsprop_lin_eq= ca.Function('f_dc', [x_values_u, v_values_u,x_values_l, v_values_l, l_u, l_l],[ mk1_max, ck1_max, mk1_min, ck1_min, x_l1, x_l2,x_l3,x_l4])
    return f_backwardsprop_lin_eq


def backwardsprop_get_intersect_k(du, dl, k_i, lin_con, params):
    N = params["N"]
    dt = params["dt"]
    mk1_max, ck1_max, mk1_min, ck1_min, x_l1, x_l2,x_l3,x_l4 = lin_con
    k_b = N
    for k in range(k_i-1,0,-1):
        #print(k)
        # upper
        x_values = [float(du[k,0]), float(du[k,1])]
        y_values = [float(du[k+1,0]-du[k,0])/dt, float(du[k+1,0]-du[k,1])/dt]
        
        x1, y1 = x_values[0], y_values[0]
        x2, y2 = x_values[1], y_values[1]
        
        # Calculate the slope (m) and intercept (b) of the line
        m1 = (y2 - y1) / (x2 - x1)  # Slope formula
        b1 = y1 - m1 * x1             # Intercept formula
        # lower
        x_values = [float(dl[k,0]), float(dl[k,1])]
        y_values = [float(dl[k+1,1]-dl[k,0])/dt, float(dl[k+1,1]-dl[k,1])/dt]
        
        x1, y1 = x_values[0], y_values[0]
        x2, y2 = x_values[1], y_values[1]
        
        # Calculate the slope (m) and intercept (b) of the line
        m2 = (y2 - y1) / (x2 - x1)  # Slope formula
        b2 = y1 - m2 * x1             # Intercept formula

        #check intersection
        x0, x1= max(du[k,0],dl[k,0]), min(du[k,1],dl[k,1])
        y00 = mk1_max*x0+ck1_max
        y01 = mk1_max*x1+ck1_max
        y10 = mk1_min*x0+ck1_min
        y11 = mk1_min*x1+ck1_min

        # print( y00, y01, y10, y11, dl[k,2],dl[k,3])
        # print((y00<dl[k,3] or y01<dl[k,3]) , (y00>dl[k,2] or y01>dl[k,2]) ,(y00<y10 or y01<y11))
        if  (y00<dl[k,3] or y01<dl[k,3]) and (y10>dl[k,2] or y11>dl[k,2])  and (y00<y10 or y01<y11):
                k_b=k
                break
        
        x_l1 = compute_x_kl1( x_l1 , params["a_max"], 0,params)
        x_l2 = compute_x_kl1( x_l2 , params["a_max"], 0,params)
        x1, y1 = x_l1[0,0],x_l1[1,0]
        x2, y2 = x_l2[0,0],x_l2[1,0]
        mk1_max = (y2 - y1) / (x2 - x1)  # Slope formula
        ck1_max = y1 - mk1_max* x1   

        x_l3 = compute_x_kl1( x_l3 , params["a_min"], 0,params)
        x_l4 = compute_x_kl1( x_l4 , params["a_min"], 0,params)
        x1, y1 = x_l3[0,0],x_l3[1,0]
        x2, y2 = x_l4[0,0],x_l4[1,0]
        mk1_min = (y2 - y1) / (x2 - x1)  # Slope formula
        ck1_min = y1 - mk1_min* x1  


    return k_b


"""Adaptive Decision Postponing"""

def get_decision_postponing(uvd, ObstacleArraySCMPCC,params, relevant_m, x_full, f_risk):

    #get collision risk
    Smax = 0
    smax = -ca.inf
    for s in range(params["S"]):
        if x_full[(params["N"])+(params["N"]-1)*(s),-1] >smax:
            smax = x_full[(params["N"])+(params["N"]-1)*(s),-1]
            Smax = s

    #print("sMAX",Smax)
    risk_matrix = []
    for obs in ObstacleArraySCMPCC:
        if Smax == 0:
           risk_matrix.append(cr.calculate_collision_risk( x_full, obs, relevant_m, params, f_risk, step=2))
        else:
            x0_ = np.concatenate((x_full[:2,:], x_full[(params["N"]-1)*Smax+2: ,:]))
            risk_matrix.append(cr.calculate_collision_risk( x0_, obs, relevant_m, params, f_risk, step=2))

    #print("risk",risk_matrix)
    dp=1
    if  params["S"]>1:
        if len(uvd)>=2:
            dp = calc_decision_postponing(uvd[0][0], uvd[1][0], ObstacleArraySCMPCC,params,risk_matrix)
            
        else:
            dp = calc_decision_postponing(uvd[0][0], uvd[0][1], ObstacleArraySCMPCC,params, risk_matrix)
    return min(dp,36)


def calc_decision_postponing( idx1, idx2, ObstacleArray,  params, risk_matrix =[] , step=2):
    
    rel_obs = get_relevant_obstacle(ObstacleArray,risk_matrix)
    #print("relobs",len(rel_obs))
    distinct = False
    for k in range(1, params["N"], step):
        if distinct:
            return k-step
        for obs in rel_obs:
            if  len(obs.x_preds)-1 < idx1 or len(obs.x_preds)-1 < idx2: 
                break
            x_preds = obs.x_preds
            y_preds =obs.y_preds
            psi_preds= obs.psi_preds
            stdx_preds =obs.stdx_preds
            stdy_preds= obs.stdy_preds
            mu1, cov1 = get_normal_dist(x_preds[idx1][k],y_preds[idx1][k],psi_preds[idx1][k],stdx_preds[idx1][k], stdy_preds[idx1][k]) ##get_normal_dist(obs, 0, k )#(final_pred_dicts, pred_dict, o, idx1, k )
            mu2, cov2 = get_normal_dist(x_preds[idx2][k],y_preds[idx2][k],psi_preds[idx2][k],stdx_preds[idx2][k], stdy_preds[idx2][k]) # (final_pred_dicts, pred_dict, o, idx2, k )
            distinct = check_distribution_distinguishable(mu1, cov1, mu2, cov2)
            if not distinct:
                break
    return k


def get_relevant_obstacle(ObstacleArray, risk_matrix):
    """use risk metrics"""
    risk_obj =np.array(risk_matrix).flatten()  #np.sum(risk_matrix, axis=0)
    idx_max = np.argsort(risk_obj)[::-1]
    threshold = 0.8*risk_obj[idx_max[0]]
    idcs = np.where(risk_obj >= threshold)[0]
    obstacles = [ObstacleArray[i] for i in idcs]
    return obstacles
        
def mahalanobis_distance(point, mean, covariance):
    delta = point - mean
    inv_covariance = np.linalg.inv(covariance)
    distance = np.sqrt(delta.T @ inv_covariance @ delta)
    return distance


def Bhattacharyya_Distance(m1, S1,m2, S2):
    chol = linalg.cholesky((S1+S2)/2, lower=True)
    det = np.prod(np.diag(chol))**2
    dist = (linalg.solve_triangular(chol, m1-m2, lower=True)**2).sum()
    det1 = linalg.det((S1))
    det2 = linalg.det((S2))
    
    b_dist= (1/8)*dist + 0.5*np.log(det/np.sqrt(det1*det2))
    return b_dist

    
def product_distribution(mu1, cov1, mu2, cov2):
    # cov1_inv = np.linalg.inv(cov1)
    # cov2_inv = np.linalg.inv(cov2)
    # cov_product = np.linalg.inv(cov1_inv + cov2_inv)
    # mu_product = cov_product @ (cov1_inv @ mu1 + cov2_inv @ mu2)
    nom = np.linalg.inv(cov1 + cov2)
    cov_product = cov1@nom@cov2
    mu_product = cov2@nom@mu1 + cov1@nom@mu2
    return mu_product

def product_prob(m1, S1,m2, S2):
    chol = linalg.cholesky(S1+S2, lower=True)
    det_advanced = 1/np.prod(np.diag(chol))
    dist_advanced = (linalg.solve_triangular(chol, m1-m2, lower=True)**2).sum()
    c_advanced = 1/(2*np.pi)**0.5 *det_advanced * np.exp(-1/2 * dist_advanced)
    
    return c_advanced
    
def check_distribution_distinguishable(mu1, cov1, mu2, cov2):
    #print("b-dist",Bhattacharyya_Distance(mu1, cov1, mu2, cov2))
    if Bhattacharyya_Distance(mu1, cov1, mu2, cov2)<= np.sqrt(0.7/3):
        distinct = False
    else:
        distinct = True
    return distinct
    
""" utils"""
def rot_mat(theta):
    theta = theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return rot



def get_normal_dist(x,y, psi, stdx, stdy ):
    mu_i = np.array([x,y])
    stdx = max(stdx, 0.08)
    stdy = max(stdy, 0.04)
    cov_i = np.diag((stdx**2, stdy**2))
    angle_i = psi # np.arctan(pred_dict["pred_trajs"][o,m,k,6]/pred_dict["pred_trajs"][o,m,k,5]) + pred_dict['input_dict']['center_objects_world'][o,6]

    cov_i = rot_mat(float(angle_i))@cov_i@rot_mat(float(angle_i)).T
    return mu_i, cov_i

def get_normal_dist_ca(x,y, psi, stdx, stdy ):
    mu_i = ca.vertcat(x,y)
    stdx = ca.fmax(stdx, 0.08)
    stdy = ca.fmax(stdy, 0.04)
    
    cov_i = ca.MX.eye( 2)
    cov_i[1,1] = stdy**2
    cov_i[0,0] =stdx**2
    angle_i = psi # np.arctan(pred_dict["pred_trajs"][o,m,k,6]/pred_dict["pred_trajs"][o,m,k,5]) + pred_dict['input_dict']['center_objects_world'][o,6]

    cov_i = rot_mat_ca(angle_i)@cov_i@rot_mat_ca(angle_i).T
    return mu_i, cov_i


def rot_mat_ca(theta):
    theta = theta
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    rot = ca.vertcat(ca.horzcat(cos_theta, -sin_theta), 
                              ca.horzcat(sin_theta, cos_theta))  # np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return rot
    
def euclidean_norm_ca(vector):
    """
    Calculate the Euclidean norm of a vector.
    
    Parameters:
        vector (list or tuple): The input vector.
        
    Returns:
        float: The Euclidean norm of the vector.
    """
    squared_sum = 0
    for i in range(vector.shape[0]):
        squared_sum += vector[i] ** 2
    return squared_sum ** 0.5



