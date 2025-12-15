"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import casadi as ca
import numpy as np
import time
import math





def funcPhi(x):
    return (1 - math.erf(-x / math.sqrt(2))) / 2

def funcGauss(x, my, sig): 
    return 1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - my) / sig)**2)

def collisionRate(poseEgo, poseObst, velEgo, velObst, sizeEgo, sizeObst, covarObst):
    """Function to calculate collision probability density between moving ego vehicle and moving obstacle for one prediction step.
    Input are pose, velocity, size and covariance matrix for ego vehicle and obstacle.
    To calculate the collision probability for one timestep, the density must be multiplicated with the length of the timestep.
    To calculate the total collision probability of two trajectories, the probabilities for all timestep must be accumulated.
    In rare cases, the total probability may be > 1 due to rounding issues.
    The expected time of the collision (TTC) is the timestep with the highest density.
    For details see: A. Philipp and D. Goehring, “Analytic collision risk calculation for autonomous vehicle navigation,” 
    in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 1744–1750. """
    alloToEgo = -poseEgo #// transform from global to ego-coordinates
    poseObstE =  poseObst + alloToEgo#// pose of obstacle in ego coordinates
    rot_matrix = rot_mat( alloToEgo[-1])
    velRel = rot_matrix @ (velObst - velEgo)# // relative velocity between ego and obstacle#// relative velocity in ego coord.
    covarXYVxVyEgo =  covarObst #// combined covar for ego and obstacle (in global coord)
    rotXYVxVy = np.zeros((4, 4)) #// rotation matrix for covariance
    rotXYVxVy[0:2, 0:2] = rot_matrix
    rotXYVxVy[2:4, 2:4] = rot_matrix
    covarXYVxVyEgo = rotXYVxVy @ covarXYVxVyEgo @ rotXYVxVy.T


    octoPts = np.zeros((5, 2)) #// corner points of collision octogon
    octoPts[0] = np.array([sizeEgo[0]/2 +max(sizeObst)/ 2 , sizeEgo[1] / 2 + max(sizeObst)/ 2]) #/ we start at upper left corner
    octoPts[1] = np.array([-sizeEgo[0]/2 -max(sizeObst)/ 2 , sizeEgo[1] / 2 + max(sizeObst)/ 2])
    octoPts[2] = -octoPts[0]
    octoPts[3] = -octoPts[1]
    octoPts[4] =  octoPts[0]


    distances = np.linalg.norm(octoPts[:4,:] - poseObstE[:2], axis=1)
    closest_indices = np.argsort(distances)[:2] 
    
    # Find indices of the two closest corners
    closest_indices = np.argsort(distances)[:2]
    colRate = 0

    fromPt = octoPts[closest_indices[1] ]
    toPt = octoPts[closest_indices[0] ]
    
    fromTo = toPt - fromPt
    fromTo = fromTo / np.linalg.norm(fromTo)
    newX = np.array([fromTo[1], -fromTo[0]])

    rotMat = np.array([[newX[0], newX[1]],
                       [-newX[1], newX[0]]])
    pObst2 = np.dot(rotMat, poseObstE[:2])
    vObst2 = np.dot(rotMat, velRel)
    fromPt = np.dot(rotMat, fromPt)
    toPt = np.dot(rotMat, toPt)
    
    MYvx_y_x = np.array([vObst2[0], pObst2[1], pObst2[0]])
    rotXYVxVy = np.array([[newX[0], -newX[1], 0, 0],
                         [newX[1], newX[0], 0, 0],
                         [0, 0, newX[0], -newX[1]],
                         [0, 0, newX[1], newX[0]]])

    covarXYVxVy = rotXYVxVy @ covarXYVxVyEgo @ rotXYVxVy.T #// rotate covar from ego coord. to side coord.
    
    SIGvx_y_x = np.array([[covarXYVxVy[2, 2], covarXYVxVy[1, 2], covarXYVxVy[0, 2]],
                          [covarXYVxVy[2, 1], covarXYVxVy[1, 1], covarXYVxVy[0, 1]],
                          [covarXYVxVy[2, 0], covarXYVxVy[1, 0], covarXYVxVy[0, 0]]]) #// reduced covariance for x-vel, y-pos and x-pos
    
    sh = SIGvx_y_x[0:2, 2] / SIGvx_y_x[2, 2]
    MYvx_y_cond_x0 = MYvx_y_x[0:2] + sh * (fromPt[0] - MYvx_y_x[2]) #/ expect. x-vel and y-pos, conditioned on x0
    SIGvx_y_cond_x0 = SIGvx_y_x[0:2, 0:2] - sh * SIGvx_y_x[0:2, 2:3].T
    my_vx_cond_x0 = MYvx_y_cond_x0[0]
    my_y_cond_x0 = MYvx_y_cond_x0[1]
    
    det = np.linalg.det(SIGvx_y_cond_x0)
    sig_vx_cond_x0 = np.sqrt(det / SIGvx_y_cond_x0[1, 1])
    sig_y_cond_x0 = np.sqrt(det / SIGvx_y_cond_x0[0, 0])
    #print(fromPt[0], MYvx_y_x[2], math.sqrt(SIGvx_y_x[2, 2]), funcGauss(fromPt[0], MYvx_y_x[2], math.sqrt(SIGvx_y_x[2, 2])))
    r =-funcGauss(fromPt[0], MYvx_y_x[2],  math.sqrt(SIGvx_y_x[2, 2])) * \
       ((my_vx_cond_x0 * funcPhi(-my_vx_cond_x0 / sig_vx_cond_x0) - sig_vx_cond_x0**2 * funcGauss(0, my_vx_cond_x0, sig_vx_cond_x0)) * \
        (funcPhi((toPt[1] - my_y_cond_x0) / sig_y_cond_x0) - funcPhi((fromPt[1] - my_y_cond_x0) / sig_y_cond_x0)))
    colRate += r
    #print("t",t_ - time.time())
    return colRate

def calculate_collision_risk( x0_, obstacle, m, params, f_risk, step=2):
    dt = params["dt"]
    #obstacle = ObstacleArray[o]
    m = min(m, len(obstacle.x_preds)-1)
    x_pred = obstacle.x_preds[m]
    y_pred =obstacle.y_preds[m]
    v_pred =obstacle.v_preds[m]
    psi_pred= obstacle.psi_preds[m]
    stdx_pred =obstacle.stdx_preds[m]
    stdy_pred= obstacle.stdy_preds[m]
    
    l =obstacle.l
    w= obstacle.w
    sizeObst =  np.array([l, w])

    coll_risk = f_risk(x_pred, y_pred,v_pred, psi_pred,stdx_pred, stdy_pred, x0_[:params["N"]+1,:], sizeObst)
    return coll_risk

def funcPhi_ca(x):
    return (1 - ca.erf(-x / ca.sqrt(2))) / 2

def funcGauss_ca(x, my, sig): 
    return 1 / (sig * ca.sqrt(2 * ca.pi)) * ca.exp(-0.5 * ((x - my) / sig)**2)

    
def init_collision_risk(params, step=2):
    dt = params["dt"]
    N = params["N"]
    sizeEgo = np.array([params["rob_r"]*3, params["rob_r"]*2]) 
    coll_risk = 0

    x_pred = ca.MX.sym('x_pred', N+1, 1) 
    y_pred = ca.MX.sym('y_pred', N+1, 1) 
    v_pred = ca.MX.sym('v_pred', N+1, 1) 
    psi_pred=  ca.MX.sym('psi_pred', N+1, 1) 
    stdx_pred =ca.MX.sym('stdx_pred', N+1, 1) 
    stdy_pred=  ca.MX.sym('stdy_pred', N+1, 1) 
    x0_ = ca.MX.sym('x0_', N+1, 7) 

    sizeObst = ca.MX.sym('sizeObst', 2, 1) 
    traj2 = ca.horzcat(x_pred,y_pred, psi_pred)  # final_pred_dicts[o]["pred_trajs"][uvdidx]
    covarObst = ca.MX(4,4)#np.diag(np.ones((4,)))*0.01  #  

    for k in range(0, N+1,step): #params["N"]+1, step):
        poseEgo  = x0_[k,:3]
        poseObst = traj2[k,:]#  np.concatenate((trajxy2[k,:], np.ravel(trajpsi2[k])))
        velEgo = x0_[k,3]
        velObst = v_pred[k]* ca.vertcat(ca.cos(psi_pred[k]),ca.sin(psi_pred[k]))

         # Covariance matrix of obstacle for X-pos, Y-pos, X-vel and Y-vel in global coordinates 4x4
        _, cov = get_normal_dist_ca(x_pred[k],y_pred[k],psi_pred[k],stdx_pred[k], stdy_pred[k])#(obstacle, m, k )

        covarObst[2:4, 2:4] = cov # (covarObst[0:2, 0:2]-cov)/params["dt"] #1st order taylor approx of v
        covarObst[0:2, 0:2] = cov
        
        """collisionRate"""
        alloToEgo = -poseEgo #// transform from global to ego-coordinates
        poseObstE =  poseObst + alloToEgo#// pose of obstacle in ego coordinates
        rot_matrix = rot_mat_ca( alloToEgo[-1])
        velRel = rot_matrix @ (velObst - velEgo)# // relative velocity between ego and obstacle#// relative velocity in ego coord.
        covarXYVxVyEgo =  covarObst #// combined covar for ego and obstacle (in global coord)
        rotXYVxVy = ca.MX(4,4) #// rotation matrix for covariance
        rotXYVxVy[0:2, 0:2] = rot_matrix
        rotXYVxVy[2:4, 2:4] = rot_matrix
        covarXYVxVyEgo = rotXYVxVy @ covarXYVxVyEgo @ rotXYVxVy.T
    
        octoPts = ca.MX(5,2)#np.zeros((5, 2)) #// corner points of collision octogon
        octoPts[0,:] = ca.vertcat(sizeEgo[0]/2 +ca.fmax(sizeObst[0],sizeObst[1])/ 2 , sizeEgo[1] / 2 + ca.fmax(sizeObst[0],sizeObst[1])/ 2) #/ we start at upper left corner
        octoPts[1,:] = ca.vertcat(-sizeEgo[0]/2 -ca.fmax(sizeObst[0],sizeObst[1])/ 2 , sizeEgo[1] / 2 + ca.fmax(sizeObst[0],sizeObst[1])/ 2)
        octoPts[2,:] = -octoPts[0,:]
        octoPts[3,:] = -octoPts[1,:]
        octoPts[4,:] =  octoPts[0,:]

        distances = ca.MX(4,1)

        for i in range(4):
            distances[i,:] = euclidean_norm_ca((octoPts[i,:] - poseObstE[:2]).T)

        #closest_indices = np.argsort(distances)[:2] 
        min_value = ca.mmin(distances)
        second_min_value = ca.inf #ca.if_else(x[1] < x[0], x[1], min_value)
        min_index = -1
        second_min_index = -1 #ca.if_else(x[1] < x[0], 1, 0)

        # Loop through each element to find the minimum and second smallest values and their indices
        for i in range(0, distances.shape[0]):
            min_index = ca.if_else(distances[i] == min_value, i, min_index)
        
        for i in range(0, distances.shape[0]):
            second_min_value1 = ca.if_else( (distances[i] < second_min_value ), distances[i], second_min_value)
            second_min_index1 = ca.if_else(distances[i] == second_min_value1, i, second_min_index)
            second_min_value = ca.if_else( ( second_min_index1 != min_index)  , second_min_value1 , second_min_value)
            second_min_index = ca.if_else(second_min_index1 != min_index, second_min_index1, second_min_index)
            
        """ Find indices of the two closest corners"""
        fromPt = octoPts[min_index,: ]
        toPt = octoPts[second_min_index,:] 

        # #closest_indices = np.argsort(distances)[:2]
        colRate = 0
    
        fromTo = toPt - fromPt
        fromTo = fromTo.T / euclidean_norm_ca(fromTo.T)
        newX = ca.vertcat(fromTo[1], -fromTo[0])

        rotMat = ca.vertcat(ca.horzcat(newX[0], newX[1]), 
                              ca.horzcat(-newX[1], newX[0])) 
        
        pObst2 = rotMat@poseObstE[:2].T
        vObst2 = rotMat@ velRel
        fromPt = rotMat@ fromPt.T
        toPt = rotMat@ toPt.T
        
        MYvx_y_x = ca.vertcat(vObst2[0], pObst2[1], pObst2[0])
        rotXYVxVy = ca.vertcat(ca.horzcat(newX[0], -newX[1], 0, 0),
                             ca.horzcat(newX[1], newX[0], 0, 0),
                             ca.horzcat(0, 0, newX[0], -newX[1]),
                             ca.horzcat(0, 0, newX[1], newX[0]))
        
        covarXYVxVy = rotXYVxVy @ covarXYVxVyEgo @ rotXYVxVy.T #// rotate covar from ego coord. to side coord.
        
        SIGvx_y_x = ca.vertcat(ca.horzcat(covarXYVxVy[2, 2], covarXYVxVy[1, 2], covarXYVxVy[0, 2]),
                              ca.horzcat(covarXYVxVy[2, 1], covarXYVxVy[1, 1], covarXYVxVy[0, 1]),
                              ca.horzcat(covarXYVxVy[2, 0], covarXYVxVy[1, 0], covarXYVxVy[0, 0])) #// reduced covariance for x-vel, y-pos and x-pos
        
        
        sh = SIGvx_y_x[0:2, 2] / SIGvx_y_x[2, 2]
        MYvx_y_cond_x0 = MYvx_y_x[0:2] + sh * (fromPt[0] - MYvx_y_x[2]) #/ expect. x-vel and y-pos, conditioned on x0
        SIGvx_y_cond_x0 = SIGvx_y_x[0:2, 0:2] - sh @ SIGvx_y_x[0:2, 2:3].T
        my_vx_cond_x0 = MYvx_y_cond_x0[0]
        my_y_cond_x0 = MYvx_y_cond_x0[1]
        
        det =SIGvx_y_cond_x0[0,0] * SIGvx_y_cond_x0[1,1] -SIGvx_y_cond_x0[0,1] * SIGvx_y_cond_x0[1,0] #ca.det(SIGvx_y_cond_x0)
        sig_vx_cond_x0 = ca.sqrt(det / SIGvx_y_cond_x0[1, 1])
        sig_y_cond_x0 = ca.sqrt(det / SIGvx_y_cond_x0[0, 0])
        
        r = -funcGauss_ca(fromPt[0], MYvx_y_x[2],  ca.sqrt(SIGvx_y_x[2, 2])) * \
           ((my_vx_cond_x0 * funcPhi_ca(-my_vx_cond_x0 / sig_vx_cond_x0) - sig_vx_cond_x0**2 * funcGauss_ca(0, my_vx_cond_x0, sig_vx_cond_x0)) * \
             (funcPhi_ca((toPt[1] - my_y_cond_x0) / sig_y_cond_x0) - funcPhi_ca((fromPt[1] - my_y_cond_x0) / sig_y_cond_x0)))
        colRate += r
        coll_risk+= colRate*dt*step

    
    f_risk = ca.Function('f_risk', [x_pred, y_pred,v_pred, psi_pred,stdx_pred, stdy_pred, x0_, sizeObst], [ coll_risk])
    return f_risk


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
    angle_i = psi 
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