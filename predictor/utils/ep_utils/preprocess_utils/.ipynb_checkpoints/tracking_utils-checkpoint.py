'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import numpy as np
from scipy.linalg import block_diag
from scipy.special import binom

from predictor.utils.ep_utils.preprocess_utils import Trajectory 
from predictor.utils.ep_utils.preprocess_utils import GaussianControlPointDensity
from predictor.utils.ep_utils.preprocess_utils import TrajectoryObservation

def polynomial_basis_function(x, power):
    return x ** power


def expand(x, bf, bf_args=None):
    if bf_args is None:
        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
    else:
        return np.array([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args]).T

# get the prior form the Empirical Bayes analysis
def get_bernstein_prior(degree, timescale, prior_data, spacedim=2):
    # The prior was calculatd for monomials and is
    # organized in the following way
    # x0x0 x0x1 x0x2 x0y0 x0y1 x0y2
    # x1x0 x1x1 x1x2 x1y0 x1y1 x1y2
    # x2x0 x2x1 x2x2 x2y0 x2y1 x2y2
    # y0x0 y0x1 y0x2 y0y0 y0y1 y0y2
    # y1x0 y1x1 y0x2 y1y0 y1y1 y1y2
    # y2x0 y2x1 y2x2 y2y0 y2y1 y2y2

    # We reorganize the prior into the following structure
    # x0x0 x0y0 x0x1 x0y1 x0x2 x0y2
    # y0x0 y0y0 y0x1 y0y1 y0x2 y0y2
    # x1x0 x1y0 x1x1 x1y1 x1x2 x1y2
    # ....
    perm = np.zeros((2 * degree, 2 * degree))
    for i in range(degree):
        perm[2 * i, i] = 1
        perm[2 * i + 1, degree + i] = 1

    # all trajectories in EB analysis start at (0, 0), so there is little uncertainty about start point
    monomial_cov_unscaled = np.diag(np.block([1e-3, 1e-3, np.zeros(spacedim * degree)]))
    monomial_cov_unscaled[2:, 2:] = np.array(perm @ prior_data['A_list'][degree - 1] @ perm.T)
    
    monomial_scale = np.diag([timescale ** deg for deg in range(0, degree + 1)])
    monomial_scale = np.kron(monomial_scale, np.eye(spacedim))
    monomial_cov =  monomial_scale @ monomial_cov_unscaled @ monomial_scale.T
    
    monomial_mean = monomial_scale @ np.kron(np.zeros(degree + 1), np.ones(spacedim))
    
    # Now we transform to Bernstein Polynomials
    M = np.zeros((degree + 1, degree + 1))

    for k in range(0, degree + 1):
        for i in range(k, degree + 1):
            M[i, k] = (-1)**(i - k) * binom(degree, i) * binom(i, k)
        
    M_inv = np.linalg.solve(M, np.eye(degree + 1))
    future_prior = np.kron(M_inv, np.eye(2)) @ monomial_cov @ np.kron(M_inv, np.eye(2)).T
    
    # reorder entries for history
    perm = np.zeros((degree + 1, degree + 1))
    perm[np.arange(degree + 1), np.arange(degree, -1, -1)] = 1
    perm = np.kron(perm, np.eye(2))
    history_prior = perm @ future_prior @ perm.T
    
    return future_prior, history_prior


# The Noise Model from the Emprical Bayes Analysis
def build_ego_observation_noise_covariance(degree, prior_data):        
    b_diag = prior_data['B_list'][degree - 1]['B_diag'][0]
    b_by_diag = prior_data['B_list'][degree - 1]['B_by_diag'][0]
    
    R_world = np.array([[b_diag, b_by_diag],
                        [b_by_diag, b_diag]])
    

    return R_world 


# The Lidar Noise Model from the Emprical Bayes Analysis
def build_obj_observation_noise_covariance(ego_pos, ego_heading, obj_pos, degree, prior_data):        
    r = np.linalg.norm(obj_pos - ego_pos)
    alpha = np.arctan2(*(obj_pos - ego_pos)[::-1]) - ego_heading

    beta0 =  prior_data['B_list'][degree - 1]['B_d'][0][0]
    beta1 = prior_data['B_list'][degree - 1]['B_d'][1][0]
    beta2 = prior_data['B_list'][degree - 1]['B_d'][2][0]
    sigma_r = np.sqrt(beta0 + beta1 * r + beta2 * r**2)
    
    R_ego, R_world = cartesian_noise_from_polar_noise(
       r, alpha, sigma_r=sigma_r, sigma_alpha=np.sqrt(prior_data['B_list'][degree - 1]['B_theta'][0]), sigma_c=np.sqrt(prior_data['B_list'][degree - 1]['B_const'][0]), ego_heading=ego_heading
    )

    return R_world 


# The Lidar Noise Model from the Emprical Bayes Analysis
def build_obj_observation_noise_covariance_batch(ego_pos, ego_heading, obj_pos, degree, prior_data):        
    d = obj_pos - ego_pos

    r = np.linalg.norm(d, axis = -1)
    
    R_temp = np.zeros((d.shape[0], 2, 2))
    
    alpha = np.arctan2(d[:,1], d[:,0]) - ego_heading
    c_ego_to_map, s_ego_to_map = np.cos(ego_heading), np.sin(ego_heading)
    R_ego_to_map = np.transpose(np.array(((c_ego_to_map, -s_ego_to_map), (s_ego_to_map, c_ego_to_map))), (2, 0, 1))
    phi_r = expand(r, bf=polynomial_basis_function, bf_args=range(1, 2+1))
    
    if isinstance(prior_data, list):
        beta_r =  np.array([p['B_list'][degree - 1]['B_d'] for p in prior_data])
        var_alpha =  np.array([p['B_list'][degree - 1]['B_theta'] for p in prior_data])
        var_const =  np.array([p['B_list'][degree - 1]['B_const'] for p in prior_data])
        
        var_r = (phi_r[:, None, :] @ beta_r)[:, 0, 0]
        var_alpha = var_alpha.squeeze(-1)
        var_const = var_const.squeeze(-1)
    else:
        beta_r =  prior_data['B_list'][degree - 1]['B_d']
        var_alpha =  prior_data['B_list'][degree - 1]['B_theta']
        var_const =  prior_data['B_list'][degree - 1]['B_const']
    
        var_r = (phi_r @ beta_r)[:, 0]
    

    var_lon_lon = var_r * np.power(np.cos(alpha), 2) + var_alpha * np.power(r, 2) * np.power(np.sin(alpha), 2)
    var_lat_lat = var_r * np.power(np.sin(alpha), 2) + var_alpha * np.power(r, 2) * np.power(np.cos(alpha), 2)
    var_lon_lat = var_r * np.sin(alpha) * np.cos(alpha) - var_alpha *  np.power(r, 2) * np.sin(alpha) * np.cos(alpha)


    R_temp[:, 0, 0] += (var_lon_lon + var_const)
    R_temp[:, 1, 1] += (var_lat_lat + var_const)
    R_temp[:, 0, 1] += var_lon_lat
    R_temp[:, 1, 0] += var_lon_lat

    R_world = R_ego_to_map @ R_temp @ R_ego_to_map.transpose((0,2,1))

    return R_world


def build_obj_observation_noise_covariance_scene(x_t, av_idx, degree, prior_data_list):
    ego_prior_data = prior_data_list[av_idx]
    agt_prior_data_list = [p for i, p in enumerate(prior_data_list) if i != av_idx]
    
    ego_mask = np.zeros((x_t.shape[0]), dtype=bool)
    ego_mask[av_idx] = True
    agt_mask = ~ego_mask
    
    x_ego = x_t[ego_mask]
    
    obs_cov = np.zeros((x_t.shape[0], 2,2))
    obs_cov[ego_mask] = build_ego_observation_noise_covariance(degree=degree,
                                                               prior_data = ego_prior_data)
    
    if np.any(agt_mask):
        x_agt = x_t[agt_mask]
        x_ego_batch = np.repeat(x_ego, x_agt.shape[0], axis=0)

        obs_cov[agt_mask] = build_obj_observation_noise_covariance_batch(ego_pos= x_ego_batch[:, :2],
                                                                   ego_heading = x_ego_batch[:, 2],
                                                                   obj_pos = x_agt[:, :2],
                                                                   degree = degree,
                                                                   prior_data = agt_prior_data_list)
    
    return obs_cov


def cartesian_noise_from_polar_noise(
    r, alpha, sigma_r, sigma_alpha, sigma_c, ego_heading
):
    """Generate a Cartesian Observation Noise covariance from a polar one

    Parameters
    ----------
    r: float
        distance between sensor and object

    alpha: float, radians
        bearing angle of object seen from sensor

    sigma_r: float
        standard deviation of r

    sigma_alpha: float
        standard deviation of alpha

    sigma_c: float
        standard deviation for timing jitter

    ego_heading: float, radians
        sensor heading in world coordinate system

    Returns
    -------
    R_ego: ndarray
        cartesian position noise covariance in sensor coordinates

    R_world: ndarray
        cartesian position noise covariance in world coordinates,
        i.e. rotated by ego_heading
    """

    s_xx = (sigma_r * np.cos(alpha)) ** 2 + (sigma_alpha * r * np.sin(alpha)) ** 2
    s_yy = (sigma_r * np.sin(alpha)) ** 2 + (sigma_alpha * r * np.cos(alpha)) ** 2
    s_xy = (sigma_r**2 - sigma_alpha**2 * r**2) * np.sin(alpha) * np.cos(alpha)

    R_ego = np.array([[s_xx, s_xy], [s_xy, s_yy]])

    Rot = np.array(
        [
            [np.cos(ego_heading), np.sin(ego_heading)],
            [-np.sin(ego_heading), np.cos(ego_heading)],
        ]
    )

    R_world = Rot.T @ (R_ego + sigma_c**2 * np.eye(2)) @ Rot

    return R_ego, R_world


def get_initial_state_ego(pos, v0, prior_cov, TRAJ, BASIS, TIMESCALE):
    initial_state = GaussianControlPointDensity(
        x=np.kron(np.arange(BASIS.size)[::-1], -v0), # note the minus here!
        P=prior_cov
    )
    
    OM = TrajectoryObservation(
        TRAJ, t=np.linspace(0, TIMESCALE, BASIS.size - 1),
        derivatives=[[1] for _ in range(BASIS.size - 2)] + [[1, 0]],
        R=[0.5**2 * np.eye(2)] * (BASIS.size - 2) + [np.diag([0.5, 0.5, 0.1, 0.1])**2] 
    )

    z = np.block([v0 for _ in range(BASIS.size - 1)] + [0, 0])
    initial_state = initial_state.update(z, OM)
    # shift to the observed position
    initial_state._x += np.kron(np.ones(BASIS.size), pos)
    return initial_state


def get_initial_state_agt(pos, v0, prior_cov, TRAJ, BASIS, TIMESCALE):
    initial_state = GaussianControlPointDensity(
        x=np.kron(np.arange(BASIS.size)[::-1], -v0), # note the minus here!
        P=prior_cov
    )
    
    OM = TrajectoryObservation(
        TRAJ, t=np.linspace(0, TIMESCALE, BASIS.size - 1),
        derivatives=[[1] for _ in range(BASIS.size - 2)] + [[1, 0]],
        R=[3**2 * np.eye(2)] * (BASIS.size - 2) + [np.diag([3, 3, 0.1, 0.1])**2] 
    )

    z = np.block([v0 for _ in range(BASIS.size - 1)] + [0, 0])
    initial_state = initial_state.update(z, OM)
    # shift to the observed position
    initial_state._x += np.kron(np.ones(BASIS.size), pos)
    return initial_state

def D_matrix(degree, derivative): 
    '''
    A derivative matrix for monomial basis function
    param degree: the order of polynomial
    param derivative: the order of derivatives
    '''
    D=np.eye(degree+1)
    for i in range(derivative):
        D = D @ np.diag(np.arange(1,degree+1), 1)

    return D 


def get_monomial_prior(prior_data, degree, spacedim = 2):
    # The prior was calculatd for monomials and is
    # organized in the following way
    # x0x0 x0x1 x0x2 x0y0 x0y1 x0y2
    # x1x0 x1x1 x1x2 x1y0 x1y1 x1y2
    # x2x0 x2x1 x2x2 x2y0 x2y1 x2y2
    # y0x0 y0x1 y0x2 y0y0 y0y1 y0y2
    # y1x0 y1x1 y0x2 y1y0 y1y1 y1y2
    # y2x0 y2x1 y2x2 y2y0 y2y1 y2y2
    #prior_data = json.load(open('logs/gradient_tape/agt_xy_polar_plus_const_50/result_summary.json', 'r'))

    # We reorganize the prior into the following structure
    # x0x0 x0y0 x0x1 x0y1 x0x2 x0y2
    # y0x0 y0y0 y0x1 y0y1 y0x2 y0y2
    # x1x0 x1y0 x1x1 x1y1 x1x2 x1y2
    # ....
    perm = np.zeros((2 * degree, 2 * degree))
    for i in range(degree):
        perm[2 * i, i] = 1
        perm[2 * i + 1, degree + i] = 1
    
    # all trajectories in EB analysis start at (0, 0), so there is little uncertainty about start point
    monomial_cov_unscaled = np.diag(np.block([1e8, 1e8, np.zeros(spacedim * degree)]))
    monomial_cov_unscaled[2:, 2:] = np.array(perm @ prior_data['A_list'][degree - 1] @ perm.T)
    
    return monomial_cov_unscaled

def monomial_to_bernstein(monomial_mean, monomial_cov, timescale, degree, spacedim = 2):
    monomial_scale = np.diag([timescale ** deg for deg in range(0, degree + 1)])
    monomial_scale = np.kron(monomial_scale, np.eye(spacedim))
    
    monomial_cov_scaled =  monomial_scale @ monomial_cov @ monomial_scale.T
    monomial_mean_scaled = monomial_scale @ monomial_mean
    
    # Now we transform to Bernstein Polynomials
    M = np.zeros((degree + 1, degree + 1))

    for k in range(0, degree + 1):
        for i in range(k, degree + 1):
            M[i, k] = (-1)**(i - k) * binom(degree, i) * binom(i, k)
        
    M_inv = np.linalg.solve(M, np.eye(degree + 1))
    bernstein_cov = np.kron(M_inv, np.eye(spacedim)) @ monomial_cov_scaled @ np.kron(M_inv, np.eye(spacedim)).T
    bernstein_mean = np.kron(M_inv, np.eye(spacedim)) @ monomial_mean_scaled
    
    return bernstein_mean, bernstein_cov


def bayesian_regression_ego(trajectory, timestamps, prior_data, timescale, degree, spacedim = 2, basis_return = 'B', ignore_idx = 15):
    '''
    trajectory: [T, 4)]
    timestamps: [T]
    prior: [(DEG+1) * SPACEDIM, (DEG+1) * SPACEDIM]
    degree: int
    basis_return: either 'B' for Bernstein or 'M' for Monomial
    '''
    assert trajectory.shape[0] == len(timestamps)
    
    omega_0 = np.zeros((degree+1)*spacedim)
    V_0 = get_monomial_prior(prior_data, degree) 
    V_0_inv = np.linalg.solve(V_0, np.eye(V_0.shape[0]))
    
    assert V_0.shape[0] == (degree+1) * spacedim
    
    start_pos = trajectory[0, :2]
    pos_obs = trajectory[:, :2] - start_pos
    heading_obs = trajectory[ignore_idx:, 2]
    vel_obs = trajectory[ignore_idx:, -2:]
    
    pos_obs_cov = build_ego_observation_noise_covariance(degree = degree, prior_data = prior_data) # [2, 2]
   
    pos_obs_cov = np.kron(np.eye(len(timestamps)), pos_obs_cov) # Note the left kronecker!
    pos_obs_cov[-2:, -2:] = pos_obs_cov[-2:, -2:] / 16 # set smaller obs cov for last point, beneficical for prediction
    
    
    vel_obs_cov = np.diag([0.5, 0.1])**2 # lon, lat
    R = np.transpose(np.array([[np.cos(heading_obs), -np.sin(heading_obs)],
                 [np.sin(heading_obs), np.cos(heading_obs)]]), (2,0,1))
     
    vel_obs_cov = R @ vel_obs_cov @ np.transpose(R, (0,2,1))

    phi_m = np.vander(timestamps, N = (degree+1), increasing= True)
    phi_m_dot = phi_m[ignore_idx:] @ D_matrix(degree = degree, derivative = 1)
    
    Phi = np.kron(phi_m, np.eye(spacedim)) # [2*T, 2*(DEG+1)]
    Phi_dot = np.kron(phi_m_dot, np.eye(spacedim)) # [2*T, 2*(DEG+1)]
    
    Phi = np.concatenate([Phi, Phi_dot], axis = 0) # [200, (DEG+1)*2]
    obs_cov = block_diag(pos_obs_cov, block_diag(*vel_obs_cov)) # [200, 200]
    y = np.concatenate([pos_obs.reshape(-1), vel_obs.reshape(-1)], axis = 0) # [200]
    
    
    # Bayesian Regression
    phi_T_sigma_o = np.transpose(np.linalg.solve(obs_cov, Phi))

    V_N_inv = V_0_inv + phi_T_sigma_o @ Phi

    params_cov = np.linalg.solve(V_N_inv, np.eye(V_N_inv.shape[0])) # [(DEG+1)*2, (DEG+1)*2]
    params_mean =  np.squeeze(params_cov @ phi_T_sigma_o @ np.expand_dims(y, axis = -1), axis=-1) # [(DEG+1)*2]
    
    if basis_return == 'B': #return in bernstein
        params_mean, params_cov = monomial_to_bernstein(monomial_mean = params_mean,
                                                        monomial_cov = params_cov,
                                                        timescale = timescale,
                                                        degree = degree,
                                                        spacedim = spacedim)
                                                        
        params_mean = np.reshape(params_mean, (-1,2)) + start_pos
    else: # return in monomial
        params_mean = np.reshape(params_mean, (-1,2))
        params_mean[0, :] = params_mean[0, :] + start_pos 
    
    return params_mean, params_cov



def bayesian_regression_agt(agt_trajectory, ego_trajectory, timestamps, prior_data, timescale, degree, spacedim = 2, basis_return = 'B',  ignore_idx = 15):
    '''
    trajectory: [T, 4)]
    timestamps: [T]
    prior: [(DEG+1) * SPACEDIM, (DEG+1) * SPACEDIM]
    degree: int
    basis_return: either 'B' for Bernstein or 'M' for Monomial
    '''
    assert agt_trajectory.shape[0] == ego_trajectory.shape[0] == len(timestamps)
    
    omega_0 = np.zeros((degree+1)*spacedim)
    V_0 = get_monomial_prior(prior_data, degree) 
    V_0_inv = np.linalg.solve(V_0, np.eye(V_0.shape[0]))
    
    assert V_0.shape[0] == (degree+1) * spacedim
    
    ignore_idx = min(ignore_idx, agt_trajectory.shape[0]-1)
    
    start_pos = agt_trajectory[0, :2]
    pos_obs = agt_trajectory[:, :2] - start_pos
    heading_obs = agt_trajectory[ignore_idx:, 2]
    vel_obs = agt_trajectory[ignore_idx:, -2:]
    
    pos_obs_cov = []
    for ego_x, ego_theta, obj_x in zip(ego_trajectory[:, :2], ego_trajectory[:, 2], agt_trajectory[:, :2]):
        pos_obs_cov.append(build_obj_observation_noise_covariance(ego_pos = ego_x, 
                                                                 ego_heading =ego_theta,
                                                                 obj_pos = obj_x,
                                                                 degree = degree, 
                                                                 prior_data = prior_data)) # [2, 2]
   
    #pos_obs_cov = np.kron(np.eye(len(timestamps)), pos_obs_cov) # Note the left kronecker!
    #pos_obs_cov[-1] = pos_obs_cov[-1] /16 # set smaller obs cov for last point, beneficical for prediction
    pos_obs_cov = block_diag(*pos_obs_cov)
    
    
    vel_obs_cov = np.diag([4, 4])**2 # lon, lat
    R = np.transpose(np.array([[np.cos(heading_obs), -np.sin(heading_obs)],
                 [np.sin(heading_obs), np.cos(heading_obs)]]), (2,0,1))
     
    vel_obs_cov = R @ vel_obs_cov @ np.transpose(R, (0,2,1))

    phi_m = np.vander(timestamps, N = (degree+1), increasing= True)
    phi_m_dot = phi_m[ignore_idx:] @ D_matrix(degree = degree, derivative = 1)
    
    Phi = np.kron(phi_m, np.eye(spacedim)) # [2*T, 2*(DEG+1)]
    Phi_dot = np.kron(phi_m_dot, np.eye(spacedim)) # [2*T, 2*(DEG+1)]
    
    Phi = np.concatenate([Phi, Phi_dot], axis = 0) # [200, (DEG+1)*2]
    obs_cov = block_diag(pos_obs_cov, block_diag(*vel_obs_cov)) # [200, 200]
    y = np.concatenate([pos_obs.reshape(-1), vel_obs.reshape(-1)], axis = 0) # [200]
    
    
    # Bayesian Regression
    phi_T_sigma_o = np.transpose(np.linalg.solve(obs_cov, Phi))

    V_N_inv = V_0_inv + phi_T_sigma_o @ Phi

    params_cov = np.linalg.solve(V_N_inv, np.eye(V_N_inv.shape[0])) # [(DEG+1)*2, (DEG+1)*2]
    params_mean =  np.squeeze(params_cov @ phi_T_sigma_o @ np.expand_dims(y, axis = -1), axis=-1) # [(DEG+1)*2]
    
    if basis_return == 'B': #return in bernstein
        params_mean, params_cov = monomial_to_bernstein(monomial_mean = params_mean,
                                                        monomial_cov = params_cov,
                                                        timescale = timescale,
                                                        degree = degree,
                                                        spacedim = spacedim)
                                                        
        params_mean = np.reshape(params_mean, (-1,2)) + start_pos
    else: # return in monomial
        params_mean = np.reshape(params_mean, (-1,2))
        params_mean[0, :] = params_mean[0, :] + start_pos 
    
    return params_mean, params_cov