"""
Model Predictive Control 
"""

import time 
import numpy as np
from casadi import *


from state_estimation import * 

from utils.helpers import normalize_angle, get_landmark_mu, get_landmark_sigma, get_landmark_side, get_heading, rotate
from utils.live_plotting import LivePlot
from utils.logger import Logger 

from experiment.task import Task 


def init_state(params, model, planner, solver):
    lm_true = vertcat(params['landmark_locations'], params['pickup_locations'])
    x_true = vertcat(params['start'], lm_true)

    obs0 = params['obs_state'][:(len(params['landmark_locations']) // 2)] + (len(params['pickup_locations']) // 2) * [0]    
    obs_state = params['obs_state']
    

    lm0 = vertcat(params['landmark_prior'], params['pickup_location_prior'])

    #belief observability 
    obs_belief = params['obs_belief']
    model.obs0 = obs_belief

    model.representation_random_walk = params['representation_random_walk']
    
    #initial belief landmaark prior 
    lm_init = vertcat(params['landmark_prior'], params['pickup_location_prior'])

    #init belief state 
    mu_t = DM(vertcat(params['start'], lm_init))
    Sigma_t = SX.eye(model.nx)
    
    #positional
    Sigma_t[0,0] = params['initial_positional_uncertainty'][0]
    Sigma_t[1,1] = params['initial_positional_uncertainty'][1]
    Sigma_t[2,2] = params['initial_positional_uncertainty'][2]

    #landmarks 
    for i in range(0,len(params['landmark_locations']),2):
        Sigma_t[3+i, 3+i] =    params['initial_landmark_uncertainty'][i//2][0]
        Sigma_t[3+i+1, 3+i+1] = params['initial_landmark_uncertainty'][i//2][1]
      

    nl = len(params['landmark_locations'])
    for i in range(0, 2 * len(params['initial_pickup_uncertainty']), 2):
        Sigma_t[3+nl+i,   3+nl+i] = params['initial_pickup_uncertainty'][i//2][0]
        Sigma_t[3+nl+i+1, 3+nl+i+1] = params['initial_pickup_uncertainty'][i//2][1]
        

    #task params 
    task = Task(mu_t, params['goal_epsilon'], obs0, params['condition'], n_landmarks= params['n_landmarks'], n_pickups = params['n_pickups'], rotation = params['landmark_rotation'], body_rotation= params['body_rotation'])
    goal = task.goal_location 

    task.t_max = params['t_max']
        
    #initialize planner parameters 
    p = planner.P(1)
    p['start'] = mu_t
    p['S0'] = Sigma_t
    p['end'] =  vertcat(goal, 0.0)
    p['alpha'] = params['a0']
    p['observable'] = params['obs_belief']
    p['Q'] = params['Q']
    p['R'] = params['R']
    p['dt'] = params['dt']
    p['u_k_prev'] = vertcat(0.0, 0.0)
    
    for k in params['cost_function_weights'].keys():
        p[k] = params['cost_function_weights'][k]
    

    dt = params['dt']
    
    logger = Logger(x_true, mu_t, Sigma_t)
    
    #initialization 
    U0 = np.zeros((planner.N,2))
    X0 = repmat(mu_t,planner.N+1,1)
    
    #planner args
    args = {}
    args['x0'] = vertcat(X0.reshape((model.nx*(planner.N+1),1)), U0.reshape((2*planner.N,1)))
    args['p'] = p
    
    return x_true, mu_t, Sigma_t, obs0, obs_state, obs_belief, p, args, task, logger


def Belief_Space_Planning(planner, solver, args, p, sol, params, alpha_update = True, verbose = False):
    start_time = time.time()
    
    if sol is None:
        sol = solver(x0 = args['x0'], lbx = planner.lbx, ubx = planner.ubx, lbg = planner.lbg, ubg = planner.ubg,  p = args['p'])
    else:
        #do warm-start 
        sol = solver(x0 = args['x0'], lbx = planner.lbx, ubx = planner.ubx, lbg = planner.lbg, ubg = planner.ubg,  p = args['p'], lam_x0 = sol['lam_x'], lam_g0 = sol['lam_g'])
    
    if alpha_update:
        sol, p = planner.alpha_update   (sol, solver, p, params['k'], params['n_updates'], verbose = False)    
    
    x_res, u_res = planner.obtain_solution(sol)
    
    if verbose:
        print(time.time() - start_time)
    
    return sol, p, x_res, u_res 


def State_Update(x_true, u_t, model, params):
    u_t_noisy = u_t + np.random.multivariate_normal([0,0], model.process_cov_generative(u_t)).reshape(1, model.nu) * params['motion_noise_enabled']
    
    #advance true state based on controls 
    x_true = x_true + model.F(x_true, u_t_noisy, params['dt'])
    x_true[2] = normalize_angle(x_true[2])
    
    return x_true, u_t_noisy  

def State_Observation(x_true, model, obs_belief, obs_state, params):
    #make observation
    z_t = model.h(x_true)
    z_t_noisy = z_t + DM(np.random.multivariate_normal(np.zeros(model.nz), model.obs_cov_generative(x_true))) * params['observation_noise_enabled']

    #get currently observable landmarks from true state 
    observable = np.array(model.d_j(x_true, 10000, obs_state)).diagonal()[0:-1:2]
    measurements = get_measurements(observable.nonzero()[0], x_true, z_t_noisy, obs_state)

    observable = [idx for idx in observable.astype(int).nonzero()[0]]

    return measurements, observable


def EKF_SLAM_Prediction_Step(mu_t, Sigma_t, u_t, model, dt):  
    mu_t, Sigma_t = EKF_SLAM_Prediction(mu_t, Sigma_t, u_t, model.process_cov(u_t), dt = dt)

    #add representation noise (biased)
    if model.representation_random_walk:
        mu_t[3:] += np.random.multivariate_normal(np.zeros(mu_t[3:].shape[0]), model.representation_cov(model.obs0))
    
    Sigma_t[3:,3:] += model.representation_cov(model.obs0)

    mu_t[2] = normalize_angle(mu_t[2])
    
    return mu_t, Sigma_t 
    
def EKF_SLAM_Correction_Step(mu_t, Sigma_t, z_t, model):
    for z_t_i in z_t['z_t']:
        lm_idx = int(z_t_i[2])
        
        obs_cov = model.obs_cov(mu_t)[2 * lm_idx : 2 * lm_idx + 2, 2 * lm_idx : 2 * lm_idx + 2]
        mu_t, Sigma_t = EKF_SLAM_Correction(mu_t, Sigma_t, z_t_i, obs_cov)
        mu_t[2] = normalize_angle(mu_t[2])
        
    return mu_t, Sigma_t 


def MPC_Reorientation(reorientation_ctr, mpc_iter, x_true, mu_t, Sigma_t, model, params, state_observation, obs_state, obs_belief, task, live_plot, logger, reorientation_tresh = 5, verbose = False):
    belief_observation = [idx for idx in np.array(model.d_j(DM(mu_t), 10000, obs_state)).diagonal()[0:-1:2].nonzero()[0]]
    
    if (belief_observation != state_observation): 
        if verbose: print('disoriented')
        reorientation_ctr +=1 
    else:
        if verbose: print('oriented')
        reorientation_ctr = 0 
        
    if reorientation_ctr >= reorientation_tresh and (len(belief_observation) > 0):
        if verbose: print('need reorientation')
        last_known_belief_observation = [idx for idx in np.array(model.d_j(DM(mu_t), 10000, obs_state)).diagonal()[0:-1:2].nonzero()[0]]
        last_known_dir = get_landmark_side(mu_t, last_known_belief_observation[0]) #rotate left or right? 
        
        while(last_known_belief_observation != state_observation):
            start_time = time.time()
            if verbose: print('reorienting')
            u_t = DM([0.0, last_known_dir * np.pi / 2 ]).T #rotate in direction of last known control 
            
            #State Update and Observation
            x_true, u_t_noisy = State_Update(x_true, u_t, model, params)
            z_t, observable   = State_Observation(x_true, model, obs_belief, obs_state, params)

            #log belief observation 
            z_t_belief, observable_ = State_Observation(DM(mu_t).full(), model, obs_belief, obs_state, params)
            logger.log_belief_measurement(z_t_belief)
            
            #Belief State Update 
            mu_t, Sigma_t  = EKF_SLAM_Prediction_Step(mu_t, Sigma_t, u_t, model, params['dt'])
            mu_t, Sigma_t  = EKF_SLAM_Correction_Step(mu_t, Sigma_t, z_t, model)


            
            # Logging + Plotting 
            logger.log(params['dt'], x_true, mu_t, Sigma_t, u_t, u_t_noisy, task, z_t)
            logger.log_planned_trajectory(- 100 * np.ones((3,15)))



            if live_plot: 
                live_plot.update_plot(mpc_iter, None, logger, None, params, model.nz, False).draw_figure(0.25)
            
            mpc_iter += 1 
            
            if len(observable) > 0:
                if verbose: print('reoriented')
                reorientation_ctr = 0
                if verbose : print(start_time - time.time())
                break 
                
    return mpc_iter, reorientation_ctr, x_true, mu_t, Sigma_t