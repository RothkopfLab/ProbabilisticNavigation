"""
 Parameters for Task, Model and Algorithm 
"""

import numpy as np 

from experiment.get_experiment import get_experiment


def get_params(experiment_name = 'chen2017' , start_position = None, trajectory_id = '1-4-8', target_jitter = True, n_landmarks = 3, landmark_shift_dir = -1):
    ####################################################################################################
                                        ##### Task Parameters #####
    ####################################################################################################
    experiment = get_experiment(experiment_name)
    params = experiment.set_task_parameters(trajectory_id, target_jitter = target_jitter, start_position = start_position, n_landmarks = n_landmarks, landmark_shift_dir = landmark_shift_dir)

    ####################################################################################################
                                    ##### Initialization Parameters #####
    ####################################################################################################
    #Prior Knowledge about landmarks (e.g. assume perfect knowledge)
    #TODO set the initial location and rotation prior 
    params['position_prior'] = None 
    params['landmark_prior'] = params['landmark_locations']
    params['pickup_location_prior'] = params['pickup_locations']

    #Initial Uncertainty 
    params['initial_positional_uncertainty'] = [0.5**2, 0.5**2, 0.05**2]
    params['initial_landmark_uncertainty'] = [(5.0, 5.0)] * params['n_landmarks']
    params['initial_pickup_uncertainty'] = [(1e6, 1e6)] * params['n_pickups']

    ####################################################################################################
                                    ##### Noise Parameters #####
    ####################################################################################################

    #enable Noise parameters in generative model 
    params['observation_noise_enabled'] = True
    params['motion_noise_enabled']      = True
    params['representation_noise_enabled']  = True

    params['representation_noise_type'] = 'all' #all,invisible,none
    params['representation_random_walk'] =  True 
     
    #used in generative model (noise parameters)
    params['motion_cov'] = np.array([0.1, 0.001, 0.0625/3, 0.125/1.25])  #np.array([0.5, 0.01,  np.deg2rad(1.33), np.deg2rad(7.5)]) 
    params['observation_cov'] = np.array([0.015,  0.125,  0.03, 0.15]) * 1.5 #np.array([0.10, 1.0, np.deg2rad(1), np.deg2rad(10.0)])    
    params['representation_cov'] =  0.01295**2 #right now mostly adding variability at the end 

    #used in state-estimation (subjectve uncertainties)
    params['motion_cov_subj'] = params['motion_cov']
    params['observation_cov_subj'] =  params['observation_cov']
    params['representation_cov_subj'] = params['representation_cov']

    ####################################################################################################
                                ##### Cost + Behavioral Parameters #####
    ####################################################################################################
   
    #Cost Function Weights
    params['cost_function_weights'] = {
                                'w_running_control': 2.5,
                                'w_running_position_uncertainty':  0.25, 
                                'w_running_map_uncertainty' : 0.25, 
                                'w_running_state':  7.5, 
                                'w_final_state': 10.0, 
                                'w_final_uncertainty': 0.0}
    #Cost Parameters  
    params['Q'] = np.diag([1.0, 1.0, 0.0]) 
    params['R'] = np.diag([1.0, 0.0]) 
    params['R_tr'] = 0.25 

    #Motion Limits 
    params['motion_limits'] = {'v_max' : 1.22, 
                               'v_min' : 0.0, 
                               'w_max' : 1.5 *  np.pi, 
                               'w_min' : 1.5 * -np.pi}

    #maximum accelerations
    params['motion_limits']['dv_max'] =  0.25
    params['motion_limits']['dw_max'] =  np.pi/4

    ####################################################################################################
                                    ##### Algorithm Parameters #####
    ####################################################################################################
    params['planning_horizon'] = 15
    params['control_horizon'] = 5
    params['goal_epsilon'] = 0.125

    params['dt'] = 1/10
    params['max_wait_time'] = 1.5 

    params['a0'] = 0.3
    params['k'] = 1.4
    params['n_updates'] = 15 

    params['reorientation_enabled'] = True
    params['reorient'] = True
    params['reorientation_tresh'] = 2
    
    return params
