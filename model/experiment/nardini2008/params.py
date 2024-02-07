"""Contains relevant task parameters and helper functions for the study by Nardini et al 2008 
Reference:
    Nardini, M., Jones, P., Bedford, R., & Braddick, O. (2008). 
    Development of Cue Integration in Human Navigation. 
    Current Biology, 18(9), 689–693. [https://doi.org/10.1016/j.cub.2008.04.021](https://doi.org/10.1016/j.cub.2008.04.021)
"""

import matplotlib.pyplot as plt 
import numpy as np
import copy 
import pickle 
import os 

from casadi import vertcat 
from utils.helpers import get_heading, get_route

from experiment.experiment_params import ExperimentParameters


class nardini_2008_parameters(ExperimentParameters):
    def __init__(self):
        super()
        self.landmark_locations = [2.5, 7.0, 4.324335495461293, 4.824335495461293, 0.6756645045387075, 4.824335495461293]
        self.unstable_lm = {}


        self.trajectories = ['0-5-8', '0-7-8', '1-4-8', '1-6-8', '2-5-8', '2-7-8', '3-4-8', '3-6-8']
        self.conditions = ['landmark', 'self-motion', 'combined', 'conflict']
        self.arena_limits = ((0,5), (-1,8)) #x_lim, y_lim
        self.rotations = [15.0]
        self.landmark_shift_dir = (-1,-1)


        self.post_ids = {
        0: np.array([3.4531183112762975, 4.467673493904492]),
        1: np.array([1.5468816887237025, 4.467673493904492]),
        2: np.array([2.8339157419089536, 4.717847571033412]),
        3: np.array([2.1660842580910464, 4.717847571033412]),
        4: np.array([3.2134771358696854, 4.098658444008506]),
        5: np.array([1.7865228641303146, 4.098658444008506]),
        6: np.array([2.749959783943274, 4.28593161031644]),
        7: np.array([2.250040216056726, 4.28593161031644]),
        8: np.array([2.5, 3])}


        self.start_position_per_id = {
        '0-5-8': np.array([2.5, 0]),
        '0-7-8': np.array([2.5, 0]),
        '1-4-8': np.array([2.5, 0]),
        '1-6-8': np.array([2.5, 0]),
        '2-5-8': np.array([2.5, 0]),
        '2-7-8': np.array([2.5, 0]),
        '3-4-8': np.array([2.5, 0]),
        '3-6-8': np.array([2.5, 0])
        }

        curr_path = os.path.dirname(os.path.abspath(__file__))
        self.swc_locations = None

    def set_task_parameters(self, trajectory_id, start_position, target_jitter = True, n_landmarks = 3, landmark_shift_dir = -1):
        ####################################################################################################
                                        ##### Task Parameters #####
        ####################################################################################################
        params = {}
        params['experiment'] = 'nardini2008'

        if trajectory_id == 'random':
            trajectory_id = np.random.choice(self.trajectories)

        #Triangle
        params['trajectory_id'] = trajectory_id
        params['arena_limits'] = ((0,5), (-1,8))

        if start_position is not None:
            params['start_position'] = start_position
        else:
            params['start_position'] = self.start_position_per_id[trajectory_id]
        
        params['route'] = get_route(self.post_ids, params['trajectory_id'], params['start_position'])
        params['target_jitter'] = (0,0)
        params['route'][1] = params['route'][1] #no target jitter 
        params['start'] = vertcat(params['route'][0], get_heading(params['route'][0], params['route'][1])[0])
        params['pickup_locations'] = np.array(params['route']).flatten().tolist()[2:-2]
        params['n_pickups'] = len(params['pickup_locations'])//2

        #Landmarks
        params['n_landmarks'] = n_landmarks 
        params['landmark_locations'] = self.landmark_locations[:2*n_landmarks]

        #Condition 
        params['condition'] = 'combined' #(landmark, self-motion, combined, conflict)

        #wait_time at the end of trajectory
        params['t_max'] = 8.0

        #conflict: turning speed and moving swivel chair 
        params['turning_speed'] = np.pi 
        params['swivel_chair'] = 'move' #stay, resample
        params['landmark_rotation_direction'] = landmark_shift_dir
        params['landmark_rotation'] = -15.0
    
        #Field of View
        params['fov'] = 180

        #observations (make all landmarks and the all pickups visible)
        params['obs_state'] =   [1] * params['n_landmarks']  + [1]  +  (params['n_pickups']-1) * [1]
        params['obs_belief'] =  [1] * params['n_landmarks']  + [1]  +  (params['n_pickups']-1) * [1]

        #add body-rotation
        params['body_rotation'] = False
        params['reorient_prior_homing'] = True 

        return params 
    




