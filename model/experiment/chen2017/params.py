"""Contains relevant task parameters and helper functions for the study by Chen et al 2017. 
Reference:
    Chen, X., McNamara, T. P., Kelly, J. W., & Wolbers, T. (2017). 
    Cue combination in human spatial navigation. Cognitive Psychology, 95, 105–144. 
"""

import matplotlib.pyplot as plt 
import numpy as np
import copy 
import os 
import pickle 

from casadi import vertcat 
from utils.helpers import get_heading, get_route

from experiment.experiment_params import ExperimentParameters

class chen_2017_parameters(ExperimentParameters):
    def __init__(self):
        super()

        self.landmark_locations = [0, 3.75, 2.50, 2.3, -2.50, 2.3]
        self.unstable_lm = {0: [[-2.8332599999999997, 2.8219, -0.00107, 3.99505, 2.82085, 2.8249]],
                            1: [[-3.44547, 2.4214, 0.013269999999999999, 2.50887, 3.4339199999999996, 2.4214]],
                            2: [[-2.94694, 1.70216, 0.85308, 4.920730000000001, 2.94329, 1.705]],
                            3: [[-2.1884900000000003, 2.60717, 1.28075, 4.83057, 4.3350599999999995, 1.16487]],
                            4: [[-2.6092400000000002, 2.19962, -1.54566, 4.21732, 2.18292, 2.6110599999999997]],
                            5: [[-1.15529, 3.19401, 2.2326099999999998, 3.88982, 4.238919999999999, 1.55894]],
                            6: [[-2.95333, 1.70908, -0.8635700000000001, 4.91501, 2.94329, 1.705]],
                            7: [[-4.22311, 1.55788, -2.26005, 3.88602, 1.14848, 3.1924200000000003]],
                            8: [[-4.34394, 1.16396, -1.31019, 4.82545, 2.18292, 2.6110599999999997]]}


        self.trajectories = ['0-4-8', '0-5-8', '0-6-8', '0-7-8', '1-4-8', '1-5-8', '1-6-8', '1-7-8', '2-4-8', '2-5-8', '2-6-8', '2-7-8', '3-4-8', '3-5-8', '3-6-8', '3-7-8']
        self.conditions = ['landmark', 'self-motion', 'combined', 'conflict']
        self.arena_limits = ((-3,3), (-5,5)) #x_lim, y_lim
        self.rotations = [15.0]
        self.landmark_shift_dir = (-1,-1)


        self.post_ids = {
        0: np.array([1.8031417210826617, 0.42097392528230815]),
        1: np.array([-1.8031417210826617, 0.42097392528230815]),
        2: np.array([2.1104984444124772, -0.3397606599404287]),
        3: np.array([-2.1104984444124772, -0.3397606599404287]),
        4: np.array([1.0986584440085057, -0.03652286413031447]),
        5: np.array([-1.0986584440085057, -0.03652286413031447]),
        6: np.array([1.2859316103164398, -0.5000402160567263]),
        7: np.array([-1.2859316103164398, -0.5000402160567263]),
        8: np.array([0, -0.75])}


        self.start_position_per_id = {
        '0-0-0': np.array([-1.,  2.]),
        '0-4-8': np.array([-1.,  2.]),
        '0-5-8': np.array([-1.,  2.]),
        '0-6-8': np.array([-1.,  2.]),
        '0-7-8': np.array([-1.,  2.]),
        '1-4-8': np.array([1., 2.]),
        '1-5-8': np.array([1., 2.]),
        '1-6-8': np.array([1., 2.]),
        '1-7-8': np.array([1., 2.]),
        '2-4-8': np.array([-1.,  2.]),
        '2-5-8': np.array([-1.,  2.]),
        '2-6-8': np.array([-1.,  2.]),
        '2-7-8': np.array([-1.,  2.]),
        '3-4-8': np.array([1., 2.]),
        '3-5-8': np.array([1., 2.]),
        '3-6-8': np.array([1., 2.]),
        '3-7-8': np.array([1., 2.])}

        curr_path = os.path.dirname(os.path.abspath(__file__))

        #load swivel chair locations 
        with open(curr_path + '/chen2017_exp4_swivel_chair_locations.pickle', 'rb') as handle:
            self.swc_locations = pickle.load(handle)
        
        #self.swc_locations = None

    def set_task_parameters(self, trajectory_id, start_position, target_jitter = True, n_landmarks = 3, landmark_shift_dir = -1):
        ####################################################################################################
                                        ##### Task Parameters #####
        ####################################################################################################
        params = {}

        params['experiment'] = 'chen2017'
        params['arena_limits'] = ((-3,3), (-5,5))

        #Triangle 
        if trajectory_id == 'random':
            trajectory_id = np.random.choice(self.trajectories)

        params['trajectory_id'] = trajectory_id
        
        if start_position is not None:
            params['start_position'] = start_position
        else:
            params['start_position'] = self.start_position_per_id[trajectory_id]
        
        params['route'] = get_route(self.post_ids, params['trajectory_id'], params['start_position'])
        params['target_jitter'] = self.calculate_target_jitter() * target_jitter
        params['route'][1] = params['route'][1] + params['target_jitter']
        params['start'] = vertcat(params['route'][0], get_heading(params['route'][0], params['route'][1])[0])
        params['pickup_locations'] = np.array(params['route']).flatten().tolist()[2:-2]
        params['n_pickups'] = len(params['pickup_locations'])//2
    
        #Landmarks
        params['n_landmarks'] = n_landmarks 
        params['landmark_locations'] = self.landmark_locations[:2*n_landmarks]

        #Condition 
        params['condition'] = 'combined' #(landmark, self-motion, combined, conflict)

        #wait_time at the end of trajectory
        params['t_max'] = 20.0

        #conflict: turning speed and moving swivel chair 
        params['turning_speed'] = np.pi 
        params['swivel_chair'] = 'move' #stay, resample
        params['landmark_shift_dir'] = landmark_shift_dir
        params['landmark_rotation'] = -15.0

        #Field of View
        params['fov'] = 60

        #observations (make all landmarks and the first pickup visible)
        params['obs_state'] =   [1] * params['n_landmarks']  + [1]  +  (params['n_pickups']-1) * [0]
        params['obs_belief'] =  [1] * params['n_landmarks']  + [1]  +  (params['n_pickups']-1) * [0]

        #add body-rotation
        params['body_rotation'] = False
        params['reorient_prior_homing'] = False 

        return params 


    def calculate_target_jitter(self):
        # by randomly sampling the jitter’s x and y coordinates from a Gaussian distribution with mean = 0 and standard deviation = 0.25 m (
        # these are not the correct values from the paper ; but ones that better approximate the true distribution from the empirical data)
        jitter = np.random.multivariate_normal(np.zeros(2), 0.125**2 * np.eye(2))

        #Then, an angle theta was ranomly sampled from a uniform distribution between 0 and 360
        theta = np.deg2rad(np.random.uniform(0,360))

        #The final coordinate values of the jitter were calculated as, x0 = x * cos(theta) and y0 = y * sin(theta)
        jitter = jitter * np.array([np.cos(theta), np.sin(theta)])

        #Both x0 and y0 were set to 0.4m or 0.4m if they exceeded the range [-0.4m,0.4m]
        jitter = np.clip(jitter, -0.4, 0.4) 
        return jitter 

