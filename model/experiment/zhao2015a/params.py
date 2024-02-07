"""Contains relevant task parameters and helper functions for the study by Zhao et al 2015 
Reference:
    Zhao, M., & Warren, W. H. (2015a). How You Get There From Here: Interaction of Visual Landmarks and Path Integration in Human Navigation. 
    Psychological Science, 26(6), 915–924. https://doi.org/10.1177/0956797615574952
"""

import matplotlib.pyplot as plt 
import numpy as np
import copy 
import pickle
import os 

from casadi import vertcat 
from utils.helpers import get_heading, get_route 

from experiment.experiment_params import ExperimentParameters

class zhao_2015a_parameters(ExperimentParameters):
    def __init__(self):
        super()
        
        self.landmark_locations = [-3.89,3.89,0.01,5.5,3.89,3.89] #TODO why is this 0.01?


        #500m 
        self.landmark_locations_distal = [353.55339059, 353.55339059, 3.061617e-14, 5.000000e+02, -353.55339059,  353.55339059]
        
        self.trajectories = ['0-14-24', '1-18-24', '2-11-24', '3-13-24', '0-15-24', '1-19-24', '2-10-24', '3-12-24', '0-16-24', '1-22-24', '2-8-24', '3-9-24', '0-21-24', '1-20-24', '2-4-24', '3-7-24', '0-17-24', '1-23-24', '2-6-24', '3-5-24']
        self.conditions = ['landmark', 'self-motion', 'combined', 'conflict']
        self.landmark_shift_dir = (1,-1)
        self.rotations = [15.0, 30.0, 45.0, 90.0, 115.0, 135.0]

        self.start_id ={0: np.array([-1.5, -2. ]),
        1: np.array([-0.75, -3.  ]),
        2: np.array([ 0., -1.]),
        3: np.array([ 0.75, -3.  ]),
        4: np.array([ 1.5, -2. ])}

        self.post_ids = {0: np.array([-3.12,  1.8 ]),
        1: np.array([-1.8 ,  3.12]),
        2: np.array([1.8 , 3.12]),
        3: np.array([3.12, 1.8 ]),
        4: np.array([-3.38,  0.91]),
        5: np.array([-2.6,  1.5]),
        6: np.array([-2.5,  0. ]),
        7: np.array([-1.77,  1.77]),
        8: np.array([-1.73,  1.  ]),
        9: np.array([-1.5 ,  3.03]),
        10: np.array([-1.414,  1.414]),
        11: np.array([-1.25,  2.17]),
        12: np.array([-0.78,  2.9 ]),
        13: np.array([0., 2.]),
        14: np.array([0., 3.]),
        15: np.array([0.91, 3.38]),
        16: np.array([1.25, 2.17]),
        17: np.array([1.73, 1.  ]),
        18: np.array([1.75, 3.03]),
        19: np.array([1.77, 1.77]),
        20: np.array([1.93, 0.52]),
        21: np.array([2.12, 2.12]),
        22: np.array([2.6, 1.5]),
        23: np.array([3., 0.]),
        24: np.array([0., 0.])}

        self.start_position_per_id = {'0-14-24': np.array([-1.5, -2. ]),
        '1-18-24': np.array([-1.5, -2. ]),
        '2-11-24': np.array([-1.5, -2. ]),
        '3-13-24': np.array([-1.5, -2. ]),
        '0-15-24': np.array([-0.75, -3.  ]),
        '1-19-24': np.array([-0.75, -3.  ]),
        '2-10-24': np.array([-0.75, -3.  ]),
        '3-12-24': np.array([-0.75, -3.  ]),
        '0-16-24': np.array([ 0., -1.]),
        '1-22-24': np.array([ 0., -1.]),
        '2-8-24': np.array([ 0., -1.]),
        '3-9-24': np.array([ 0., -1.]),
        '0-21-24': np.array([ 0.75, -3.  ]),
        '1-20-24': np.array([ 0.75, -3.  ]),
        '2-4-24': np.array([ 0.75, -3.  ]),
        '3-7-24': np.array([ 0.75, -3.  ]),
        '0-17-24': np.array([ 1.5, -2. ]),
        '1-23-24': np.array([ 1.5, -2. ]),
        '2-6-24': np.array([ 1.5, -2. ]),
        '3-5-24': np.array([ 1.5, -2. ])}

        #swivel chair locations 
        curr_path = os.path.dirname(os.path.abspath(__file__))

        #load swivel chair locations 
        with open(curr_path + '/zhao2015_swivel_chair_locations.pickle', 'rb') as handle:
            self.swc_locations = pickle.load(handle)
        
        self.swc_locations = None

    def set_task_parameters(self, trajectory_id, start_position, target_jitter = False, n_landmarks = 3, landmark_shift_dir = -1):
        ####################################################################################################
                                        ##### Task Parameters #####
        ####################################################################################################
        params = {}
        params['experiment'] = 'zhao2015a'

        if trajectory_id == '1-4-8':
            trajectory_id = '1-20-24'
        elif trajectory_id == 'random':
            trajectory_id = np.random.choice(self.trajectories)
            
        #Triangle 
        params['trajectory_id'] = trajectory_id
        params['arena_limits'] = ((-6,6), (-6,6))

        if start_position is not None:
            params['start_position'] = start_position
        else:
            params['start_position'] = self.start_position_per_id[trajectory_id]
        
        params['route'] = get_route(self.post_ids, params['trajectory_id'], params['start_position'])
        params['target_jitter'] = (0.0, 0.0) 
        params['route'][1] = params['route'][1]
        params['start'] = vertcat(params[ 'route'][0], get_heading(params['route'][0], params['route'][1])[0])
        params['pickup_locations'] = np.array(params['route']).flatten().tolist()[2:-2]
        params['n_pickups'] = len(params['pickup_locations'])//2
    
        #Landmarks
        params['n_landmarks'] = n_landmarks 
        params['landmark_locations'] = self.landmark_locations[:2*n_landmarks]

        #puts the landmarks 500m away maintaining the same degree of visual angle (distal landmarks condition)
        
        if target_jitter:
            params['landmark_locations'] = self.landmark_locations_distal[:2*n_landmarks]

        #Condition 
        params['condition'] = 'combined' #(landmark, self-motion, combined, conflict)

        #wait_time at the end of trajectory
        params['t_max'] = 10.0

        #conflict: turning speed and moving swivel chair 
        params['turning_speed'] = np.deg2rad(73) # 73 according to paper
        params['swivel_chair'] = 'move' #stay, resample
        params['landmark_shift_dir'] = landmark_shift_dir
        params['landmark_rotation'] = 15.0 * params['landmark_shift_dir']
        params['chair_replacement_variability'] = np.diag([0.5, 0.5])**2 #affects variability as well 
    
        #Field of View (63 degrees)=
        params['fov'] = 63

        #observations (make all landmarks and pickups are visible sequentially)
        params['obs_state'] =   [1] * params['n_landmarks']  + [1]  +  (params['n_pickups']-1) * [0]
        params['obs_belief'] =  [1] * params['n_landmarks']  + [1]  +  (params['n_pickups']-1) * [0]

        #add body-rotation
        params['body_rotation'] = False
        params['reorient_prior_homing'] = False 

        return params 
