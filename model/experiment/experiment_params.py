import matplotlib.pyplot as plt 
import numpy as np
import copy 
import pickle 
from utils.helpers import get_heading, get_route


trial_types = {
    1 : 'landmark',
    2 : 'self-motion',
    3 : 'combined', 
    4 : 'conflict'
}

trial_nums ={v: k for k, v in trial_types.items()}

class ExperimentParameters():

  def __init__(self):
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
    self.landmark_rotations = [15.0]
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

    self.swc_locations = None


    def set_task_parameters(self,trajectory_id, start_position, target_jitter = True, n_landmarks = 3, landmark_shift_dir = -1):
        pass 

