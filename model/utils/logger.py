import numpy as np 
import pandas as pd 

from casadi import DM 

class Logger:
    def __init__(self, x_true, mu_t, Sigma_t):      
        self.nx = mu_t.shape[0]
        self.xs = [DM(x_true).full().reshape(self.nx,)]
        self.mus = [DM(mu_t).full().reshape(self.nx,)]
        self.sigmas = [DM(Sigma_t).full()]
        self.us_planned = []
        self.us = []
        self.ts = [0]
        self.segments = [1]
        
        #assume no initial measurement 
        self.measurements = [[]]
        self.measured_coords = [[]]

        self.planned_trajectories = [[]]

        self.measurements_belief = [[]]
        self.measured_coords_belief = [[]]
        
    def log(self, dt, x_true, mu_t, Sigma_t, u_t, u_t_noisy, task, measurements):        
        self.ts.append(self.ts[-1] + dt) 
        self.xs.append(x_true.full().reshape(self.nx,))
        self.mus.append(DM(mu_t).full().reshape(self.nx,))
        self.sigmas.append(DM(Sigma_t).full())
        self.us_planned.append(u_t)
        self.us.append(u_t_noisy) ## record the noisy? 
        self.segments.append(task.segment)

        self.measurements.append(measurements['z_t'])
        self.measured_coords.append(measurements['coords'])


    def log_belief_measurement(self,measurements):

        self.measurements_belief.append(measurements['z_t'])
        self.measured_coords_belief.append(measurements['coords'])



    def log_planned_trajectory(self, planned_tr):
        self.planned_trajectories.append(planned_tr)

        
    def get_dict(self):    
        log_dict = {}
        
        log_dict['ts'] = self.ts
        log_dict['xs'] = self.xs
        log_dict['mus'] = self.mus
        log_dict['us'] = self.us
        log_dict['us_planned'] = self.us_planned
        log_dict['sigmas'] = self.sigmas
        log_dict['segments'] = self.segments

        #also add the measurements
        log_dict['measurements'] = self.measurements
        log_dict['measured_coords'] = self.measured_coords
        
        return log_dict


def save_trajectory(xs, segments, timing, tr_id, unique_t_id, params, jitter = (0,0)):
    from datetime import datetime
    tr = np.array(xs)
    
    df = pd.DataFrame(columns = ['VPCode', 'trajectory_id', 'unique_trajectory_id', 'trial', 'condition', 'segment', 'timing', 'x', 'z', 'angle', 'jitterx', 'jittery'])
    
    df.x = tr[:,0]
    df.z = tr[:,1]
    df.angle = tr[:,2]
    df.trial = 1
    df.condition = 1 
    df.segment = segments
    df.timing = timing
    df.trajectory_id = tr_id
    df.VPCode = 'Simulation'
    df.day = 1 
    df.unique_trajectory_id = unique_t_id
    df.jitterx = jitter[0]
    df.jittery = jitter[1]
    
    for p in ['w_running_control', 'w_running_position_uncertainty', 'w_running_map_uncertainty', 'w_running_state', 'w_final_state', 'w_final_uncertainty']:
        df[p] = float(params[p])
        
    return df 

