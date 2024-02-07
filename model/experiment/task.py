import numpy as np 
from casadi import DM 
import copy 

from utils.helpers import get_landmark_mu, normalize_angle

class Task:
    def __init__(self, mu_t, goal_epsilon, obs0, condition, n_landmarks = 1, n_pickups = 3, rotation = -15, body_rotation = False):
        self.curr_path = 'outbound'
        self.curr_goal = 0
        self.segment  = 1 
        self.obs0 = obs0

        self.n_landmarks = n_landmarks
        self.n_pickups = n_pickups 

        self.goal_location = np.array(get_landmark_mu(mu_t, self.n_landmarks + self.curr_goal)).reshape(2,)
        self.goal_epsilon = goal_epsilon
        self.t_wait = 0.0
        self.t_max = 20.0

        self.rotated = True
        self.landmark_rotation = rotation
        self.body_rotation  = body_rotation
        self.start_rotation = 0.0
        
        self.reinit = False
        self.spin_in_chair = False
        self.moved_swivel_chair = False 
        self.condition = condition
        self.task_finished  = False
        self.n_landmarks = n_landmarks


        self.obs_belief = []
        self.obs_state  = []
              
    def set_next_goal(self, x_true, mu_t, obs_belief, obs_state):     
        #curr_goal cant be seen anymore
        obs_belief[self.n_landmarks + self.curr_goal] = 0
        obs_state[self.n_landmarks  +  self.curr_goal] = 0

        #advance goal and make only landmarks + current goal visible 
        self.curr_goal += 1 
        
        #TODO (change this to adopt for different number of landmarks)
        if self.curr_goal < self.n_pickups:
            #change observability of targets
            obs_state = self.obs0
            obs_state[self.n_landmarks + self.curr_goal] = 1 
            obs_belief[self.n_landmarks + self.curr_goal] = 1 
                
            self.segment +=1
        
        else:
            self.curr_path = 'return path'
            self.segment = 0 
            self.curr_goal = 0

        
            if self.condition == 'landmark':
                obs_belief =  self.n_landmarks  * [1]  + self.n_pickups * [0]
                obs_state  =  self.n_landmarks  * [1]  + self.n_pickups * [0]
                self.spin_in_chair = True
            elif self.condition == 'self-motion':
                # No Landmarks (Self-motion): In the self-motion condition, the visual world was rendered invisible when participants reached the last post, so that they could only use self-motion cues.
                obs_belief =  self.n_landmarks  * [0]  + [1] +(self.n_pickups-1) * [0] #no landmark available; and black screen 
                obs_state  =  self.n_landmarks  * [0]  + self.n_pickups * [0]
                self.spin_in_chair = False
            elif self.condition == 'combination':
                # Combined (Visual + Self-motion): in the combination condition, participants were not disoriented and the world remained visible, so both landmarks and self-motion cues were available.
                obs_belief =  self.n_landmarks  * [1]  + self.n_pickups * [0]
                obs_state  =  self.n_landmarks  * [1]  + self.n_pickups * [0]
                self.spin_in_chair = False
            elif self.condition == 'conflict':
                # Conclict (Landmarks 15° Rotated): conflict condition, the landmark configuration was rotated clockwise by 15° so that the correct location defined by landmarks was different from the one defined by self-motion cues.
                obs_belief =  self.n_landmarks  * [1]  +  self.n_pickups * [0]
                obs_state  =  self.n_landmarks  * [1]  +  self.n_pickups * [0]
                self.spin_in_chair = False
                self.rotated = False
                        
        self.set_belief_goal(mu_t)
        self.obs_belief = obs_belief 
        self.obs_state  = obs_state 
        
    
    def set_belief_goal(self,mu_t):  
        self.goal_location = np.array(get_landmark_mu(DM(mu_t), self.n_landmarks + self.curr_goal)).reshape(2,)

    def move_swivel_chair(self,x_true, mu_t, Sigma_t, experiment, params):

        if params['motion_noise_enabled']:
            x_true_prior = copy.deepcopy(x_true[:2])
            swivel_chair_location = np.zeros(2)

            #TODO make this if statement flexible, such that it supports no resampling as well 
            if experiment.swc_locations is not None:
                swivel_chair_location = experiment.swc_locations[params['trajectory_id']][int(np.random.choice(np.arange(len(experiment.swc_locations[params['trajectory_id']]))))]
                x_true[:2] = swivel_chair_location
            else:
                swivel_chair_location = DM(np.random.multivariate_normal(np.array(x_true[:2]).reshape(2,), params['chair_replacement_variability'])) #slighly randomly reposition
                x_true[:2] = swivel_chair_location
                pass 

            #move the swivel chair in the same direction (translation of x and y)
            swivel_chair_movement = (x_true[:2] - x_true_prior).full()
            swivel_chair_variability = np.diag(swivel_chair_movement.reshape(2,))**2 / 2
            
            #update belief accordingly
            mu_t[:2] += np.random.multivariate_normal(swivel_chair_movement.reshape(2,), swivel_chair_variability)
            Sigma_t[:2,:2] += DM(swivel_chair_variability)  #increase uncertainty about position 

        #is this correct??
        curr_goal = DM(mu_t[:2])

        self.moved_swivel_chair = True

        return x_true, mu_t, Sigma_t, curr_goal 

    def waiting_period(self, x_true, mu_t, Sigma_t, experiment, params, curr_goal, verbose):
        if self.spin_in_chair:
            #move participant into swivel chair 
            if not self.moved_swivel_chair:
                x_true, mu_t, Sigma_t, curr_goal = self.move_swivel_chair(x_true, mu_t, Sigma_t, experiment, params)
            
            #rotate 
            if verbose: print('spinning')

            #reorient towards landmarks again as in Nardini 2008
            if ((self.t_max - self.t_wait) < 1.2) and params['reorient_prior_homing']:
                if verbose: print('rotate until 90 degrees are reached')
                if np.allclose(np.pi/2, normalize_angle(x_true[2]), np.deg2rad(10.0)):
                    if verbose: print('rotation has been reached')
                    self.t_wait = self.t_max #end the waiting 
                    u_t = DM([0.0, 0.0]).T
                else: 
                    u_t = DM([0.1, params['turning_speed']]).T
            else:
                #only turn a little in the beginning, don't go completely crazy
                if (self.t_wait < self.t_max/4): 
                    u_t = DM([0.1, params['turning_speed']]).T
                else: 
                    u_t = DM([0.0, 0.0]).T

        elif self.body_rotation:
            #turn until within 10 degrees tolerance 
            if verbose: print('rotate until 270 degrees are reached')
            if np.allclose(normalize_angle(self.start_rotation + np.deg2rad(270)), x_true[2], np.deg2rad(10.0)):
                if verbose: print('rotation has been reached')
                self.body_rotation = False
            else: 
                u_t = DM([0.0, np.pi]).T

        else:
            if verbose: print('waiting')
            #u_t =  DM([np.random.randn() * 0.025, np.random.randn() * 0.025]).T #random jitter
            u_t =  DM([0.0, 0.0]).T #random jitter
            #Sigma_t[:3,:3] += DM(np.diag([0.025, 0.025, 0.01])) #small process noise 
           
        
        #Sigma_t[3:,3:] += DM(np.diag((params['n_landmarks'] + params['n_pickups']) * [1e6, 1e6])) #small process noise 

        return x_true, mu_t, Sigma_t, u_t, curr_goal
