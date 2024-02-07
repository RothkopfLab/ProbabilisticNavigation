import copy
import time
import random 

import numpy as np
import pandas as pd 
from numpy import matlib as mb
import matplotlib.pyplot as plt

#Casadi
from casadi import * 
import casadi.tools as cat

#Model 
from model import Model
from belief_space_planner import BeliefSpacePlanner
from state_estimation import * 
from model_predictive_control import *

#Task 
from experiment.get_experiment import get_experiment
from experiment import task

#general utils 
from utils.helpers import normalize_angle, read_file, get_landmark_mu, get_landmark_sigma, get_heading, gauss_2d_distance

#logging + plotting 
from utils.logger import Logger
from utils.animation import get_SLAM_animation
from utils.live_plotting import Plotting 


class MPC(): 
    def __init__(self, model, solver, planner, sol, params, obs0, obs_state, obs_belief, p, args, task, logger, live_plot = None):
        #model, planner and solver 
        self.model = model
        self.planner = planner  
        self.solver = solver 

        #model, task and algorithm parameters 
        self.params = params 
        self.dt = params['dt']
        
        #current solution of solver 
        self.sol = sol 
    
        #observability 
        self.obs0 = obs0 
        self.obs_state = obs_state
        self.obs_belief = obs_belief 

        #solver paraameters 
        self.p = p 
        self.args = args 

        #experiment, task and logging 
        self.experiment = get_experiment(params['experiment'])
        self.task = task 
        self.logger = logger
        self.live_plot = live_plot 
        
        #timesteps since which planner is disoriented 
        self.reorientation_ctr = 0 
        self.model_nudges = 0
        self.time_since_start = 0

        self.start_time = None 
        self.wait_time = 0

        self.curr_goal  = self.task.curr_goal
        self.goal_reached = False 

        self.Sigma_t = None 


    def update(self, x_true, mu_t, Sigma_t, mpc_iter, verbose = False):

        if not self.start_time: self.start_time = time.time() 
        self.Sigma_t = Sigma_t
        loop_time = time.time()

        # ___________________________ Initial Observation _______________________ #

        #first observation
        if mpc_iter == 0:
            for _ in range(10):
                z_t, observable = State_Observation(x_true, self.model, self.obs_belief, self.obs_state, self.params)
                mu_t, Sigma_t  = EKF_SLAM_Correction_Step(mu_t, Sigma_t, z_t, self.model)
                self.logger.log(self.dt, x_true, mu_t, Sigma_t, DM.zeros(2), DM.zeros(2), self.task, z_t)


                #log belief observation 
                z_t_belief, observable = State_Observation(DM(mu_t).full(), self.model, self.obs_belief, self.obs_state, self.params)
                self.logger.log_belief_measurement(z_t_belief)
                self.logger.log_planned_trajectory(- 100 * np.ones((3,15)))
                
            if self.live_plot is not None:
                    self.live_plot.update_plot(0, self.experiment, self.logger, None, self.params, self.model.nz).draw_figure(0.25)



        # ___________________________ Planning + Control + State Estimation _______________________ #
        #Belief Space Planning 
        self.sol, self.p, x_res, u_res = Belief_Space_Planning(self.planner, self.solver, self.args, self.p, self.sol, self.params)
        
        for j in range(self.params['control_horizon']):
            u_t = u_res[j,:] 

            #State Update and Observation 
            x_true, u_t_noisy = State_Update(x_true, u_t, self.model, self.params) 
            z_t, observable = State_Observation(x_true, self.model, self.obs_belief, self.obs_state, self.params)

            #log belief observation 
            z_t_belief, observable = State_Observation(DM(mu_t).full(), self.model, self.obs_belief, self.obs_state, self.params)
            self.logger.log_belief_measurement(z_t_belief)
            
            #Belief State Update 
            mu_t, Sigma_t  = EKF_SLAM_Prediction_Step(mu_t, Sigma_t, u_t, self.model, self.dt)    
            mu_t, Sigma_t  = EKF_SLAM_Correction_Step(mu_t, Sigma_t, z_t, self.model)

            # Logging + Plotting 
            self.logger.log(self.dt, x_true, mu_t, Sigma_t, u_t, u_t_noisy, self.task, z_t)
            self.logger.log_planned_trajectory(x_res.full()[j:,:3])
            
            #update iteration timers 
            self.wait_time += self.params['dt']
            mpc_iter += 1 
            self.time_since_start += 1

        #output to live plot 
        if self.live_plot is not None:
            self.live_plot.update_plot(mpc_iter, self.experiment, self.logger, x_res.full()[:,:3], self.params, self.model.nz).draw_figure(0.25)
            
          
        # _____________________________________ Re-orientation ________________________________________ #        
        if self.params['reorientation_enabled']:
            mpc_iter, self.reorientation_ctr, x_true, mu_t, Sigma_t = MPC_Reorientation(self.reorientation_ctr, mpc_iter,
                                                                                  x_true, mu_t, Sigma_t, 
                                                                                  self.model, self.params, observable, 
                                                                                  self.obs_state, self.obs_belief, 
                                                                                  self.task, self.live_plot, self.logger,
                                                                                  self.params['reorientation_tresh'])      
      
        # _______________________________________ Task ___________________________________________ #
        if self.task.curr_path == 'outbound':
            #if the agents true state reaches the true target the next goal is chosen
            if np.linalg.norm(x_true[:2]- np.array(get_landmark_mu(x_true, self.params['n_landmarks'] + self.task.curr_goal))) < self.task.goal_epsilon:
                if verbose: print('goal reached')
                self.task.set_next_goal(x_true, mu_t, self.obs_belief, self.obs_state)
                self.obs_state = self.task.obs_state
                self.obs_belief = self.task.obs_belief 
                self.planner.reinit = True
                self.wait_time = 0 
                self.goal_reached = True 

                #last-known rotation 
                self.task.start_rotation = copy.deepcopy(x_true[2])
                
        if self.task.curr_path == 'return path':
            
            #rotate landmarks (conflict codition only)
            if not self.task.rotated:
                for i in range(self.task.n_landmarks):
                    x_true[3+2*i:3+2*i+2] =  DM(rotate(x_true[-2:].full().reshape(2,), np.array(x_true[3+2*i:3+2*i+2]).reshape(2,), np.deg2rad(self.task.landmark_rotation)))
                self.task.rotated = True

            #wait / spin in chair 
            while self.task.t_wait < self.task.t_max: 
                self.model.obs0 = np.zeros(len(self.obs_belief)) #now we can't see anything 

                if verbose: print(self.task.t_wait)

                mpc_iter +=1 

                #Task Update (wait; rotate; move agent?)
                x_true, mu_t, Sigma_t, u_t, self.curr_goal = self.task.waiting_period(x_true, mu_t, Sigma_t, self.experiment, self.params, self.curr_goal, verbose)

                #State Update 
                x_true, u_t_noisy = State_Update(x_true, u_t, self.model, self.params) 
                mu_t, Sigma_t  = EKF_SLAM_Prediction_Step(mu_t, Sigma_t, u_t, self.model, self.dt)
                                
                z_t = {'z_t' : [], 'coords' : []} #only empty observation 
                self.logger.log(self.dt, x_true, mu_t, Sigma_t, u_t, u_t_noisy, self.task, z_t)
                self.logger.log_belief_measurement(z_t_belief)
                self.logger.log_planned_trajectory(- 100 * np.ones((3,15)))

                # live plotting 
                if self.live_plot:
                    self.live_plot.update_plot(mpc_iter, self.experiment, self.logger, x_res.full()[:,:3], self.params, self.model.nz).draw_figure(0.25)

                #advance time 
                self.task.t_wait += self.params['dt']
                self.time_since_start = 0 #reset the time since agent has started as to not appear static 
                self.wait_time = 0

            #if the agents belief state reaches the belief target on the return path the task is finished
            if np.linalg.norm(DM(mu_t[:2])- np.array(get_landmark_mu(DM(mu_t).full(), self.params['n_landmarks'] + self.task.curr_goal))) < self.task.goal_epsilon:
                #even if you reached the goal wait for a bit (to correct some last minute errors ?)
                if (self.wait_time >= 3.0):
                    self.task.task_finished = True 
                    if verbose: print('task finished')

        if self.goal_reached: 
            if verbose: print('waiting for next goal')

        if self.goal_reached and (self.wait_time >= self.params['max_wait_time']):
            if verbose: print('next goal')
            self.curr_goal  = self.task.curr_goal
            self.planner.reinit = True 

            self.goal_reached = False 
            self.wait_time = 0

        # _________________________________ Planner Re-Initialization __________________________________ #
        self.p, self.args, x_res, u_res = self.planner.reinitialize_planner(x_res, u_res, self.p, x_true, mu_t, Sigma_t, self.logger.us_planned[-1], self.curr_goal, self.obs_belief, self.params)
        self.model.obs0 = self.obs_belief 
                    
        return x_true, mu_t, Sigma_t, mpc_iter


