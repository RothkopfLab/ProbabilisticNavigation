import string
import time
import pickle
import copy

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
from mpc import * 


from experiment.get_experiment import get_experiment

#logging + plotting 
from utils.logger import Logger, save_trajectory
from utils.live_plotting import Plotting, plot_intermediate_step

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter

from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.stopper import Stopper

import argparse


os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = str(int(1e10))

#arg parse helper methods 
def csv_list(string):
   return np.array([float(item) for item in string.split(',')])

def csv_list_string(string):
   return [str(item) for item in string.split(',')]

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'



#saving and loading of pickles 
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol = 4)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)   

#custom stopping method for runs that take longer than x seconds 
class CustomStopper(Stopper):
    def __init__(self, deadline=1000):
        self._deadline = deadline

    def __call__(self, trial_id, result):
        return ((result["time_total_s"] > self._deadline))
    
    def stop_all(self):
        return False


#interindividual variability of virtual participants 
def add_interindividual_variability(params, variability):
    for var in variability.keys():
        curr_var = variability[var]
        
        params[var] += curr_var
        
        #dont allow any negative values...
        params[var] = np.abs(params[var])
        
    return params 

def simulate_trial(config, data = None, checkpoint_dir = None):
    conditions = {'landmark': 1, 'self-motion' : 2, 'combined' : 3, 'conflict' : 4, 'disorientation' : 5, 'ground' : 6}
    trial_settings = None 

    #load trial settings from participant
    if config['individual_participant_file']:
        #get path pointing to the participants folder for a particular experiment 
        trial_settings = config['trial'].split('_')  #format: #VP07_combined_1-6-8
        print(trial_settings)
        config['trajectory_id'] = trial_settings[2] #change trajectory_id 
        print(config['trajectory_id'])
        config['landmark_shift_dir'] = np.sign(float(trial_settings[3]))
        config['landmark_rotation'] = float(trial_settings[3])

    params = get_params(experiment_name = config['experiment'], 
                        start_position = config['start_position'], 
                        trajectory_id = config['trajectory_id'], 
                        n_landmarks = config['n_landmarks'], 
                        landmark_shift_dir = config['landmark_shift_dir'],  
                        target_jitter = config['enable_target_jitter'])

    #change those parameters based on those from the tune config 
    for key in config.keys():
        if key in params['cost_function_weights'].keys():
             params['cost_function_weights'][key] = config[key]
        else:
            if config[key] is not None:
                params[key] = config[key]

    if params['individual_participant_file']:
        # load participants variability and apply individual VP variability to the parameters 
        path = data['path'] #experiment path 
        VP = trial_settings[0] #individual subject ID 
        params['VPCode'] = VP
        params['condition'] = trial_settings[1] #set the condition 

    ################################################################################################
                                ### RANDOM COST PARAMETERS ###
    ################################################################################################        
    #parameter_set = pd.read_csv('/home/iv55otop/00-Projects/probabilistic-navigation/' + 'random_parameters_cost.csv').iloc[int(config['parameter_set'])]

    #params['cost_function_weights'] = {
    #                            'w_running_control': parameter_set.w1,
    #                            'w_running_control_smoothness' : 0.0, 
    #                            'w_running_position_uncertainty':  parameter_set.w2, 
    #                            'w_running_map_uncertainty' : parameter_set.w3, 
    #                            'w_running_state':  parameter_set.w4, 
    #                            'w_final_state': parameter_set.w5, 
    #                            'w_final_uncertainty': 0.0}
    
    ################################################################################################
                                ### RANDOM COST PARAMETERS ###
    ################################################################################################     
        

    ################################################################################################
                                ### RANDOM NOISE PARAMETERS ###
    ################################################################################################     
    #parameter_set = pd.read_csv('/home/iv55otop/00-Projects/probabilistic-navigation/' + 'random_parameters_noise.csv').iloc[int(config['parameter_set'])]

    #random_parameters = parameter_set
    #params['observation_cov'] = [random_parameters['obs_0'], random_parameters['obs_1'], random_parameters['obs_2'], random_parameters['obs_3']]
    #params['motion_cov'] = [random_parameters['mo_0'], random_parameters['mo_1'], random_parameters['mo_2'], random_parameters['mo_3']]
    #params['representation_cov'] = random_parameters['re']
    ################################################################################################
                                ### RANDOM NOISE PARAMETERS ###
    ################################################################################################   


    model_type = params['model_type']

    if model_type == 'Zero Model':
        params['observation_cov'] = 4 * [1e-20]
        params['motion_cov'] = 4 * [1e-20]
        params['representation_cov'] = 1e-20
    elif model_type == 'Perceptual':
        params['motion_cov'] = 4 * [1e-20]
        params['representation_cov'] = 1e-20
    elif model_type == 'Motor':
        params['observation_cov'] = 4 * [1e-20]
        params['representation_cov'] = 1e-20
    elif model_type == 'Motor & Perceptual':
        params['representation_cov'] = 1e-20
    elif model_type == 'Representation':
        params['observation_cov'] = 4 * [1e-20]
        params['motion_cov'] = 4 * [1e-20]
    elif model_type == 'Perceptual & Representation':
        params['motion_cov'] = 4 * [1e-20]
    elif model_type == 'Motor & Representation':
        params['observation_cov'] = 4 * [1e-20]
    elif model_type == 'Full Model':
        pass 


    #subjective uncertainties
    params['motion_cov_subj'] = params['motion_cov']
    params['observation_cov_subj'] = params['observation_cov']
    params['representation_cov_subj'] = params['representation_cov']

    with open('params.p', 'wb') as handle:
        pickle.dump(params, handle)

    #create a model of the agent (TODO obs0?)
    model = Model(n_landmarks = len(params['landmark_locations'] + params['pickup_locations'])//2,
                  motion_limits = params['motion_limits'],
                  R_diag = params['observation_cov_subj'], 
                  Q_diag = params['motion_cov_subj'],
                  V = params['representation_cov_subj'], 
                  R_diag_generative  = params['observation_cov'],
                  Q_diag_generative = params['motion_cov'],
                  V_generative = params['representation_cov'],
                  representation_type=params['representation_noise_type'], 
                  fov  = params['fov'])

    #create planner 
    planner = BeliefSpacePlanner(model, 
                                 N = params['planning_horizon'],
                                 verbose = False)

    #create solver instance 
    solver, opts = planner.create_solver(solver = 'ma57', 
                                         BFGS = True, 
                                         warm_start = True)



    #init state and planner params 
    x_true, mu_t, Sigma_t, obs0, obs_state, obs_belief, p, args, task, logger  = init_state(params, model, planner, solver)

    #live plotting 
    live_plot = None

    #init model predictive control 
    sol = None 
    mpc = MPC(model, solver, planner, sol, params, obs0, obs_state, obs_belief, p, args, task, logger, live_plot)
    mpc_iter = 0
    
    rotated = False 

    #agent position state 
    #randomly perturb starting location 
    x_true[:3] = np.random.multivariate_normal(x_true[:3].full().reshape(3,), np.diag([0.2, 0.2, np.deg2rad(10)])**2)
    mu_t[:3] = x_true[:3]

    while True:
        x_true, mu_t, Sigma_t, mpc_iter = mpc.update(x_true, mu_t, Sigma_t, mpc_iter)
        #x_res, u_res = planner.obtain_solution(mpc.sol)

        #live plotting of intermediate steps at every n-th iteration 
        if ((mpc_iter//params['control_horizon']) % config['logging_step'] == 0) or mpc.task.task_finished:    
            #get dataframe from current trajectory 
            df = save_trajectory(mpc.logger.xs, mpc.logger.segments, mpc.logger.ts, params['trajectory_id'], 0, p, (params['target_jitter'][0],params['target_jitter'][1]))
            df['experiment'] = params['experiment']
            df['VPCode'] = params['VPCode']
            df['condition'] = conditions[params['condition']]
            df['landmark_rotation'] = params['landmark_rotation']
            df.to_csv('curr_trial_log.csv')
        
            #dump logger output 
            file = open( "logger.p", "wb" )
            pickle.dump(mpc.logger.get_dict(), file)
            file.close()

            #plot intermediate step 
            plot_intermediate_step(experiment = experiment, df = df, logger = mpc.logger, params = params, fname = params['trajectory_id'] + '_curr_trial_progress.png')
        

        tune.report(mpc_iter = mpc_iter, mean_loss = np.random.uniform(0.0, 0.05), task_finished = mpc.task.task_finished)
        time.sleep(0.01)

        if mpc.task.task_finished:
            tune.report(mpc_iter = mpc_iter, mean_loss = 0, task_finished = mpc.task.task_finished)        
            return
        elif mpc_iter > params['max_iter']: #abort 
            return 


if __name__ == "__main__":
    from parameters import get_params 

    ray.shutdown()
    
    from ray.cluster_utils import Cluster

    # Starts a head-node for the cluster.
    cluster = Cluster(
    initialize_head=True,
    head_node_args={
        "num_cpus": 96,
    })

    ray.init(address=cluster.address, log_to_driver = False)

    params = get_params()

    #arg parse 
    parser = argparse.ArgumentParser(description = 'pass model, task or algorithm arguments')

    #experiment 
    parser.add_argument('-experiment', type = str, default = 'chen2017')
    parser.add_argument('-individual_participant_file', type = bool, default = True, help = 'use individual participants')
    parser.add_argument('-individual_participant_file_type', type = str, default = '', help = 'e.g. _proximal_base')

    #task options 
    parser.add_argument('-n_landmarks', type = int, default = params['n_landmarks'], help = 'number of landmarks')
    parser.add_argument('-start_position', type = csv_list, default = None, help ='start position')
    parser.add_argument('-body_rotation', type = bool, default = False, help = 'enable body rotation at the end')
    parser.add_argument('-stable_landmarks', type = boolean_string, default = 'True', help = 'stable landmarks?')
    parser.add_argument('-enable_target_jitter', type = boolean_string, default = 'True', help = 'shift targets or landmarks in case of zhao2015a')
    parser.add_argument('-conditions', type = csv_list_string, default = ['landmark', 'self-motion', 'combined', 'conflict'], help = 'conditions')
    parser.add_argument('-target_jitter_enabled', type = boolean_string, default = 'True', help = 'target jitter')
    parser.add_argument('-conflicts', type = csv_list, default = None, help = 'conflicts')

    #cost function & behavior 
    for cost_function_weight in params['cost_function_weights'].keys():
        parser.add_argument('-' + cost_function_weight, nargs= '?', type = float, default = params['cost_function_weights'][cost_function_weight], help = 'weight of cost function')

    parser.add_argument('-reorientation_enabled', type = boolean_string, default = params['reorientation_enabled'], help = 'enable reorientation behavior')    
    parser.add_argument('-reorient', type = boolean_string, default = 'True', help = 'enable reorientation behavior')

    #state-estimation parameters B
    parser.add_argument('-initial_positional_uncertainty', type = csv_list, nargs= '?', default= params['initial_positional_uncertainty'])
    parser.add_argument('-motion_cov', type = csv_list,  nargs = '?', default = params['motion_cov'], help = ['(sigma_r_min, sigma_r_max, sigma_phi_min, sigma_phi_max)'])
    parser.add_argument('-observation_cov', type = csv_list, nargs = '?', default = params['observation_cov'], help = ['(sigma_r_min, sigma_r_max, sigma_phi_min, sigma_phi_max)'])
    parser.add_argument('-representation_cov', type = float, nargs = '?', default = params['representation_cov'], help = 'noise in representation')
    
    #enabling and disabling in generative model 
    parser.add_argument('-observation_noise_enabled', type = boolean_string, nargs = '?', default = params['observation_noise_enabled'])
    parser.add_argument('-motion_noise_enabled', type = boolean_string, nargs = '?', default = params['motion_noise_enabled'])
    parser.add_argument('-representation_noise_enabled', type = boolean_string, nargs = '?', default = params['motion_noise_enabled'])
    parser.add_argument('-representation_noise_type', type = str, default = params['representation_noise_type'], help = '?')
    parser.add_argument('-representation_random_walk', type = boolean_string, default = params['representation_random_walk'], help = '?')

    #algorithm parameters 
    parser.add_argument('-planning_horizon', nargs= '?', type=int, default = params['planning_horizon'], help='number of steps for planning horizon')
    parser.add_argument('-control_horizon', nargs = '?', type=int, default = params['control_horizon'], help = 'number of actions u_t executed after planning')
    parser.add_argument('-fov', type = int, nargs= '?',  help = 'field of view') #no default argument here 
    parser.add_argument('-dt', type = float, nargs= '?', default = params['dt'], help = 'timestep') 

    #run options 
    parser.add_argument('-output_folder', nargs = '?', type=str, default = 'test_run')
    parser.add_argument('-VPCode', nargs = '?', type=str, default = 'None')
    parser.add_argument('-run_deadline', nargs = '?', type=float, default = 2000)
    parser.add_argument('-max_iter', nargs = '?', type=int, default = 1000)
    parser.add_argument('-n_samples', nargs = '?', type=int, default = 1500)
    parser.add_argument('-logging_step', nargs = '?', type=int, default = 10)

    parser.add_argument('-parameter_set', nargs = '?', type = int, default = 0)
    parser.add_argument('-model_type', nargs = '?', type = str, default = 'full')

    args = parser.parse_args()

    #imitialize config 
    experiment_name = getattr(args, 'experiment')
    
    experiment = get_experiment(experiment_name)
    
    print(getattr(args, 'conditions'))

    if getattr(args, 'conflicts') is not None:
        experiment.rotations = getattr(args, 'conflicts')


    run_cnfg = {'trajectory_id' : tune.choice(experiment.trajectories),
                'condition' : tune.choice(getattr(args, 'conditions')),
                'landmark_rotation'  : tune.choice(experiment.rotations),
                'landmark_shift_dir' : tune.choice(experiment.landmark_shift_dir)}

    n_samples = getattr(args, 'n_samples')
    parameter_set = getattr(args, 'parameter_set')

    individual_participant_file_type = getattr(args, 'individual_participant_file_type')
    #generate virtual participants for the experiments ... 
    if getattr(args, 'individual_participant_file') is not None:
        #obtain virtual participants and trial structure 
        # if this is not just an empty string thing 
        individual_participant_file_type = individual_participant_file_type.strip()
        path = '/home/iv55otop/00-Projects/probabilistic-navigation/experiment/' + experiment_name.split('-')[0] + '/simulated_participants/'
        all_trials = load_obj(path + 'all_trials' + individual_participant_file_type) #virtual trials (~ balanced)

        params['path'] = path 
        run_cnfg = {'trial' : tune.grid_search(all_trials), #grid search takes each trial just once xxi
                    'trajectory_id' : '1-4-8',
                    'parameter_set' : parameter_set,
                    'landmark_rotation'  : tune.choice(experiment.rotations),
                    'landmark_shift_dir' : tune.choice(experiment.landmark_shift_dir)}

        #how often the grid search is repeated
        n_samples = 1

    #print(args.keys())

    for arg in vars(args):
        run_cnfg[arg] =  getattr(args, arg)
    
    analysis = tune.run(
                        tune.with_parameters(simulate_trial, data=params),
                        search_alg = BasicVariantGenerator(),
                        local_dir = '/work/scratch/iv55otop/ray_tune_runs',
                        name = run_cnfg['output_folder'],
                        config = run_cnfg, 
                        num_samples = n_samples,
                        verbose = 0, 
                        log_to_file = False,
                        reuse_actors = True, 
                        checkpoint_freq=0, 
                        resources_per_trial = {'cpu': 1},
                        stop = CustomStopper(deadline = run_cnfg['run_deadline']),
                        )
