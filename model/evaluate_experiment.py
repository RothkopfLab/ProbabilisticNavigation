import os 
import pickle 
import copy 
import json

import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import normalize_angle, read_file, get_landmark_mu, get_landmark_sigma, get_heading

import argparse

def evaluate_experiment(run_folder):
    all_run_logs = []
    all_runs_csv = []
    all_run_params = []
    all_run_times = []
    all_run_names = []

    u_tr_id = 0 
    not_finished = 0 
    finished = 0
    for run in os.listdir(run_folder):
        run_path = run_folder + '/' + run
        if os.path.isdir(run_path):
            
            try:
                progress = pd.read_csv(run_path + '/progress.csv')
            except:
                #delete the run folder if i cannot open it 
                #try:
                #    print('deleting file')
                #    print(run_path)
                #    shutil.rmtree(run_path)
                #except:
                #    pass

                continue
                    
            if not progress.iloc[-1].task_finished:
                #print(progress.iloc[-1].time_total_s)
                not_finished += 1
                
                #try:
                #    print('deleting file')
                #    print(run_path)
                #    shutil.rmtree(run_path)
                #except:
                #    pass

                print('task not finished: ' + str(not_finished / (finished + not_finished)))
            else:
                #print('task finished')
                print(progress.iloc[-1].time_total_s)
                finished += 1
            
            try:
                run_csv = pd.read_csv(run_path + '/curr_trial_log.csv')
                run_csv['unique_trajectory_id'] = u_tr_id
                run_csv['task_finished'] = progress.iloc[-1].task_finished

                run_protocol = pd.read_csv(run_path + '/progress.csv')
                run_time = run_protocol.iloc[-1].time_total_s

                run_log = pickle.load(open(run_path + '/logger.p', "rb" ))
                run_params = pickle.load(open(run_path + '/params.p', 'rb'))
                #print(run_params['landmark_locations'])
            except:
                print('error opening') 
                continue
        
            all_runs_csv.append(run_csv)
            all_run_params.append(run_params)
            all_run_logs.append(run_log)
            all_run_times.append(run_time)
            all_run_names.append(run)


            u_tr_id +=1 
   
    print('runs aborted: ' +  str(not_finished))
    print('runs finished: ' + str(finished))
    print('percentage: ' + str(not_finished / (not_finished + finished)))
    print('run parameters')
    #print(run_params)

    
            
    from parameters import get_params
    dfs = pd.concat(all_runs_csv)

    
    premature_stops = 0 
    coords = []
    data = []
    trajectories = []
    for u_t_id, tr in dfs.groupby('unique_trajectory_id'):
        #params = get_params()
        params = all_run_params[u_t_id]

        trajectories.append(copy.deepcopy(tr))

        if tr[tr.segment == 0].shape[0] > 0:
            start = tr[tr.segment == 0].iloc[0] #not exactly sure whether this encodes the proper timestep
            end = tr[tr.segment == 0].iloc[-1]
            
            #todo add jitter to targetx anf targety 
            tmp  =  {'unique_trajectory_id' : u_t_id,
                    'condition' : tr.condition[0],
                    'VPCode' : params['VPCode'],
                    'landmark_rotation' : tr.landmark_rotation[0],
                    'respx': end['x'], 
                    'respz': end['z'], 
                    'targetx' : params['route'][1][0], 
                    'targetz' : params['route'][1][1],
                    'trajectory_id' : params['trajectory_id'],
                    'target' : params['trajectory_id'][0],
                    'jitterx' : params['target_jitter'][0],
                    'jitterz' : params['target_jitter'][1],
                    'startx'  : start['x'],
                    'startz'  : start['z'],
                    'belief_respx' : get_landmark_mu(all_run_logs[u_t_id]['mus'][-1], params['n_landmarks'])[0],
                    'belief_respz' : get_landmark_mu(all_run_logs[u_t_id]['mus'][-1], params['n_landmarks'])[1],
                    'endidx' : (tr['segment'] == 0).idxmax(),
                    'task_finished' : tr['task_finished'][0]}

            #print(tr['task_finished'][0])
            
            df = pd.DataFrame(tmp, index = [u_t_id]) 
            
            target = np.array([tmp['targetx'], tmp['targetz']]).T
            response = np.array([tmp['respx'], tmp['respz']]).T
            
            ##filter out those runs which where aborted
            #if np.linalg.norm(target-response) < 0.25:
            data.append(df)
        else:
            #print('stopped prematurely')
            premature_stops += 1
            
    simulated_data_endpoints = pd.concat(data)
    simulated_data_trajectories = pd.concat(trajectories)
    run_params = params 
    
    run_time_names = list(zip(all_run_names, all_run_times))
    simulation_times = pd.DataFrame(run_time_names, columns = ['name', 'time'])
    

    print('premature stops: ' + str(premature_stops))

    return simulated_data_endpoints, simulated_data_trajectories, run_params, simulation_times  


if __name__ == '__main__':

    run_folder = 'ray_tune_runs/all_runs2' 

    parser = argparse.ArgumentParser(description = 'generate simulated data csv')
    
    parser.add_argument('-input', nargs= '?', type = str,   help = 'input file')
    parser.add_argument('-output', nargs= '?', type = str,  help = 'output file')

    args = parser.parse_args()
    
    try:
        simulated_data_endpoints, simulated_data_trajectories, params, simulation_times = evaluate_experiment(args.input)
    except:
        print('evaluate experiment: aborting')

    simulated_data_endpoints.to_csv(args.output + '_endpoints.csv')
    #simulated_data_trajectories.to_csv(args.output + '_trajectories.csv')
    simulation_times.to_csv(args.output + '_times.csv')
    
    with open(args.output + "_params.txt", "w") as outfile:
        for key in params.keys():
            outfile.write(key + ': '+ str(params[key]) + '\n')
