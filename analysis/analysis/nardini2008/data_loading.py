import pandas as pd 
import os 

trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}

def load_nardini_data(path):
    files = [file for file in os.listdir(path) if 'adult' in file]
    dataset = []

    for file in files:
        condition = file.split('_')[-1][:-4]
        data = pd.read_csv(path + file)
        data = data[['x','y']] / 100 #go from cm to m 
        data['condition'] = condition

        dataset.append(data)

    dataset = pd.concat(dataset)
    
    return dataset 

def load_nardini_data_simulated(file):
    simulated_data =  pd.read_csv('../ray_tune_runs/' + file)
    simulated_data = simulated_data[simulated_data.task_finished.fillna(False)]
    simulated_data.condition = simulated_data.condition.apply(lambda x : trial_types[x])
    
    return simulated_data