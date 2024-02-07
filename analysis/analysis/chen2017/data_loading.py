import copy 
import numpy as np 
import pandas as pd 

from analysis.chen2017.utils import * 

trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}

################################################################################################
                                    ### Data Loading  ###
################################################################################################

#empirical data 
def load_chen_data(path, file, environment_choice, rezero = False):
    dataset = pd.read_csv(path + file)
    dataset.condition = dataset.condition.apply(lambda x : trial_types[x])
    #dataset = dataset[dataset.VPCode == dataset.VPCode.unique()[3]]

    environment = {1 : 'rich', 2 : 'poor'}

    split_data = {}
    for env, data in dataset.groupby('environment'):
        split_data[environment[env]] = data

    env = environment_choice
    data = split_data[env]

    data.loc[:,'targetx'] += data['jitterx'] 
    data.loc[:,'targetz'] += data['jitterz']

    data.loc[:,'target_loc'] = np.round(data['targetx'] - data['jitterx'], 3)
    data.loc[:,'side'] = np.sign(data['target_loc'])

    data.loc[:,'targetx'] -= data['jitterx'] 
    data.loc[:,'targetz'] -= data['jitterz']

    #move the world up 
    if rezero:
        data.loc[:('respz', 'targetz', 'startz')] += 0.75
    
    return data 


#simulated data 
def process_simulated_data(simulated_data, outlier_deletion = False, rezero = False) :
    simulated_data = simulated_data[simulated_data.task_finished.fillna(False)]
    simulated_data.condition = simulated_data.condition.apply(lambda x : trial_types[x])
    normalized_data = normalize_data_by_condition(simulated_data, normalize_by = 'chen')

    #outlier deletion 
    if outlier_deletion: 
        response_mean = np.mean(normalized_data[['x','y']].values, axis = 1)
        normalized_data.loc[:,'euclidean_distance']  = np.sqrt((normalized_data['x'] - response_mean)**2 + (normalized_data['y'] - response_mean)**2)
        normalized_data = subset_by_iqr(normalized_data, 'euclidean_distance', whisker_width = 3)
        
    simulated_data.loc[:,'target_loc'] = np.round(simulated_data['targetx'] - simulated_data['jitterx'], 3)
    simulated_data.loc[:,'side'] = np.sign(simulated_data['target_loc'])

    #move the world up
    if rezero:
        simulated_data[['respz', 'targetz', 'startz']] += 0.75
    
    return simulated_data, normalized_data





