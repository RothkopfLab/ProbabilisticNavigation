import numpy as np 
import pandas as pd
import copy  




from scipy.special import i0
from scipy.special import i1

from analysis.zhao2015.utils import * 
from analysis.zhao2015.circstat import * 



def compute_search_distances(normalized_data):
    pooled_centroid = np.mean(normalized_data, axis = 0)
    search_distances = np.linalg.norm(normalized_data - pooled_centroid, axis = 1)
    return search_distances 

def get_heading_errors(data):
    return np.array([np.arctan2(point[0], point[1]) for point in data])

def subset_by_iqr(df, column, whisker_width=1.5):
    # Calculate Q1, Q2 and IQR
    q1 = df[column].quantile(0.25)                 
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    # Apply filter with respect to IQR, including optional whiskers
    filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)
    return df.loc[filter]    

def remove_outliers_heading_direction(dataset):
    dfs = []
    dataset = copy.deepcopy(dataset)
    dataset['condition'] = dataset['condition'] + '_' + dataset['landmark_rotation'].astype(str)
    
    for vp, vp_data in dataset.groupby('VPCode'):

        for condition, condition_data in vp_data.groupby('condition'):
            condition_data = copy.deepcopy(condition_data)
            condition_data['euclidean_error'] = compute_search_distances(condition_data[['x', 'y']].values)
            condition_data['heading_error'] = get_heading_errors(condition_data[['x', 'y']].values)
            condition_data = subset_by_iqr(condition_data, 'heading_error', whisker_width=3)


            dfs.append(condition_data)
    
    dfs = pd.concat(dfs)
    dfs['condition'].apply(lambda x : x.split('_')[0])
    dfs['condition'].apply(lambda x : x.split('_')[0])
        
    return dfs

'''
Participants performed 40 homing trials in each of the nine conditions, 
with trials for each condition grouped together in the same block, for a total of 360 trials
'''
#zhao and warren requires a somewhat different normalization
def normalize_data_by_conflict_condition(data, return_dataframe=False):
    import math
    
    start =  np.zeros(data[['targetx', 'targetz']].shape) #is the final post at 0.0 ? yes!
    target = data[['targetx', 'targetz']].values
    
    '''
    Compute response direction
    '''
    
    #what is the angle between the start and the end location? (use this as the north direction)
    angle = np.rad2deg(np.arctan2(start[:,1]- target[:,1],  start[:,0] - target[:,0])) 

    rotated_targets = []
    for s,t,a in zip(start,target,angle):
        '''
        Response directions were standardized so that the correct homing direction was always at 0° (“north”), 
        with positive values in the direction of landmark shift.
        '''
        
        #rotate target locations upwards 
        rotated_target = rotate(s, t, -np.deg2rad(a) - np.pi/2)
        rotated_targets.append(rotated_target)

    rotated_targets = np.array(rotated_targets)
    #print(rotated_targets.round(4)) # we learn that the target is 3.602 meters away (path-integration prediction)
    
    
    #now rotate all the responses into a common north frame (they are all the same distance from 0.0)
    responses = data[['respx', 'respz']].values
    rotated_responses = []

    for r, s, a in zip(responses, start, angle):
        rotated_response = rotate(s, r, (-np.deg2rad(a) - np.pi/2))
        rotated_responses.append(rotated_response)

    rotated_responses = np.array(rotated_responses)
    
    if return_dataframe:
        df = pd.DataFrame([])
        df['x'] = rotated_responses[:,0]
        df['y'] = rotated_responses[:,1]
        df['condition'] = data.condition.values
        df['VPCode'] = data.VPCode.values
        df['landmark_rotation'] = data.landmark_rotation.values
        return df 
    
    return rotated_responses

def normalize_data_empirical(dataset_empirical):
    #Empirical 
    normalized_data_empirical = []
    for condition, data in dataset_empirical.groupby('condition'):
        #flip x coordinates for conflict condition
        if condition.split('_')[0] == 'conflict':
            normalized = normalize_data_by_conflict_condition(data, return_dataframe=True)
            if condition.split('_')[3] == 'left':
                normalized['x'] = - normalized['x']
        else:
            normalized = normalize_data_by_conflict_condition(data, return_dataframe=True)

        normalized_data_empirical.append(copy.deepcopy(normalized))

    normalized_data_empirical = pd.concat(normalized_data_empirical)
    
    return normalized_data_empirical


import scipy.stats as ss 
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html#scipy.special.jv
def transform_circular_sd(k):
    #sigma = np.sqrt(2*(1 - i1(k) / i0(k)))

    #uses an algorithm to determine the std 
    sigma = ss.vonmises.std(k)
    
    return sigma 


def get_heading_errors(data):
    return np.array([np.arctan2(point[0], point[1]) for point in data])


def fit_von_mises(data):
    heading_errors = get_heading_errors(data[['x', 'y']].values)
    k, theta, __ = ss.vonmises.fit(heading_errors, fscale = 1)

    #circ_sd = np.rad2deg(transform_circular_sd(k))
    circ_sd = np.rad2deg(std(heading_errors))
    
    return k, theta, circ_sd

def fit_von_mises_per_condition_df(normalized_data):
    dfs = []
    for condition, data in normalized_data.groupby('condition'):
        for rotation, data in data.groupby('landmark_rotation'):
            k, theta, circ_sd = fit_von_mises(data)
            df_temp = {'condition' : condition,
                    'k' : k, 
                    'theta' : theta,
                    'theta_deg' : np.rad2deg(theta),
                    'circ_sd' : circ_sd, 
                    'landmark_rotation' : rotation}
            
            dfs.append(pd.DataFrame(df_temp, index = [0]))
        
    return pd.concat(dfs).reset_index()

def fit_von_mises_per_condition_dict(normalized_data):
    fit_per_condition = {}
    for condition, data in normalized_data.groupby('condition'):
        for rotation, data in data.groupby('landmark_rotation'):
            k, theta, circ_sd = fit_von_mises(data)
            if condition == 'conflict':
                condition = 'conflict' + '_' + str(rotation)

            fit_per_condition[condition] = {'k' : k, 
                                            'theta' : theta,
                                            'theta_deg' : np.rad2deg(theta),
                                            'circ_sd' : circ_sd,
                                            'landmark_rotation' : rotation}
    return fit_per_condition