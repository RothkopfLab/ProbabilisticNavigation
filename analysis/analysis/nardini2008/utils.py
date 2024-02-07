#Preprocessing 
import math 
import sys
import os
import copy
sys.path.append("..") 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}


def euclidean_dist(df):
    return np.linalg.norm(df[['respx', 'respz']].values - df[['targetx', 'targetz']].values,
                   axis=1)


def subset_by_iqr(df, column, q = None, whisker_width=1.5):
    # Calculate Q1, Q2 and IQR
    if q is None:
        q1 = df[column].quantile(0.25)                 
        q3 = df[column].quantile(0.75)
    else:
        q1 = q[0]
        q3 = q[1]
        
    iqr = q3 - q1
    
    # Apply filter with respect to IQR, including optional whiskers
    #filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)
    
    #Outliers were defined as responses whose distance from their respective response centroids exceeded the 3rd quartile by 3 times the interquartile range
    filter = (df[column] <= q3 + whisker_width*iqr)
    return df.loc[filter]    



def rotate(origin, point, angle):    
    if point.shape[0] == 3:
        ox, oy     =  origin
        px, py, ID = point
    else:
        ox, oy = origin
        px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy


def normalize_endpoints(data, response = ('respx', 'respz'), normalize = 'chen'):
    respx_prime = data[response[0]] - data['targetx']
    respz_prime = data[response[1]] - data['targetz']
    angle1 = np.rad2deg(np.arctan2(respz_prime - 0, respx_prime - 0))
    
    if normalize == 'chen':
        ang_targ = np.rad2deg(np.arctan2(data.targetz - 0, data.targetx - 0))
        angle = np.deg2rad(angle1 + 90 - ang_targ)
    else:
        angle = np.deg2rad(angle1)

    dist = np.sqrt(respx_prime**2 + respz_prime**2)
    
    outputx = dist * np.cos(angle)
    outputz = dist * np.sin(angle)
        
    points = np.array([outputx.values, outputz.values]).T
    
    return points 


def normalize_endpoints_nardini(data, response = ('respx', 'respz')):
    rotated_points = []

    for i, point in data.iterrows():
        #center into zero
        start = np.array([2.5, 3])
        target = point[['targetx', 'targetz']] - start 
        response = point[['respx', 'respz']] - start 
        start = start - start

        respx_prime = start[0] - target[0]
        respz_prime = start[1] - target[1]

        #distance and angle 
        dist = np.sqrt(respx_prime**2 + respz_prime**2)
        angle = np.arctan2(respz_prime, respx_prime)

        #foreach point do the rotation to normalize properly 
        rotated_point = np.array(rotate(start, response, -angle - np.pi/2))
        rotated_point[1] = rotated_point[1] - dist #subtract the distance 

        rotated_points.append(rotated_point)
                                
    return np.array(rotated_points)


def normalize_data_by_condition(simulated_data, plot = False, normalize_by = 'chen'):
    endpoints_by_condition = []
    for cond, data in simulated_data.groupby('condition'):
        if normalize_by == 'chen':
            normalized_points = normalize_endpoints(data, response = ('respx', 'respz'), normalize = normalize_by)
        else:
            normalized_points = normalize_endpoints_nardini(data, response = ('respx', 'respz'))
            
        normalized_data = pd.DataFrame({'condition' : [cond] * normalized_points.shape[0], 
                                        'VPCode' : data.VPCode,
                                        'x' : normalized_points[:,0], 
                                        'y' : normalized_points[:,1] + 0.15}) 

        pooled_centroid = np.mean((normalized_points[:,0], normalized_points[:,1]), axis = 1)
        endpoints_by_condition.append(normalized_data)

    normalized_data = pd.concat(endpoints_by_condition)
    
    return normalized_data

def extract_mean_response_locations(normalized_data):
    mean_locations = {}
    for condition, data in normalized_data.groupby('condition'):
        mean_locations[condition] = np.mean(data[['x','y']].values, axis = 0)
    return mean_locations

def compute_search_distances(normalized_data):
    pooled_centroid = np.mean(normalized_data, axis = 0)
    search_distances = np.linalg.norm(normalized_data - pooled_centroid, axis = 1)
    return search_distances 

def compute_search_distance_by_condition(normalized_data):
    dfs = []
    for condition, data in normalized_data.groupby('condition'):
        d = compute_search_distances(data[['x','y']].values)
        data.loc[:,'euclidean_distance'] = d    
        dfs.append(copy.deepcopy(data))
        
    return pd.concat(dfs)

def compute_SDs(normalized_data, return_dict = False):
    SDs = {}
    for condition, data in normalized_data.groupby('condition'):
        d = data['euclidean_distance'].values 
        SDs[condition] = np.mean(d)
        
    if return_dict:
        return SDs 

    return normalized_data.groupby('condition').apply(np.mean)['euclidean_distance']

def compute_variance(normalized_data):
    var = {}
    for condition, data in normalized_data.groupby('condition'):
        d = data['euclidean_distance'].values 
        var[condition] = np.mean(d)**2
        
    return var

def compute_rotated_target_locations(simulated_data):
    target_locations = np.unique(simulated_data[['targetx', 'targetz']].values, axis = 0)
    rotated_target_locations = []

    dfs = []

    for target in target_locations:
        plt.scatter(target[0], target[1], marker = 'D', color = 'grey')

        start = np.array([2.5, 3.0])


        target_rotated = rotate(start, target, np.deg2rad(-15))
        plt.scatter(target_rotated[0], target_rotated[1], marker = '+', color = 'lightgrey')

        rotated_target_locations.append(target_rotated)

        df_temp = {'startx' : start[0],
                   'startz' : start[1], 
                   'respx'  : target_rotated[0],
                   'respz'  : target_rotated[1], 
                   'targetx' : target[0], 
                   'targetz' : target[1]}

        dfs.append(pd.DataFrame(df_temp, index = [0]))

    target_data = pd.concat(dfs)
    
    return target_data