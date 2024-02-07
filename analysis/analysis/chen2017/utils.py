import copy 
import os 
import math 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}

################################################################################################
                                    ### Utils  ###
################################################################################################

def rotate(origin, point, angle):
    import math     
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
                                        'y' : normalized_points[:,1]})

        pooled_centroid = np.mean((normalized_points[:,0], normalized_points[:,1]), axis = 1)
       
        endpoints_by_condition.append(normalized_data)

    normalized_data = pd.concat(endpoints_by_condition)
    
    return normalized_data

def normalize_by_post_location(dataset, target_loc, flip_conflict = False):
    dfs = []

    for target, data in dataset.groupby(['target_loc']):
        #normalized_by_condition 
        normalized_data = normalize_data_by_condition(data, normalize_by = 'chen')
        normalized_data['target_loc'] = target
        normalized_data['side'] = np.sign(target)
        
        dfs.append(normalized_data)
    
    return pd.concat(dfs)


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


def compute_quantiles(df, column):
    q1 = df[column].quantile(0.25)                 
    q3 = df[column].quantile(0.75)

    return (q1, q3)

def subset_by_iqr(df, column, q = None, whisker_width=3):
    """Remove outliers from a dataframe by column, including optional 
       whiskers, removing rows for which the column value are 
       less than Q1-1.5IQR or greater than Q3+1.5IQR.

           '''
           Chen 2016
           Outliers were defined as responses whose distance from their respective response centroids 
           exceeded the 3rd quartile by 3 times the interquartile range 
           (because of the small numbers of responses collected on each side, 
           these quartiles were computed using responses from all four target locations)
           '''


    Args:
        df (`:obj:pd.DataFrame`): A pandas dataframe to subset
        column (str): Name of the column to calculate the subset from.
        whisker_width (float): Optional, loosen the IQR filter by a
                               factor of `whisker_width` * IQR.
    Returns:
        (`:obj:pd.DataFrame`): Filtered dataframe
    """
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


def recenter_data_by_condition(side_data):
    dfs = []
    
    for condition, data in side_data.groupby('condition'):
        response_mean = np.mean(data[['x', 'y']].values, axis = 0)

        #if condition is 'conflict':
        #    response_mean = np.zeros(2)

        data[['x_recentered', 'y_recentered']] = data[['x', 'y']] - response_mean
        dfs.append(data)

    return pd.concat(dfs)

def compute_rotated_target_locations(simulated_data):
    target_locations = np.unique(simulated_data[['targetx', 'targetz']].values, axis = 0)
    rotated_target_locations = []

    dfs = []

    for target in target_locations:
        plt.scatter(target[0], target[1], marker = 'D', color = 'grey')

        start = np.array([0, 0])


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
    
    plt.xlim(-3,3)
    plt.ylim(-1,5)
    
    return target_data


#compute response variability (equation 9)
def compute_SDs_Chen(normalized_data_search_distances):
    #Chen Formula 
    S2s = {} ; Ss  = {} ; dfs = []
    for condition, data in normalized_data_search_distances.groupby('condition'):
        d = data['euclidean_distance'].values 
        n = d.shape[0]

        S2 = np.sum(d**2)/(n-1)
        S  = np.sqrt(S2)

        df_temp = {'condition' : condition,
                   'euclidean_distance (S)' : S,
                   'euclidean_distance (S2)' : S2}
        
        dfs.append(pd.DataFrame(df_temp, index = [condition]))
        
    return pd.concat(dfs)

def recover_relative_segment_pos(agent_pos):
    from analysis.trajectories.helpers import compute_relative_segment_position
    segment_start = np.array([0.0, 0.0])
    segment_end = np.array([0.55646095, -0.07325947])
    epsilon = 0.00001




    p1,p2, segment_intersection = compute_relative_segment_position(segment_start = segment_start,
                                                            segment_end = segment_end, 
                                                            agent_pos = agent_pos, 
                                                            epsilon = epsilon)
    
    
    direction = segment_start - segment_end
    orth_direction = np.array([-direction[1], direction[0]])
    
    segment_intersection = int(np.sign(np.linalg.det(np.vstack([orth_direction,agent_pos]))) == 1) * np.array(segment_intersection)

    segment_length = np.linalg.norm(segment_start-segment_end)
    segment_progress = np.linalg.norm(segment_start-segment_intersection) #where between the two points of the segment is the agent currently
    relative_segment_position = segment_progress / segment_length
    
    


    return relative_segment_position   
