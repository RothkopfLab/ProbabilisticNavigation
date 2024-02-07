import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.metrics import mean_squared_error


trial_types = {
    1 : 'landmark',
    2 : 'self-motion',
    3 : 'combined', 
    4 : 'conflict'
}


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
                                        'x' : normalized_points[:,0], 
                                        'y' : normalized_points[:,1]})

        pooled_centroid = np.mean((normalized_points[:,0], normalized_points[:,1]), axis = 1)
        
        if plot:
            plt.figure()
            plt.scatter(normalized_points[:,0], normalized_points[:,1])
            plt.scatter(pooled_centroid[0], pooled_centroid[1], marker = 'D', color = 'red')
            plt.scatter(0.0, 0.0, marker = '+', color = 'black')

            plt.xlim(-5,5)
            plt.ylim(-5,5)
            plt.gca().set_aspect('equal')
            plt.suptitle(trial_types[cond])

        endpoints_by_condition.append(normalized_data)

    normalized_data = pd.concat(endpoints_by_condition)
    
    return normalized_data


def compute_search_distances(normalized_data):
    pooled_centroid = np.mean(normalized_data, axis = 0)
    search_distances = np.linalg.norm(normalized_data - pooled_centroid, axis = 1)
    
    return search_distances 


def compute_metrics(normalized_data):
    metrics = []
    for condition, data in normalized_data.groupby('condition'):
        df = pd.DataFrame(columns = ['condition', 'metric', 'value'])

        normalized_points = data[['x', 'y']].values

        actual = np.zeros(normalized_points.shape)
        pooled_centroid = np.mean((normalized_points[:,0], normalized_points[:,1]), axis = 1)

        mean = np.tile(pooled_centroid, (normalized_points.shape[0],1))

        RMSE = np.sqrt(mean_squared_error(actual, normalized_points))
        SD =  np.sqrt(mean_squared_error(mean, normalized_points))


        entry = pd.DataFrame({'condition' : [condition] * 2,
                              'metric' : ['RMSE', 'SD'],
                              'value' : [RMSE, SD]})

        metrics.append(entry)

    metrics = pd.concat(metrics).reset_index()
    
    return metrics 

def extract_means_and_sigmas(normalized_data, metrics):
    means = {}
    sigmas = {}

    for cond, data in normalized_data.groupby('condition'):
        normalized_points = data[['x', 'y']].values

        m = metrics[metrics.condition == cond]

        SD = m[m.metric == 'SD'].value

        means[cond] = np.mean((normalized_points[:,0], normalized_points[:,1]), axis = 1)
        sigmas[cond] = float(SD**2)
    
    return means, sigmas 