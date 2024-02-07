import math
import numpy as np 
import pandas as pd 
import copy 

def read_file(path, filename):
    measurement_data = pd.read_csv(path + filename + '_MeasurementLog.txt')
    control_data     = pd.read_csv(path + filename + '_ControlLog.txt')
    
    controls  = control_data[['v','w']].values[:-1]
    timesteps = control_data[['time']].values[:-1].flatten()

    x0 = control_data[['x', 'z', 'theta']].values[0,0]
    y0 = control_data[['x', 'z', 'theta']].values[0,1]
    t0 = control_data[['x', 'z', 'theta']].values[1,2]
    x_t = np.array([x0, y0, t0]).reshape(3,1)
    
    return x_t, controls, control_data, measurement_data, timesteps 


def covariance_ellipse(P, deviations=1):    
    U, s, _ = np.linalg.svd(P)

    orientation = np.math.atan2(U[1, 0], U[0, 0])
    width  = deviations * np.math.sqrt(s[0])
    height = deviations * np.math.sqrt(s[1])

    if height > width:
        raise ValueError("width must be greater than height")

    return (orientation, width, height)


def get_ellipse_params(X,P):
    # state of the agent (position, sd)
    mean = (X[0], X[1])
    x, y = mean
    sd = 1

    # fit the ellipse
    ellipse = covariance_ellipse(P, sd)
    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.0
    height = ellipse[2] * 2.0
    
    
    return width, height, angle 


#read in Landmark data 

#normalize angle
def normalize_angle(phi):
    while(phi > np.pi): 
        phi = phi - 2*np.pi
    while(phi < -np.pi):
        phi = phi + 2*np.pi
        
    return phi 
    
    
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


def sigmoid(x,alpha):
  return 1 - 1 / (1 + np.exp(- alpha * x))

def gauss_2d_distance(mu_lm, Sigma_lm, mu_pos, Sigma_pos):
    Sigma_pooled = (Sigma_lm + Sigma_pos) / 2 
    Sigma_frac = np.linalg.det(Sigma_pooled) / np.sqrt(np.linalg.det(Sigma_lm) *  np.linalg.det(Sigma_pos))
    
    return 1/8 * (mu_lm - mu_pos).T @ np.linalg.inv(Sigma_pooled) @ (mu_lm - mu_pos) + 1/2 * np.log(Sigma_frac)

def get_landmark_mu(mu_t, lm_id):
    
    if lm_id == -1: 
        return mu_t[:2]
    else:
        return mu_t[3+(2*lm_id):3+2*lm_id+2]
    
def get_landmark_sigma(Sigma_t, lm_id):
    return Sigma_t[3+2*lm_id:3+2*lm_id+2,3+2*lm_id:3+2*lm_id+2]

def get_heading(point1,point2):
    d = point2 - point1
    heading_angle = np.arctan2(d[1], d[0]).reshape(1)
    return heading_angle

def get_landmark_side(mu_t, lm_idx):
    #get forward direction
    a = mu_t[:2]
    b = mu_t[:2] + 1 * np.array([np.cos(mu_t[2]), np.sin(mu_t[2])])
    
    #get query point 
    q = get_landmark_mu(mu_t, lm_idx)

    return np.sign((b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]))


def get_route(post_ids, route, start_position):
    route = route.split('-')
    l = [start_position, post_ids[int(route[0])], post_ids[int(route[1])], post_ids[int(route[2])], post_ids[int(route[0])]]
    l = [copy.deepcopy(i) for i in l]

    return l



