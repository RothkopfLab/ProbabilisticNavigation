import math 
import numpy as np 
import pandas as pd 


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


def polar_plot(heading_errors, bin_size = 10, figsize = (10,8), color = '.8', ax = None): 
    radians = heading_errors
    degrees = np.rad2deg(heading_errors)
    degrees = (degrees + 360) % 360
    
    a , b=np.histogram(degrees, bins=np.arange(0, 360+bin_size, bin_size), density = True)
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])


    if ax is None: 
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
    
    fig = ax.figure
    ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color=color, edgecolor='k')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticks([])
    
    return fig, ax 


'''
Compute Heading Directions 
'''

def get_heading_error(p1,p2, normalize = True):
    delta =  p1 - p2 
    angle =  np.arctan2(delta[1],delta[0])
    
    
    if normalize: 
        angle = normalize_angle(angle)
    
    return angle

def get_heading_errors(data):
    return np.array([np.arctan2(point[0], point[1]) for point in data])
    

'''
We measured homing direction as the circular mean of response directions.
Accordingly, the variability in homing was measured as the circular standard deviation of response direction
'''

def get_circ_stats_rad(heading_errors):
    circ_mean = ss.circmean(heading_errors)
    circ_std =  ss.circstd(heading_errors)
    
    return circ_mean, circ_std 

def get_circ_stats_deg(heading_errors):
    circ_mean = ss.circmean(np.deg2rad(heading_errors))
    circ_std =  ss.circstd(np.deg2rad(heading_errors))
    
    return np.rad2deg(circ_mean), np.rad2deg(circ_std)

def normalize_angle(phi):
    '''
    normalize the angle between -pi and pi 
    '''
    while(phi > np.pi): 
        phi = phi - 2*np.pi
    while(phi < -np.pi):
        phi = phi + 2*np.pi
        
    phiNorm = phi
    return phi 