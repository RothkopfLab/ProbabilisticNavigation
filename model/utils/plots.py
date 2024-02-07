import numpy as np 
import matplotlib.pyplot as plt 
import math
import pandas as pd
import os

from utils.helpers import rotate, covariance_ellipse 
from matplotlib.patches import Ellipse


def get_positional_belief_ellipse(
    X,
    P,
    sd=1.96,
    edgecolor=None,
    facecolor=None,
    alpha=1.0,
    ls="solid",
):
    # state of the agent (position, sd)
    mean = (X[0], X[1])
    x, y = mean
    orientation = np.deg2rad(45)  # viewing angle
    
    # fit the ellipse
    ellipse = covariance_ellipse(P, sd)
    angle = np.degrees(ellipse[0])
    width = ellipse[1] #* 2.0  #why exactly do we multiply by two here ??? --> i dont think we need this at all (!)
    height = ellipse[2] #* 2.0 #why exactly do we multiply by two here ??? --> i dont think we need this at all (!)

    e = Ellipse(
        xy=mean,
        width=sd * width,
        height=sd * height,
        angle=angle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        lw=2,
        ls=ls,
    )
    
    return e

def plot_positional_belief(x_t, Sigma_t): 
    e = get_positional_belief_ellipse(x_t[:2],Sigma_t[:2,:2], edgecolor = 'black', alpha = .25)
    
    #get the current axis 
    ax = plt.gca()
   
    # add the positional belief ellipse 
    ax.add_patch(e)

    #plots the positions xy coordinates 
    position = plt.scatter(x_t[0], x_t[1], marker="+", color = 'grey')
    
    return position, e 

def plot_agent_heading(mu_t, height):
    heading_dir = mu_t[:2] + height * np.array([np.cos(mu_t[2]), np.sin(mu_t[2])])
    
    heading = plt.plot((mu_t[0], heading_dir[0]), (mu_t[1], heading_dir[1]), c = 'red', ls = 'dotted')

    return heading[0]

def plot_measurement(x_t, lm_coords):
    line = plt.plot([x_t[0,0], lm_coords[0]], [x_t[1,0], lm_coords[1]], color=(0,0,1))
    return line
        
def plot_agent_fov(mu_t, fov, cone_len, color = 'r', ls = '-', ax = None):
    fov = np.deg2rad(fov)
    left  = plt.plot([mu_t[0,-1], mu_t[0,-1]+cone_len*np.cos(mu_t[2,-1] + fov/2)], [mu_t[1,-1], mu_t[1,-1]+cone_len*np.sin(mu_t[2,-1] + fov/2)], color=color, label = 'fov left',  ls = ls)
    right = plt.plot([mu_t[0,-1], mu_t[0,-1]+cone_len*np.cos(mu_t[2,-1] - fov/2)], [mu_t[1,-1], mu_t[1,-1]+cone_len*np.sin(mu_t[2,-1] - fov/2)], color=color, label = 'fov right', ls = ls)
    
    return left[0], right[0]


def plot_arena(startpoint, landmarks, pickups, landmark_rotation = 0, xlim = (-3, 15), ylim = (-3,15), show_start = False, figsize = (6,3)):
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_xticks(np.arange(xlim[0],xlim[1]+1,1))
    ax.set_yticks(np.arange(ylim[0],ylim[1]+1,1))

    if show_start:
        ax.scatter(startpoint[0], startpoint[1], s = 60)
        ax.text(startpoint[0] + 0.5, startpoint[1] - 0.25, s = 'Start', label = 'Start')
    
    #landmark locations 
    for i, landmark in enumerate(landmarks): 
        landmark = rotate(startpoint,landmark, np.deg2rad(landmark_rotation))
        ax.scatter(landmark[0], landmark[1], s = 150, marker = '*', label = 'Landmark: ' + str(i))

    for i, pickup in enumerate(pickups): 
        ax.scatter(pickup[0], pickup[1], s = 20, marker = 'D', label = 'Pickup: ' + str(i))
    
        
    return fig, ax 


def plot_idealized_trajectory(experiment, trajectory_id, start, jitterx = 0.0, jittery = 0.0, ax = None):

    if ax is None:
        fig = plt.figure(figsize = (7,10))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
    else:
        fig = ax.figure
     
            
    post_ids = experiment.post_ids
    
    posts = trajectory_id.split('-')
    posts = [int(post) for post in posts]
    segment_ends = [0, post_ids[posts[0]].copy(), post_ids[posts[1]].copy(), post_ids[posts[2]].copy()]
    

    segment_ends[1] += np.array([jitterx,jittery])
    
    
    for i,entry in enumerate(segment_ends[1:]):
        ax.scatter(entry[0],entry[1], label = 'Post: ' + str(i+1))

        
    #first segment
    dxdy = segment_ends[1] - start
    ax.arrow(start[0], start[1], dxdy[0], dxdy[1], width = 0.025, length_includes_head = True, color = 'orange')

    #second segment
    dxdy = segment_ends[2] - segment_ends[1]
    ax.arrow(segment_ends[1][0], segment_ends[1][1], dxdy[0], dxdy[1], width = 0.025, length_includes_head = True, color = 'orange')

    #third segment
    dxdy = segment_ends[3] - segment_ends[2]
    ax.arrow(segment_ends[2][0], segment_ends[2][1], dxdy[0], dxdy[1], width = 0.025, length_includes_head = True, color = 'orange')
    
    #return paths 
    dxdy = segment_ends[1] - segment_ends[3]
    retrn = ax.arrow(segment_ends[3][0], segment_ends[3][1], dxdy[0], dxdy[1], width = 0.025, length_includes_head = True, color = 'lightblue', ls = '--', label = 'Estimated homing')
    
    return fig, ax 