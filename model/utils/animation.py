import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML

from utils.plots import plot_arena, plot_positional_belief, plot_measurement, plot_agent_fov, covariance_ellipse, plot_agent_heading, plot_idealized_trajectory
from utils.helpers import get_ellipse_params, get_landmark_mu, get_landmark_sigma

def get_SLAM_animation(xs, mus = None, sigmas = None, start = None, measurement_data = None, showLandmarkSigma = False, showFov = False, fov = 100, cone_len = 20, interval = 127, save = False, experiment = None, idealized_tr = None, jitter = (0.0, 0.0), pattern = '*...', arena_limits = ((-3,3), (-5,5)), scale = 2):
    """Animates the agents position (and gaussian belief about own position and landmark) throughout navigation sequence (0:T)

    Args:
        xs (list (np.array(nx)): Sequence of states (i.e. coordinates of each timestep t)
        mus (list (np.array(nx)): Sequence of mean belief states (i.e. coordinates of each timestep t)
        sigmas ((np.array(nx,nx)) , optional): Agents uncertainty about belief for each timestep t used for gaussian belief ellipse. Defaults to None.
        start (np.array(2), optional): start point for agent. Defaults to None.
        measurement_data ([type], optional): Plot measurements in animation. Defaults to None.
        showLandmarkSigma (bool, optional): Show landmark uncertainties in animation. Defaults to False.
        showFov (bool, optional): Enable Field of View in animation. Defaults to False.
        fov (float, optional): Field of view angle. Defaults to 60.
        cone_len (int, optional): Field of View Cone Length. Defaults to 20.
        interval (int, optional): Animation frame intervall. Defaults to 20.
        save (bool, optional): Return Anim Object for saving. Defaults to False.
        idealized_tr (string, optional): Plot post layout from Chen 2017 (e.g. '148'). Defaults to None.
        pattern(string, optional): Sequence of markers for pickups and landmarks (e.g. '***...')

    Returns:
        [FuncAnimation]: Animation sequence of agent navigating 
    """    

    fig, ax = plt.subplots(figsize=(scale * np.abs(arena_limits[0][0] - arena_limits[0][1]), scale * (np.abs(arena_limits[0][0] - arena_limits[0][1]))))
    ax.set_xlim(arena_limits[0])
    ax.set_ylim(arena_limits[1])
    ax.set_aspect('equal')
    
    #get only the coordinates 
    coords = np.array(xs)[:,:3]

    #time 
    t = len(coords)
    
    animation_buffer = []
    landmark_sigmas = []
    landmark_positions = []
    n_landmarks = int(xs[0].shape[0] - 3) // 2 

    if sigmas:
        n_landmarks = int((sigmas[0].shape[0] - 3) / 2)

    text = ax.text(s = 'iteration:' + str(0), x = 0.65, y = 0.85, transform = fig.transFigure)


    
    #state 
    position, ___ = plot_positional_belief(coords[0], 0.0 * np.eye(2)) 
    tr, = ax.plot(coords[0,0], coords[0,0], color = 'black')


    landmarks = [] 
    if not pattern: pattern = n_landmarks * '*'
    for lm_id in range(n_landmarks):
        lm_mu = get_landmark_mu(xs[0], lm_id)
        #TODO maybe different patterns based on some input string (e.g. (*DDD))
        lm_plot = ax.scatter(lm_mu[0], lm_mu[1], marker = pattern[lm_id], s = 250)
        landmarks.append(lm_plot)

        
    if (idealized_tr is not None) and (experiment is not None): 
        plot_idealized_trajectory(experiment, idealized_tr, start, ax = ax, jitterx=jitter[0], jittery=jitter[1])

    

    if showFov:
        left, right = plot_agent_fov(coords[0].reshape(3,1), fov, cone_len)
   
    if measurement_data:
        for m in measurement_data[0]:
            line = plot_measurement(coords[0].reshape(3,1), m)
            animation_buffer.append(line)

    #belief 
    if mus:
        coords_belief =  np.array(mus)[:,:3]
        position_belief, e_belief = plot_positional_belief(coords_belief[0], 0.0 * np.eye(2))
        tr_belief, = ax.plot(coords_belief[0,0], coords_belief[0,0], color='grey', ls = 'dotted') 
        
        if showFov:
            left_belief, right_belief = plot_agent_fov(coords_belief[0].reshape(3,1), fov, cone_len, color = 'grey', ls = 'dotted')
    
        if sigmas:
            width, height, angle =  get_ellipse_params(coords_belief[0],sigmas[0][:2,:2])
            e_belief.width = width
            e_belief.height = height
            e_belief.angle = angle 

            if showLandmarkSigma:
                for lm_id in range(n_landmarks):
                    lm_mu = get_landmark_mu(mus[0], lm_id)
                    lm_sigma = get_landmark_sigma(sigmas[0], lm_id)
                    
                    position_belief, e_sigma = plot_positional_belief(lm_mu, 0.0 * np.eye(2)) 
                    
                    landmark_positions.append(position_belief)
                    landmark_sigmas.append(e_sigma)
                    
                    position.set_visible(False)
                    e_sigma.set_visible(False)
                    
                    width, height, angle = get_ellipse_params(lm_mu,lm_sigma)
                    e_sigma.width = width
                    e_sigma.height = height
                    e_sigma.angle = angle
        
    else: 
        showLandmarkSigma = False 
    

    def animate(i):
        text.set_text('iteration: ' + str(i))

        #state update 
        position.set_offsets(coords[i][:2])
        tr.set_data(coords[:i,0],coords[:i,1])

        #update landmark locations 
        for lm_id in range(n_landmarks):
            lm_mu = get_landmark_mu(xs[i], lm_id)
            landmarks[lm_id].set_offsets(lm_mu[:2])

        if showFov: 
            update_fov(left,right,coords[i].reshape(3,1), fov, cone_len)

        if measurement_data:
             #clear lines from plot
            for line in animation_buffer:
                line[0].remove()   
                
            #reset
            animation_buffer.clear()
            
            for m in measurement_data[i]:
                line = plot_measurement(coords[i].reshape(3,1), m)
                animation_buffer.append(line) #append to plotted lines     
    

        #belief update 
        if mus: 
            position_belief.set_offsets(coords_belief[i][:2])
            e_belief.set_center(coords_belief[i][:2])
            tr_belief.set_data(coords_belief[:i,0],coords_belief[:i,1])

            #set belief ellipse 
            if sigmas:
                width, height, angle = get_ellipse_params(coords_belief[i].reshape(3,1),sigmas[i][:2,:2])
                e_belief.width = width
                e_belief.height = height
                e_belief.angle = angle 

            if showFov: 
                update_fov(left_belief, right_belief, coords_belief[i].reshape(3,1), fov, cone_len)
                 
            if showLandmarkSigma:
                for lm_id in range(n_landmarks):
                    #get landmark mean and associated uncertainty
                    lm_mu = get_landmark_mu(mus[i], lm_id)
                    lm_sigma = get_landmark_sigma(sigmas[i], lm_id)
                    
                    #get plot object
                    e_position = landmark_positions[lm_id]
                    e_sigma = landmark_sigmas[lm_id]
                    
                    #set position of landmarks belief ellipse 
                    e_position.set_offsets(lm_mu[:2])
                    e_sigma.set_center(lm_mu[:2])

                    #covariance params 
                    width, height, angle = get_ellipse_params(lm_mu,lm_sigma)
                    e_sigma.width = width
                    e_sigma.height = height
                    e_sigma.angle = angle 
                    
                    if lm_sigma.max() < 100: 
                        e_position.set_visible(True)
                        e_sigma.set_visible(True)
                    else:
                        e_sigma.set_visible(False)


                

    anim = FuncAnimation(
        fig, animate, interval=interval, frames=t)
    
    if save:
        return anim 

    return HTML(anim.to_html5_video())

def update_fov(left,right,mu_t,fov,cone_len):
    """Updates the Field of View of the plot in-place 

    Args:
        left, right (pyplot line object): fov line objects, which can be used to later alter the coordinates
        mu_t (np.array(n,1)): agents state, usually mu_t[:2] includes the position of the agent 
        fov (float): the agents field of view angle 
        cone_len (float):  the length of the field of view cone being displayed 
    """    
    fov = np.deg2rad(fov)
    
    left.set_data([mu_t[0,-1], mu_t[0,-1]+cone_len*np.cos(mu_t[2,-1] + fov/2)], [mu_t[1,-1],  mu_t[1,-1]+cone_len*np.sin(mu_t[2,-1] + fov/2)])
    right.set_data([mu_t[0,-1], mu_t[0,-1]+cone_len*np.cos(mu_t[2,-1] - fov/2)], [mu_t[1,-1], mu_t[1,-1]+cone_len*np.sin(mu_t[2,-1] - fov/2)])


def update_heading(heading, mu_t, cone_len):
    """Updates heading of the agent in-place

    Args:
        heading (float): heading angle 
        mu_t (np.array(n,1)): agents state, usually mu_t[:2] includes the position of the agent 
        cone_len (float):  the length of the field of view cone being displayed 
    """    
    heading_dir = mu_t[:2] + cone_len * np.array([np.cos(mu_t[2]), np.sin(mu_t[2])])
    heading.set_data((mu_t[0], heading_dir[0]), (mu_t[1], heading_dir[1]))




