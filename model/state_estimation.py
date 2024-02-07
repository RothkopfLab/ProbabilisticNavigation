from casadi import * 
import numpy as np 
import pandas as pd 
import copy 


from utils.helpers import get_landmark_mu, get_landmark_sigma, normalize_angle

def construct_Fx(N):
    pos = np.eye(3)
    lm  = np.zeros((3,2*N))
    return np.hstack((pos,lm))


def construct_Fx_j(j,N):
    F = np.zeros((5,N))
    F[:3,:3] = np.eye(3)
    F[3,2*j+3] = 1
    F[4,2*j+4] = 1

    return F

def unicycle_motion_model(x_t, u_t, dt):
    theta = x_t[2,0]
    v_t = u_t[0]
    w_t = u_t[1]
    
    xp = np.cos(theta) * v_t * dt 
    yp = np.sin(theta) * v_t * dt 
    thetap = w_t * dt
   

    return np.array([xp,yp,normalize_angle(thetap)]).reshape(3,1)

def unicycle_motion_model_jacobian(x_t, u_t, dt):
    theta =  x_t[2,0]
    v_t = u_t[0]
    w_t = u_t[1]
    
    #first two columns 
    cols = np.array([[0, 0],
                     [0, 0],
                     [0, 0]])
    
    motion_derivative = np.array([
                            - dt * v_t * np.sin(theta),
                              dt * v_t * np.cos(theta), 
                              0
                                ]).reshape(3,1)

    return np.hstack((cols,motion_derivative))

def process_cov_jacobian(mu_t, dt):
    G_x = np.zeros((mu_t.shape[0], 2))

    sinTheta, cosTheta = sin(mu_t[2,0]), cos(mu_t[2,0])

    G_x[0, 0] = cosTheta * dt
    G_x[1, 0] = sinTheta * dt
    G_x[2, 1] = dt

    return G_x 

def EKF_SLAM_Prediction(mu_t, Sigma_t, u_t, R_t, dt): 
    #number of landmarks 
    N = (mu_t.shape[0] - 3)//2
    
    #non-linear update step of the state g(u_t, u_t-1)
    Fx = construct_Fx(N)
    mu_t = mu_t + Fx.T @ unicycle_motion_model(mu_t, u_t, dt)

    P_x = process_cov_jacobian(mu_t, dt)
    
    #compute the jacobian estimate of motion 
    G_t = np.eye(2*N+3) + Fx.T @ unicycle_motion_model_jacobian(mu_t,u_t, dt) @ Fx

    #update the covariance of the state (doesn't touch the landmark block)
    Sigma_t = G_t @ Sigma_t @ G_t.T + P_x @ R_t @ P_x.T
    
    return mu_t, Sigma_t

################################################################################################
                                ### Correction  Step ###
################################################################################################

def expected_observation(mu,j):
    # compute expected observation
    delta = np.array([mu[2*j+3][0] - mu[0][0], mu[2*j+4][0] - mu[1][0]])
    #delta = delta.reshape(2,1) #added this 
    q = delta.T.dot(delta)
    sq = np.sqrt(q)
    z_theta = np.arctan2(delta[1],delta[0])
    z_hat = np.array([[sq], [z_theta-mu[2][0]]]) 
    
    return delta, z_hat

def expected_observation_jacobian(delta):    
    q = delta.T @ delta
    dx = delta[0]
    dy = delta[1]
    
    sq = np.sqrt(q)

    
    H_z = np.array([[-sq*dx, -sq*dy, 0, sq*dx, sq*dy],
                    [dy, -dx, -q, -dy, delta[0]]], dtype='float')
        
    return H_z


def EKF_SLAM_Correction(mu_t, Sigma_t, obs, Qt):    
    #number of landmarks 
    N = mu_t.shape[0]

    #range bearing landmark_id 
    r = obs[0]
    theta = obs[1]
    j = int(obs[2])
    
    #comptue expected observation 
    delta, z_hat = expected_observation(mu_t,j)
    q = delta.T @ delta

    init = False 

    #initialize an unseen landmark 
    if (get_landmark_sigma(Sigma_t, j)[0,0]) >= 1e6 and (get_landmark_sigma(Sigma_t, j)[1,1] >= 1e6):
        print('initialize landmark at new position')
        mu_t[3+(2*j):3+2*j+2] = mu_t[:2] + np.array([r * np.cos(normalize_angle(theta + mu_t[2])), r * np.sin(normalize_angle(theta + mu_t[2]))])
        init = True 
            
    # calculate expected observation Jacobian
    F = construct_Fx_j(j,N)
    H_z = expected_observation_jacobian(delta)
    H = 1/q * H_z @ F
    
    # calculate difference between expected and real observation (equation 6)
    z_dif = np.array(DM(np.array([[r],[theta]]) - np.array(z_hat)))
    z_dif[1] = normalize_angle(z_dif[1]) #normalize angle of innovation 
    
    Sigma_k_t = H @ Sigma_t @ H.T + Qt #(equation 7)
    K = Sigma_t @ H.T @ inv(Sigma_k_t) #(equation 8)
           
    # update state vector and covariance matrix  (equation 9) 
    if init:
        mu_t = mu_t 
    else: 
        mu_t = mu_t + K @ z_dif
      
    I_KH = np.eye(N) - K @ H
    Sigma_t = I_KH @ Sigma_t @ I_KH.T + K @ Qt @ K.T
    
    return mu_t, Sigma_t


def range_bearing_observation(x_t, z_t): 
    #measurement data   
    r = z_t[0]   #range
    phi = z_t[1] #bearing
    
    #estimated agent position
    agent_location = x_t[:2]
    
    #estimated agents positiion
    agent_rotation = x_t[2]
    
    #take a relative measurement 
    rotation = normalize_angle(phi + agent_rotation)
    relative_measurement = np.array([r * np.cos(rotation),
                                     r * np.sin(rotation)
                                     ]).reshape(2,1)
        
    #observed location of landmark j 
    observed_location = agent_location + relative_measurement
    
    return observed_location

#determine observable landmarks for the state
def get_measurements(obs, x_true, z_t, obs_state):
    #get the measurements and get them for plotting 
    measurements = {'z_t' : [], 'coords' : []}

    #instantiate yet unknown landmark to the belief state 
    for idx in obs:        
        #obtain range bearing measurement 
        r = z_t[2*idx]
        phi = z_t[2*idx+1]

        phi = normalize_angle(phi)

        measurements['coords'].append(range_bearing_observation(x_true[:3], (r,phi)))
        measurements['z_t'].append(np.array([r, phi, int(idx)]))
    
    
    return measurements
