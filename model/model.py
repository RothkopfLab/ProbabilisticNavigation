from casadi import * 
import casadi.tools as cat

import numpy as np
from numpy import matlib as mb
import matplotlib.pyplot as plt

import time
from utils.helpers import read_file, normalize_angle, sigmoid 

class Model:

    def __init__(self, n_landmarks, motion_limits, R_diag, Q_diag, R_diag_generative, Q_diag_generative, V, V_generative, observation_type = 'range_bearing', representation_random_walk = False, representation_type = None, verbose = False, fov = 100):
        self.n_landmarks = n_landmarks
        self.obs0 = np.ones(n_landmarks)
        
        #control limits 
        self.v_max = motion_limits['v_max']
        self.v_min = motion_limits['v_min']
        self.w_max = motion_limits['w_max']
        self.w_min = motion_limits['w_min']

        self.dv_max = motion_limits['dv_max']
        self.dw_max = motion_limits['dw_max']


        self.representation_random_walk = representation_random_walk


        #subjective
        self.Q_diag = Q_diag # motion covariance
        self.R_diag = R_diag # measurement covariance 
        self.V = V # representation covariance

        #generative model 
        self.Q_diag_generative = Q_diag_generative #motion covariance
        self.R_diag_generative = R_diag_generative #measurement covariance 
        self.V_generative = V_generative #representation covariance 



        #weight matrix for state cost
        Q = np.zeros((3,3))
        Q[0,0] = 1
        Q[1,1] = 1
        Q[2,2] = 0
        self.Q = Q

        #weight matrix for control cost 
        R = np.zeros((2,2))
        R[0,0] = 1
        R[1,1] = 0.0       
        self.R = R 
        self.Rtr = 0.25

    

        self.representation_type = representation_type

        self.observation_type = observation_type
    
        #setup FOV and height 
        self.fov = fov
        self.height = 600.0
        
        #setup the casadi state
        self.setup_state()
        
        #setup functions 
        self.setup_functions()


    def setup_state(self):    
        self.nx = 3 + self.n_landmarks * 2
        self.m = [] # define map 

        for i in range(self.n_landmarks):
            self.m.append('lm' + str(i+1) + 'x') 
            self.m.append('lm' + str(i+1) + 'y')

        #Belief State         
        self.mu = cat.struct_symSX(['x', 'y','theta'] + self.m)
        self.Sigma = SX.sym('Sigma', self.nx, self.nx)

        self.alpha = SX.sym('alpha')
        self.dt_ = SX.sym('dt')

        #is the landmark observable?
        self.observable = SX.sym('obs', self.n_landmarks)


        # Controls 
        self.u = cat.struct_symSX(['v', 'w'])
        self.nu = self.u.shape[0]


        self.v = SX.sym('v')


        #Range Bearing Observations
        zs = []

        if self.observation_type == 'range_bearing':
            for i in range(self.n_landmarks):
                zs.append('lm' + str(i+1) + '_range') 
                zs.append('lm' + str(i+1) + '_bearing')

        #Identity Observation Observations 
        if self.observation_type == 'identity':
            for i in range(self.n_landmarks):
                zs.append('lm' + str(i+1) + '_x') 
                zs.append('lm' + str(i+1) + '_y')

        self.z = cat.struct_symSX(zs)
        self.nz = self.z.shape[0]


        #Map containing landmarks and pickup objects 
        lms = []
        for i in range(self.n_landmarks):
            lms.append('lm' + str(i+1) + 'x') 
            lms.append('lm' + str(i+1) + 'y')

        self.lms = cat.struct_symSX(lms)
        self.nlm = self.lms.shape[0]

  
    def setup_functions(self):
        ### 
            # Generative Model 
        ###

        self.process_cov_generative   = self.process_cov_signal_dependent(self.u, self.Q_diag_generative)
        self.obs_cov_generative = self.get_R_lm_gaze_dependent(self.mu, self.R_diag_generative)

        #### 
            # Belief Dynamics and Observation 
        ####
        #initialized dynamics 
        self.F   = self.dynamics(self.mu, self.u, self.dt_)
        self.F_j = self.F.jac() #TODO create dynamics Jacobian

        #process noise 
        self.process_cov   = self.process_cov_signal_dependent(self.u, self.Q_diag)
        #self.process_cov   = self.process_cov_control(self.u)
        self.process_cov_j = self.process_cov_jacobian(self.mu, self.u, self.dt_)

        #intialize representation noise 
        if self.representation_type == 'all':
            self.representation_cov = self.get_V_lm_all(self.observable)
        elif self.representation_type == 'invisible':   
            self.representation_cov = self.get_V_lm_invisible(self.observable)
        else: 
            self.representation_cov = self.get_V_lm_none(self.observable)

        #initialize observation function 
        if self.observation_type == 'range_bearing':
            self.h = self.range_bearing_observation(self.mu)

        if self.observation_type == 'identity':
            self.h = self.identity_observation(self.mu)

        self.H_j = self.h.jacobian()
        #self.obs_cov = self.get_R_lm(self.mu)
        self.obs_cov = self.get_R_lm_gaze_dependent(self.mu, self.R_diag)

        #discontinous sensing via SDF f
        self.d_j = self.delta_j_triangle(self.mu, fov = self.fov, height = self.height, alpha = self.alpha, obs = self.observable)
        self.d_j_real = self.get_delta_j(self.mu, self.observable)

        #initialize belief dynamics 
        self.BF = self.Belief_Propagation(self.mu, self.Sigma, self.u, self.alpha, self.dt_, self.observable)


    #unicycle dynamics model 
    def dynamics(self, state, controls, dt):
        rhs = vertcat(cos(state['theta']) * controls['v'] * dt, 
                  sin(state['theta']) * controls['v'] * dt, 
                  controls['w'] * dt)

        rhs = vertcat(rhs, vertcat(self.n_landmarks * [0.0, 0.0]))
    
        return Function('f', [state,controls, dt], [rhs], ['x', 'u', 'dt'],['x_t+1'])
    
    #constant process_cov 
    def process_cov_control(self, u_t):
        rhs = SX.zeros(self.nu,self.nu)
        rhs[0,0] = self.Q_diag[0]
        rhs[1,1] = self.Q_diag[1]

        return Function('Q_t', [u_t], [rhs], ['u_t'], ['Q_t'])

    def process_cov_signal_dependent(self, u_t, Q_diag):
        rhs = SX.zeros(self.nu,self.nu)
        rhs[0,0] = Q_diag[0] * u_t['v']**2 + Q_diag[1] * u_t['w']**2 
        rhs[1,1] = Q_diag[2] * u_t['v']**2 + Q_diag[3] * u_t['w']**2 

        return Function('Q_t', [u_t], [rhs], ['u_t'], ['Q_t'])

    def process_cov_jacobian(self, mu_t, u_t, dt):
        jac = jacobian(self.F(mu_t, u_t, dt), u_t)
        
        return Function('process_cov_jacobian', [mu_t, u_t, dt], [jac])


    def process_cov_constant(self, u_t):
        rhs = SX.zeros(self.nx,self.nx)
        rhs[0,0] = self.Q_diag[0]
        rhs[1,1] = self.Q_diag[1]
        rhs[2,2] = self.Q_diag[2]

        return Function('Q_t', [u_t], [rhs], ['u_t'], ['Q_t'])


    def get_V_lm_invisible(self, obs):
        rhs = self.z.squared(SX.zeros(self.n_landmarks * 2, self.n_landmarks * 2))

        for i in range(0,2*self.n_landmarks,2):
            rhs[i,i] = self.V * (1-obs[i//2])
            rhs[i+1,i+1] = self.V * (1-obs[i//2])

        return Function('V_t', [obs], [rhs], ['obs'], ['V_t'])


    def get_V_lm_all(self, obs):
        rhs = self.z.squared(SX.zeros(self.n_landmarks * 2, self.n_landmarks * 2))

        for i in range(0,2*self.n_landmarks,2):
            rhs[i,i] = self.V
            rhs[i+1,i+1] = self.V

        return Function('V_t', [obs], [rhs], ['obs'], ['V_t'])



    def get_V_lm_none(self, obs):
        rhs = self.z.squared(SX.zeros(self.n_landmarks * 2, self.n_landmarks * 2))

        for i in range(0,2*self.n_landmarks,2):
            rhs[i,i] = 0
            rhs[i+1,i+1] = 0

        return Function('V_t', [obs], [rhs], ['obs'], ['V_t'])


    def range_bearing_observation(self,mu):      
        rhs = cat.struct_SX(self.z)
                                  
        for lm_i in range(1,self.n_landmarks+1):
            
            delta = vertcat(mu['lm' + str(lm_i) + 'x'], mu['lm' + str(lm_i) + 'y']) - vertcat(mu['x'], mu['y'])
            
            q = delta.T @ delta
            sq = sqrt(q)
            z_theta = arctan2(delta[1], delta[0])

            #range
            rhs['lm' + str(lm_i) + '_range'] = sq
            #bearing    
            rhs['lm' + str(lm_i) + '_bearing'] =  z_theta - mu['theta']


        return Function('h', [mu], [rhs], ['x'],['z_t'])


    def identity_observation(self,mu):
        rhs = cat.struct_SX(self.z)
                                  
        for lm_i in range(1,self.n_landmarks+1):
            #x
            rhs['lm' + str(lm_i) + '_x'] = mu['lm' + str(lm_i) + 'x'] - mu['x']
            #y  
            rhs['lm' + str(lm_i) + '_y'] = mu['lm' + str(lm_i) + 'y'] - mu['y']

        return Function('h', [mu], [rhs], ['x'],['z_t'])
    

    def range_bearing_measurement_noise(self, z_t, sigma_r, sigma_phi):
        noise = DM.zeros(z_t.shape)

        for i in range(0, z_t.shape[0],2):
            noise[i]   = np.random.randn() * sigma_r**2
            noise[i+1] = np.random.randn() * sigma_phi**2

        return DM(noise)
    
    def get_R_lm(self,mu, R_diag):       
        rhs = self.z.squared(SX.zeros(self.nz, self.nz))

        for i in range(0,2*self.n_landmarks,2):
            rhs[i,i] = R_diag[0]
            rhs[i+1,i+1] = R_diag[1]

        return Function('R_t', [mu], [rhs], ['x_t'], ['R_t'])

    def get_R_lm_gaze_dependent(self, mu, R_diag):
        rhs = self.z.squared(SX.zeros(self.nz, self.nz))
        heading_dir  = vertcat(cos(mu['theta']), sin(mu['theta']))

        for i in range(0,2*self.n_landmarks,2):
            landmark_dir = vertcat(mu['lm' + str(i//2 + 1) + 'x'] - mu['x'], mu['lm' + str(i//2 + 1) + 'y'] - mu['y'])
            
            #calculate distance to landmark 
            landmark_dist =  norm_2(landmark_dir)

            #calculate angle 
            cos_angle = dot(heading_dir,  (landmark_dir / landmark_dist))      

                            #distance to landmark  #maximal noise              #minimal noise
            rhs[i,i]     = 0.05**2  +  (landmark_dist * (R_diag[1] * (1-cos_angle) + R_diag[0]))**2 #range noise
            rhs[i+1,i+1] = np.deg2rad(0.5)**2 +  (log(landmark_dist+1)  * (R_diag[3] * (1-cos_angle) + R_diag[2]))**2 #bearing noise


        return Function('R_t', [mu], [rhs], ['x_t'], ['R_t'])


    #Partial sensing via SDF 
    def delta_j_triangle(self, mu, fov, height, alpha, obs):
        #make the belief fov slightly smaller to avoid boundary conflict
        fov = np.deg2rad(fov - 5.0)
        height = height
        width = height * np.tan(fov/2)

        #define triangle
        q = vertcat(width, height)

        #compute the signed distance for a triangular sensing region 
        rhs = SX.eye(self.nz)

        #set agent rotation
        R = SX.zeros(2,2)
        c = cos(- mu['theta'] + pi/2)
        s = sin(- mu['theta'] + pi/2)

        R[0,0] =   c
        R[0,1] = - s
        R[1,0] =   s
        R[1,1] =   c

        #get all landmarks 
        lms = mu.keys()[3:]

        for i in range(0,2*self.n_landmarks,2):
            p = vertcat(mu[lms[i]], mu[lms[i+1]])

            #rotate and translate the point 
            p = p - vertcat(mu['x'], mu['y'])
            p = R @ p 

            p[0] = fabs(p[0])

            a = p - q * mmin(vertcat(mmax(vertcat(dot(p,q) / dot(q,q), 0)), 1))
            b = p - q * vertcat(mmin(vertcat(mmax(vertcat(p[0] / q[0],0)), 1)), 1.0)

            k = sign(q[1])
            d = mmin(vertcat(dot(a,a), dot(b,b)))
            s = mmax(vertcat(k * (p[0] * q[1] - p[1] * q[0]), k * (p[1] - q[1])))

            sd = sign(s) * sqrt(d)

            sd_a = sigmoid(sd, alpha) * obs[i//2]

            rhs[i,i] = sd_a
            rhs[i+1,i+1] = sd_a

        return Function('delta_j_triangle', [mu, alpha, obs], [rhs], ['mu', 'alpha', 'obs'], ['delta_x'])


    def get_delta_j(self, mu, obs):
        rhs = SX.eye(self.nz)

        for i in range(0,2*self.n_landmarks,2):
            rhs[i,i] = obs[i//2]
            rhs[i+1,i+1] = obs[i//2]

        return Function('delta_j_obs', [mu, obs], [rhs], ['mu', 'obs'], ['delta_x'])


    def Belief_Propagation(self,x_t, Sigma_t, u_t, alpha, dt, obs):      
        #init new belief 
        x_next = x_t + self.F(x_t, u_t, dt)

        #linearization of motion model and observation model
        A_t = SX.eye(2*self.n_landmarks+3) + self.F_j(x_t, u_t, dt, 1)[0]
        H_t = self.H_j(x_next, 1)

        #obtain process and measurement noise 
        R_t = self.obs_cov(x_next)
        Q_t = self.process_cov(u_t)
        G_t = self.process_cov_j(x_t, u_t, dt) 

        #Predict Covariance (based on Update Step)
        S_next_ = A_t @ Sigma_t @ A_t.T + G_t @ Q_t @ G_t.T

        #Make ML Observation (which dimensions can be measured?)
        delta_t = self.d_j(x_next, alpha, obs)

        #Kalman gain
        K_t = S_next_ @ H_t.T @ delta_t @ inv(delta_t @ H_t @ S_next_ @ H_t.T @ delta_t + R_t) @ delta_t

        #Belief State Evolution (assume maximum likelihood observations)
        mu_next = x_next 
        Sigma_next = ((SX.eye(self.nx) - K_t @ H_t) @ S_next_)

        if self.representation_type is not None:
            Sigma_next[3:,3:] += self.representation_cov(obs) #add representation_noise

        return Function('belief_propagation', [x_t, Sigma_t, u_t, alpha, dt, obs], [mu_next, Sigma_next],
                        ['x_t', 'Sigma_t', 'u_t', 'alpha', 'dt', 'observable'], ['mu_next', 'Sigma_next'])
