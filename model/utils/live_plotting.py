from datetime import datetime

from utils.plots import *
from utils.animation import get_ellipse_params,update_fov
from utils.helpers import get_landmark_mu, get_landmark_sigma

class LivePlot(object):       
    def __init__(self, mu_t, Sigma_t, x_t, true_tr, belief_tr, planned_tr, mpc_iter, experiment, showLandmarkSigma = True, true_lm_pos = None, start_point = np.array([1,2]), idealized_trajectory = '1-4-8', jitter = (0.0), fov = 100, arena_limits = ((-3,3), (-5,5))):
        self.landmark_positions = []
        self.landmark_sigmas = []
        self.n_landmarks = (Sigma_t.shape[0] - 3) // 2 
        
        self.experiment = experiment 

        x_extend = np.abs(arena_limits[0][0] - arena_limits[0][1])
        y_extend = np.abs(arena_limits[1][0] - arena_limits[1][1])

        #standardize live_plotting to x-side of 10 
        fig, ax = plt.subplots(figsize=(10, y_extend/x_extend * 10))
        ax.set_xlim(arena_limits[0])
        ax.set_ylim(arena_limits[1])
        ax.set_aspect('equal')
        
        if x_t is not None:
            self.position_x_t , ____ = plot_positional_belief(x_t[:2], np.zeros((2,2)))
        
        #Belief State
        self.position, self.e = plot_positional_belief(mu_t[:2], 0.5 * np.eye(2)) 
        self.text = ax.text(s = 'iteration:' + str(mpc_iter), x = 0.65, y = 0.85, transform = fig.transFigure)
        
        if Sigma_t is not None:
            width, height, angle =  get_ellipse_params(mu_t[:3],Sigma_t[:2,:2])
            self.e.width = width
            self.e.height = height
            self.e.angle = angle 
            
        if showLandmarkSigma:
            for lm_id in range(self.n_landmarks):
                lm_mu = get_landmark_mu(mu_t, lm_id)
                lm_sigma = get_landmark_sigma(Sigma_t, lm_id)

                position, e_sigma = plot_positional_belief(lm_mu, 0.1 * np.eye(2)) 

                self.landmark_positions.append(position)
                self.landmark_sigmas.append(e_sigma)
                
                #if lm_id > 3:
                #    position.set_visible(False)
                #    e_sigma.set_visible(False)

                width, height, angle = get_ellipse_params(lm_mu,lm_sigma)
                e_sigma.width = width
                e_sigma.height = height
                e_sigma.angle = angle 

        ### Show the true trajectory so far (HISTORY)
        self.true_tr = None 
        if true_tr is not None: 
            self.true_tr, = ax.plot(true_tr[:,0], true_tr[:,1], color='black')
            
        ### Show the belief trajectory so far (HISTORY)
        self.belief_tr = None
        if belief_tr is not None:
            self.belief_tr, = ax.plot(belief_tr[:,0], belief_tr[:,1], color = 'grey', ls = 'dotted')

        ### Show the current trajectory proposed by the planner 
        #self.planned_tr = None 
        #if planned_tr is not None:
        
        #init planned_tr 
        self.planned_tr,  = ax.plot(belief_tr[:,0], belief_tr[:,0], color='grey', ls = '--')
            
        #show the start point
        if start_point is not None:
            ax.scatter(start_point[0], start_point[1], marker = 's', s = 50)


        self.landmarks = []
            
        #plot real_landmarks  
        for i in range(0,len(true_lm_pos[0]),2):
            landmark = ax.scatter(true_lm_pos[0][i], true_lm_pos[0][i+1], marker = '*', s = 250)
            self.landmarks.append(landmark) 
                
        for i in range(0,len(true_lm_pos[1]),2):
            pickup = ax.scatter(true_lm_pos[1][i], true_lm_pos[1][i+1], marker = 'D', s = 50)
            self.landmarks.append(pickup)

        self.fov = fov
        
        ### Plot the FOV of the Agent 
        left, right = plot_agent_fov(x_t[:3].reshape(3,1), fov, 500)
        self.left_fov = left
        self.right_fov = right 
        
        #TODO real vs belief FOV 
        left, right = plot_agent_fov(mu_t[:3].reshape(3,1), fov, 500, color = 'grey', ls = 'dotted')
        self.left_belief_fov = left
        self.right_belief_fov = right 
        
        
        #measurements
        self.measurement_buffer = []

        #plot idealized Trajectory 
        if idealized_trajectory is not None:
            plot_idealized_trajectory(self.experiment, trajectory_id = idealized_trajectory, start = start_point, ax = ax, jitterx = jitter[0], jittery =jitter[1])


                
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        
        ax.set_title('Model Predictive Control /w Belief Space Planning')
        
        self.ax = ax
        self.fig = fig
        
        
    def update(self, mu_t, Sigma_t, x_t, true_tr, belief_tr, planned_tr, mpc_iter, measured_coords = None, showLandmarkSigma = True):    
        self.position.set_offsets(mu_t[:2])
        self.e.set_center(mu_t[:2])
        
        if x_t is not None:
            self.position_x_t.set_offsets(x_t[:2])
            
            #set landmark offsets (i.e. when rotated)
            for i, lm in enumerate(self.landmarks):
                lm.set_offsets(x_t[3+2*i:3+2*i+2])
                
        if Sigma_t is not None:
            width, height, angle =  get_ellipse_params(mu_t[:3].reshape(3,1),Sigma_t[:2,:2])
            self.e.width = width
            self.e.height = height
            self.e.angle = angle 

            if showLandmarkSigma:
                 for lm_id in range(self.n_landmarks):
                    lm_mu = get_landmark_mu(mu_t, lm_id)
                    lm_sigma = get_landmark_sigma(Sigma_t, lm_id)
                    
                    e_position = self.landmark_positions[lm_id]
                    e_sigma = self.landmark_sigmas[lm_id]
                    
                    e_position.set_offsets(lm_mu[:2])
                    e_sigma.set_center(lm_mu[:2])

                    width, height, angle = get_ellipse_params(lm_mu,lm_sigma)
                    e_sigma.width = width
                    e_sigma.height = height
                    e_sigma.angle = angle 
                    
                    if lm_sigma.max() < 100: 
                        e_position.set_visible(True)
                        e_sigma.set_visible(True)
                    else:
                        e_sigma.set_visible(False)

        
        #update the trajectories 
        if measured_coords is not None:
        #clear lines from plot
            for line in self.measurement_buffer:
                line[0].remove()   
                
            #reset
            self.measurement_buffer.clear()
            
            for m in measured_coords:
                line = plot_measurement(x_t[:3].reshape(3,1), m)
                self.measurement_buffer.append(line) #append to plotted lines
        
        #HISTORY 
        if true_tr is not None:
            self.true_tr.set_data(true_tr[:,0], true_tr[:,1])
        
        #HISTORY
        if belief_tr is not None:
            self.belief_tr.set_data(belief_tr[:,0], belief_tr[:,1])

        if (planned_tr is not None) and (self.planned_tr is not None): 
            self.planned_tr.set_data(planned_tr[:,0], planned_tr[:,1])

        #update MPC iteration 
        self.text.set_text('iteration: ' + str(mpc_iter))
        
        #update FOV 
        update_fov(self.left_fov,self.right_fov, x_t[:3].reshape(3,1), self.fov, 500)
        update_fov(self.left_belief_fov,self.right_belief_fov, mu_t[:3].reshape(3,1), self.fov, 500)


    def draw_figure(self, waittime = 0.001):    
        plt.pause(waittime)
        self.fig.canvas.draw()

    def save_figure(self, fname):    
        self.fig.savefig(fname)

class Plotting:
    from utils.live_plotting import LivePlot

    def __init__(self):
        self.liveplot = None 
        
    def update_plot(self, mpc_iter, experiment, logger, curr_plan, params, nz, show_idealized = True):
        idealized = None 
        start = None 
        
        if show_idealized:
            idealized = params['trajectory_id']
            start = params['route'][0]
            

        if mpc_iter == 0:
            lms = np.array(logger.xs[0][3:]).reshape(nz)   
            self.liveplot = LivePlot(logger.mus[-1], logger.sigmas[-1], 
                                        logger.xs[-1], 
                                        np.array(logger.xs), 
                                        np.array(logger.mus), 
                                        curr_plan,
                                        0, 
                                        experiment,
                                        True, 
                                        fov = params['fov'],
                                        true_lm_pos = (params['landmark_locations'], params['pickup_locations']),
                                        start_point = start,
                                        idealized_trajectory = idealized,
                                        jitter = (params['target_jitter'][0], params['target_jitter'][1]),
                                        arena_limits = params['arena_limits'])
        else:
            self.liveplot.update(logger.mus[-1], 
                            logger.sigmas[-1], 
                            logger.xs[-1], 
                            np.array(logger.xs), 
                            np.array(logger.mus), 
                            curr_plan, 
                            mpc_iter, 
                            logger.measured_coords[-1])

        return self.liveplot
    
    def save_figure(self, fname):
        """Saves the current state of liveplot as figger to file 

        Args:
            fname (string): path + fname 
        """        

        self.liveplot.save_figure(fname)


def plot_intermediate_step(experiment, df, logger, params, fname):
    x_true = logger.xs[0].reshape(logger.xs[0].shape[0],1)
    plt.ioff()
    plt.figure(figsize = (6,10))
    ax = plt.gca()

    plt.plot(df['x'], df['z'])

    plt.xlim(-3,3)
    plt.ylim(-5,5)

    plot_positional_belief(logger.xs[-1], logger.sigmas[-1])
    plot_agent_fov(logger.xs[-1].reshape(logger.xs[-1].shape[0],1), 60, 100)
    

    plot_idealized_trajectory(experiment, params['trajectory_id'], params['route'][0], jitterx = params['target_jitter'][0], jittery = params['target_jitter'][1], ax = ax)
    
    
    plt.scatter(x_true[3,0], x_true[4,0], marker = '*')

    plt.savefig(fname)
    plt.close()
