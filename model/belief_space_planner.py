from casadi import * 
import casadi.tools as cat

import numpy as np
from numpy import matlib as mb
import matplotlib.pyplot as plt

import time
from utils.helpers import read_file, normalize_angle, get_landmark_mu



class BeliefSpacePlanner:   
    def __init__(self, model, N, verbose = False):       
        self.N = N 
        self.model = model
        self.verbose = verbose

        self.reinit = False 

        #Model Parameters 
        self.P = cat.struct_symSX([
            (
                cat.entry('alpha', shape = 1),
                cat.entry('start', shape = self.model.nx),
                cat.entry('end', shape = 3), 
                cat.entry('R', shape = self.model.R.shape), 
                cat.entry('Q', shape = self.model.Q.shape),
                cat.entry('Rtr', shape = 1),
                cat.entry('w_running_control', shape = 1),
                cat.entry('w_running_control_smoothness', shape = 1), 
                cat.entry('w_running_position_uncertainty', shape = 1),
                cat.entry('w_running_map_uncertainty', shape = 1),
                cat.entry('w_running_state', shape = 1),
                cat.entry('w_final_state', shape = 1),
                cat.entry('w_final_uncertainty', shape = 1), 
                cat.entry('dt', shape = 1),
                cat.entry('S0', shape = (self.model.nx, self.model.nx)),
                cat.entry('observable', shape = self.model.n_landmarks), 
                cat.entry('u_k_prev', shape = self.model.nu)
            )
            ]) 

        #set constraints
        self.x_max = 5000
        self.x_min = -self.x_max 
        self.y_max = 5000
        self.y_min = -self.y_max
        self.lmx_max = 5000
        self.lmy_max = 5000
        self.lmx_min = - self.lmx_max
        self.lmy_min = - self.lmy_max
        self.theta_max = np.inf
        self.theta_min = -np.inf
        
        self.setup_optimization_problem_shooting()

    def create_solver(self, warm_start = False, opts = None, BFGS = True, jit_compile = False, solver = 'ma27', jit_options = {"flags": ['-O0'], "verbose": True, "compiler": 'clang'}):     
        #create solver / planner 
        nlp_prob = {'f' : self.obj, 'x' : self.OPT_variables, 'g' : self.g, 'p' : self.P}
        
        #create default options 
        if opts is None:
            #optimization Options 
            opts = {}

            #expand symbolic tree 
            opts['expand'] = True

            #just in time c-code generation 
            opts['jit'] = jit_compile
            opts['compiler'] =  "shell"
            opts['jit_options'] = jit_options

            #print time
            opts['print_time'] = 0

            #ipopt options 
            opts['ipopt'] = {}

            #verbosity 
            if self.verbose:
                opts['ipopt']['print_level'] = 5
                opts['print_time'] = 1
            else:
                opts['ipopt']['print_level'] = 0

            #iterations
            opts['ipopt']['max_iter'] = 1500
            opts['ipopt']['tol'] = 1e-8
            opts['ipopt']['constr_viol_tol'] = 1e-5
            opts['ipopt']['compl_inf_tol'] = 1e-4

            #termination
            opts['ipopt']['acceptable_tol'] = 1e-6
            opts['ipopt']['acceptable_iter'] = 5
            opts['ipopt']['acceptable_obj_change_tol'] = 1e-4

            #quasi-newton BFGS
            if BFGS: 
                opts['ipopt']['hessian_approximation'] = 'limited-memory' 
                opts['ipopt']['limited_memory_max_history'] = self.model.nx * self.N 
                opts['ipopt']['limited_memory_max_skipping'] = 1

            if warm_start:
                opts['ipopt']['warm_start_init_point'] = 'yes'

            #choose linear solver (i.e. ma57)
            opts['ipopt']['linear_solver'] = solver
            opts['ipopt']['sb'] = 'yes' 


        if self.verbose:
            print('create solver')
            start_time = time.time()

        solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

        if self.verbose: 
            print("--- %s seconds ---" % (time.time() - start_time))
        
        
        return solver, opts 
    

    def create_objective(self, X, U, P):   
        dt = P['dt']
        Sigma_k = P['S0']

        #define running costs 
        running_control_cost = 0
        running_position_uncertainty_cost = 0
        running_map_uncertainty_cost = 0 
        running_state_cost = 0 

        #get Q, R and Rtr 
        Q = P['Q']
        R = P['R']
        Rtr = P['Rtr']

        u_k_prev = P['u_k_prev']

        #calculate running costs 
        for k in range(self.N):
            mu_k = X[:,k]
            u_k = U[:,k]

            #Belief Propagation 
            mu_next, Sigma_next = self.model.BF(mu_k, Sigma_k, u_k, P['alpha'], dt, P['observable'])
            Sigma_k = Sigma_next

            #Control Cost
            running_control_cost += u_k.T @ R * dt @ u_k + (u_k[0] * u_k[1])**2 * dt * Rtr

            #Position Uncertainty
            running_position_uncertainty_cost += trace(Sigma_k[:3,:3] @ Q) * dt

            #Map Uncertainty
            running_map_uncertainty_cost += trace(Sigma_k[3:,3:]) * dt 

            #State Cost 
            running_state_cost += (mu_next[:2]-P['end'][:2]).T @ Q[:2,:2] * dt @ (mu_next[:2]-P['end'][:2])

        #Final state cost
        final_state_cost = (X[:,-1][:2]-P['end'][:2]).T @ Q[:2,:2] @ (X[:,-1][:2]-P['end'][:2])

        #final uncertainty cost 
        #final_uncertainty_cost = trace(Sigma_k[:3,:3])


        #objective function 
        obj = P['w_final_state'] * final_state_cost + \
              P['w_running_position_uncertainty'] * running_position_uncertainty_cost  + \
              P['w_running_map_uncertainty'] * running_map_uncertainty_cost  + \
              P['w_running_control']  * running_control_cost + \
              P['w_running_state'] * running_state_cost

        return obj
    
    def create_nonlinear_constraints(self, X , U, P):
        g = []
        g2 = []
        st = X[:,0]

        #initial position matches the one speficied 
        g = vertcat(g, st-P['start'])

        u_k_prev = P['u_k_prev']

        for k in range(self.N):
            mu_k = X[:,k] #mu_k
            u_k = U[:,k]

            #belief propagation (without Sigma)
            mu_next = mu_k + self.model.F(mu_k, u_k, P['dt'])

            #obtain the next belief 
            st_next = X[:,k+1] 

            #ensure belief dynamics (multiple shooting)
            g = vertcat(g, mu_next - st_next) #compute constraint (f(x0,u0)-x1 = 0)   
            g2 = vertcat(g2, u_k - u_k_prev)  #difference in control signals has to be small between two subsequent actions
            
            u_k_prev = u_k 

        #set lower and upper bounds 
        lbg = np.zeros(self.model.nx*(self.N+1))
        ubg = np.zeros(self.model.nx*(self.N+1))

        #put constraint in on maximum difference in control signals (i.e. acceleration)
        ubg2 = np.array([ self.model.dv_max,   self.model.dw_max] * (self.N))
        lbg2 = np.array([-self.model.dv_max,  -self.model.dw_max] * (self.N))

        g = vertcat(g, g2)
        ubg = numpy.concatenate([ubg,ubg2])
        lbg = numpy.concatenate([lbg,lbg2])

        return g,ubg, lbg
    
    
    def create_box_constraints(self):     
        nx = self.model.nx 
        N  = self.N

        ##lower and upper bound of the states and controls 
        lbx = np.zeros(nx*(N+1)+2*N)
        ubx = np.zeros(nx*(N+1)+2*N)

        #state constraints 
        lbx[0:nx*(N+1):nx] = self.x_min #state x lower bound 
        ubx[0:nx*(N+1):nx] = self.x_max #state x upper bound

        lbx[1:nx*(N+1):nx] = self.y_min #state y lower bound
        ubx[1:nx*(N+1):nx] = self.y_max  #state y upper bound

        lbx[2:nx*(N+1):nx] = self.theta_min #state theta upper bound
        ubx[2:nx*(N+1):nx] = self.theta_max  #state theta upper bound

        for i in np.arange(nx - 2*self.model.n_landmarks, self.model.nx):
            lbx[i:nx*(N+1):nx] = self.lmx_min #state x lower bound 
            ubx[i:nx*(N+1):nx] = self.lmx_max  #state x upper bound

        #control constraints 

        #v
        lbx[nx*(N+1):nx*(N+1) + 2*N:2] = self.model.v_min
        ubx[nx*(N+1):nx*(N+1) + 2*N:2] = self.model.v_max

        #w 
        lbx[nx*(N+1)+1:nx*(N+1) + 2*N:2] = self.model.w_min
        ubx[nx*(N+1)+1:nx*(N+1) + 2*N:2] = self.model.w_max

        return lbx, ubx
    
    def setup_optimization_problem_shooting(self):
        if self.verbose:
            print('setup optimization problem')

        # define optimization variables 
        self.X = SX.sym('X', self.model.nx, self.N+1) # sequence of states 
        self.U = SX.sym('U', self.model.nu, self.N)   # sequence of controls 

        if self.verbose:
            print('    create shooting objective')
        #create objective function
        self.obj = self.create_objective(self.X,self.U,self.P)

        #constraints
        if self.verbose:
            print('    create non-linear constraints')
        self.g, self.ubg, self.lbg = self.create_nonlinear_constraints(self.X,self.U,self.P)

        if self.verbose:
            print('    create linear constraints')
        self.lbx, self.ubx = self.create_box_constraints()

        self.OPT_variables = vertcat(self.X.reshape((self.model.nx*(self.N+1),1)), self.U.reshape((2*self.N,1)))



    def alpha_update(self, sol ,solver, p, k, n_iter, verbose = True):             
        #get lagrange multipliers for warm start 
        lam_g = sol['lam_g']
        lam_x = sol['lam_x']
        
        args = {}

        for i in range(n_iter):

            if verbose:
                print('calculating alpha update')
                print(p['alpha'])
                print(solver.stats()['return_status'])
                print(solver.stats()['success']) 


            p['alpha'] = k * p['alpha']
            args['p'] = p
            args['x0'] = sol['x']
            sol = solver(x0 = args['x0'], lbx = self.lbx, ubx = self.ubx, lbg = self.lbg, ubg = self.ubg,  p = args['p'], lam_x0 = lam_x, lam_g0 = lam_g)

            #obtain new langrangian 
            lam_g = sol['lam_g']
            lam_x = sol['lam_x']
                        
        return sol, p

    def reinitialize_planner(self, x_res, u_res, p, x_true, mu_t, Sigma_t, u_t, curr_goal, obs_belief, params):      
        p['start'] = mu_t
        p['S0'] = Sigma_t
        p['observable'] = obs_belief
        p['alpha'] = params['a0']

        if np.isscalar(curr_goal):
            p['end'] = vertcat(np.array(get_landmark_mu(DM(mu_t), params['n_landmarks'] + curr_goal)).reshape(2,), 0.0)
        else:  
            p['end'] = vertcat(np.array(curr_goal).reshape(2,), 0.0)
            
        p['u_k_prev'] = u_t 

        args = {}

        #TODO extend for varying control horizons 
        next_x = vertcat(DM(mu_t).T, x_res[(params['control_horizon'] + 1):,:], repmat(x_res[-1,:], params['control_horizon'])).T
        next_u = vertcat(u_res[params['control_horizon']:,:], repmat(u_res[-1,:], params['control_horizon']))
        x0 = vertcat(next_x.reshape((self.model.nx*(self.N+1),1)), next_u.reshape((2*self.N,1))) 

        args['x0'] = x0
        args['p'] = p
        

        if self.reinit:
            u_res = np.ones((self.N,2)).reshape((2*self.N,1))
            x_res, Sigmas = self.forward_simulation(x_true, Sigma_t, u_res, params['dt'], p['alpha'], p['observable'], params['planning_horizon'])
            x_res = DM(x_res)
            self.reinit = False 

        
        return p, args, x_res, u_res 

    
    def obtain_solution(self, sol): 
        x = reshape(sol['x'][0:self.model.nx*(self.N+1)].T,self.model.nx,self.N+1).T
        u = reshape(sol['x'][self.model.nx*(self.N+1):].T,2,self.N).T

        return x, u


    def forward_simulation(self, x_t, Sigma_t, u_res, dt, alpha, observable, N):   
        xs = [np.array(DM(x_t)).reshape(self.model.nx,)]
        Sigmas = [np.array(DM(Sigma_t))]
        if N is None: N = self.N

        for i in range(N):
            u_t = u_res[i,:]

            #Belief Propagation 
            x_t, Sigma_t = self.model.BF(x_t, Sigma_t, u_t, alpha, dt, observable) 
            
            #get results 
            xs.append(np.array(DM(x_t)).reshape(self.model.nx,))
            Sigmas.append(np.array(DM(Sigma_t)))

        return xs, Sigmas 

       
