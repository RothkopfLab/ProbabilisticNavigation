import matplotlib.pyplot as plt 
import numpy as np 
import pingouin as pg 
import pandas as pd 
import dcor


from analysis.utils import * 



def compare_end_point_distributions(normalized_data_simulated, normalized_data_empirical, file = 'test.txt', figsize = (12,3), nsamples = 100, p_limit = 0.05, method = 'holm'):
 

    pvals_k2 = []
    pvals_hoteling = []
    pvals_box_m = []
    pvals_energy = []

    normalized_data_simulated['kind'] = 'Model'
    normalized_data_empirical['kind'] = 'Chen 2017'


    fig, ax = plt.subplots(1,4, figsize = figsize)
    
    
    with open(file, 'w') as f:

        for i,condition in enumerate(['self-motion', 'landmark', 'combined', 'conflict']):
            print('------', file = f)
            print(condition, file = f)
            print('------', file = f)

            s1 = normalized_data_empirical[normalized_data_empirical.condition == condition]
            s2 = normalized_data_simulated[normalized_data_simulated.condition == condition]

            ax[i].scatter(s1.x, s1.y, label ='empirical')
            ax[i].scatter(s2.x, s2.y, label = 'simulated')
            ax[i].set_xlim(-3,3)
            ax[i].set_ylim(-3,3)
            ax[i].set_aspect('equal')
            if i == 0: ax[i].legend()
            ax[i].set_title(condition)

            print('------', file = f)
            print('Homogenity Energy Test', file = f)
            res = dcor.homogeneity.energy_test(s1[['x', 'y']].values, s2[['x', 'y']].values,num_resamples=nsamples)
            print(res.p_value, file = f)
            print('are the distributions different: ', res.p_value < 0.05, file = f)
            pvals_energy.append(res.p_value)


    
        print('Applying Bonferroni Correction', file = f)
        print(pg.multicomp(pvals = pvals_energy, alpha = 0.05, method = 'holm'), file = f)
    
    
    sign, pvals = pg.multicomp(pvals = pvals_energy, alpha = p_limit, method = method)
    
    
    #plotting p values in graphic 
    for i,p in enumerate(pvals):
        ax[i].text(x = -2.5, y = 2.35, s = 'p = {:}'.format(np.round(p,3)))
        
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=False, ncol=5)
    
    return (sign, pvals), (fig,ax)

