import numpy as np

from analysis.zhao2015.utils import * 
from analysis.zhao2015.preprocessing import * 

trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}


def cue_integration_circular(theta_lm, theta_pi, k_lm, k_pi, delta):
    
    #equation 1
    theta_combined = theta_lm + delta - np.arctan2(np.sin(delta), k_lm / k_pi + np.cos(delta))
    
    #equation 2 
    k_combined = np.sqrt(k_pi**2 + k_lm**2 + 2*k_pi*k_lm*np.cos(delta))
    
    #equation 3 
    circ_sd_combined = np.rad2deg(transform_circular_sd(k_combined))
    
    return theta_combined, k_combined, circ_sd_combined


def cue_integration_linear(vm):
    w_lm = vm['self-motion']['circ_sd']**2 / (vm['self-motion']['circ_sd']**2 + vm['landmark']['circ_sd']**2)
    w_pi = 1 - w_lm
    
    #equation 5: mean homing response direction
    theta = w_pi * vm['self-motion']['theta'] + w_lm * vm['landmark']['theta']
    #equation 6: response variability 
    sigma_2 = vm['self-motion']['circ_sd']**2 * vm['landmark']['circ_sd']**2 / (vm['self-motion']['circ_sd']**2 + vm['landmark']['circ_sd']**2)
    
    return {'w' : {'lm' : w_lm, 'pi' : w_pi}, 'theta' : theta, 'sigma_2' :  sigma_2}
    

def zhao_2015a_cue_integration_model(normalized_data):
    vm_df   = fit_von_mises_per_condition_df(normalized_data)

    #computes endpoint variability per target location if multiple are available and later averages
    vm_df['true_condition'] = vm_df['condition'].apply(lambda x : x.split('_')[0])
    vm_df = vm_df.groupby(['true_condition', 'landmark_rotation']).mean().reset_index()
    vm_df['condition'] = vm_df['true_condition']
    vm_df[['condition', 'landmark_rotation', 'theta', 'k', 'theta_deg', 'circ_sd']]

    vm_landmark = vm_df[vm_df.condition == 'landmark'].iloc[0]
    vm_sm = vm_df[vm_df.condition == 'self-motion'].iloc[0]

    
    #vm = fit_von_mises_per_condition_dict(normalized_data)
    
    ''' 
    Circular Data Analysis 
    
    Maybe a prediction can be computed for all possible landmark shifts
    '''
    
    circular_ci = []
    
    for rotation in normalized_data.landmark_rotation.unique():

        rotation = int(rotation)

        theta_combined, k_combined, circ_sd_combined = cue_integration_circular(
                                 theta_lm = vm_landmark['theta'], 
                                 theta_pi = vm_sm['theta'], 
                                 k_lm = vm_landmark['k'], 
                                 k_pi = vm_sm['k'], 
                                 delta = np.deg2rad(float(rotation))) #fixed this error here?
        
        circular_ci.append(pd.DataFrame({'condition' : 'cue-integration_' + str(rotation), 
                       'k' : k_combined, 
                       'theta' : theta_combined, 
                       'theta_rad' : np.rad2deg(theta_combined), 
                       'circ_sd' : circ_sd_combined,
                       'rotation' : rotation}, 
                        index = [9]
                       ))
        
    circular_ci = pd.concat(circular_ci) 
    vm_df = vm_df[['condition', 'k', 'theta', 'theta_deg', 'circ_sd', 'landmark_rotation']]
    vm_df['landmark_rotation'] = vm_df['landmark_rotation'].astype(int)

    ''' 
    Linear Data Analysis 
    '''
    
    
    #linear_ci = cue_integration_linear(vm)
    linear_ci = None 
    
    return {'full data' : vm_df, 'circular model' : circular_ci, 'linear model' : linear_ci} 

def evaluate_zhao_cue_integration(results, kind):
    order = {'self-motion' : 0, 'landmark' : 1, 'combined' : 2, 'conflict' : 3, 'cue-integration' : 4}

    full_data = results['full data']
    full_data = full_data[full_data.landmark_rotation <= 15]
    
    
    combined = copy.deepcopy(results['circular model'][results['circular model'].condition == 'cue-integration_0'])
    combined.loc[:,'condition'] = combined.loc[:,'condition'].apply(lambda x : x.split('_')[0])
    
    full_results = pd.concat([full_data, combined])
    full_results = full_results[['circ_sd', 'condition', 'k', 'theta']]
    full_results['theta_deg'] = np.rad2deg(full_results['theta'].values)
    full_results['kind'] = kind
    
    
    full_results['circ_sd'] = full_results['circ_sd']


    full_results['order'] = full_results.condition.apply(lambda x : order[x])
    full_results = full_results.sort_values('order', ascending = True)
    
    return full_results