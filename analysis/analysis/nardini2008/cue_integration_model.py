import copy 
import pandas as pd
import numpy as np 

order = {'self-motion' : 0, 'landmark' : 1, 'combined' : 2, 'conflict' : 3, 'cue-integration' : 4}

################################################################################################
                                    ### HELPERS  ###
################################################################################################

def compute_SDs_per_condition(data):
    temp_dfs = []
    for condition, cond_data in data.groupby('condition'):
        responses = cond_data[['x','y']].values
        mean_response = np.mean(responses, axis = 0)

        condition_sd = np.sqrt(np.sum((responses - mean_response) ** 2) / (responses.shape[0] - 1))

        temp_dfs.append(pd.DataFrame({'VPCode' : cond_data['VPCode'].iloc[0], 
                          'condition' : condition,
                          'x' : mean_response[0],
                          'y' : mean_response[1],              
                          'SD' : condition_sd,
                          'kind' : 'empirical',
                          'order' : order[condition]}, index = [0]))

    SDs = pd.concat(temp_dfs)

    return SDs 


def compute_search_distances(normalized_data):
    pooled_centroid = np.mean(normalized_data, axis = 0)
    search_distances = np.linalg.norm(normalized_data - pooled_centroid, axis = 1)
    return search_distances 


def compute_search_distance_by_condition(normalized_data):
    dfs = []
    for condition, data in normalized_data.groupby('condition'):
        d = compute_search_distances(data[['x','y']].values)
        data.loc[:,'euclidean_distance'] = d    
        dfs.append(copy.deepcopy(data))
        
    return pd.concat(dfs)


################################################################################################
                                    ### Key Equations  ###
################################################################################################

def extract_mean_response_locations(SDs):
    mean_response_by_condition = {}

    for condition, data in SDs.groupby('condition'):
        mean_response = np.mean(data[['x', 'y']].values, axis = 0)
        mean_response_by_condition[condition] = mean_response

    return mean_response_by_condition

def extract_variance(SDs):
    variance = {}

    for condition, data in SDs.groupby('condition'):
        variance[condition] =  data['SD'].values**2
    
    return variance

def compute_d_chen_formula(mean_locations):
    d_lm = np.linalg.norm(mean_locations['landmark'] - mean_locations['conflict'])
    d_sm = np.linalg.norm(mean_locations['self-motion'] - mean_locations['conflict'])
    
    return d_lm, d_sm 

def compute_d_nardini_formula(mean_locations, rotated_target_location = np.array([ 0.45293333 -0.0596298 ])):
    d_lm = np.linalg.norm(rotated_target_location -  mean_locations['conflict'])
    d_sm = np.linalg.norm(np.zeros(2) -  mean_locations['conflict'])
                              
    return d_lm, d_sm

#equation 1
def relative_proximity(d_lm, d_sm): 
    rprox_lm = d_sm / (d_sm + d_lm)
    rprox_sm = 1 - rprox_lm
    return rprox_lm, rprox_sm

#equation 2 
def predicted_variance_cue_combination(sigma_lm, sigma_sm, w_lm, w_sm):
    sigma_sm_lm = w_lm**2 * sigma_lm + w_sm**2 * sigma_sm 
    return sigma_sm_lm

#equation 3
def predicted_variance_cue_alternation(p_sm, p_lm, mu_sm, sigma_sm, mu_lm, sigma_lm):
    # in the model mu_sm = 0 and mu_lm = 46 -- because landmark rotations shifts it by 46cm
    sigma_sm_lm = p_sm * (mu_sm**2 + sigma_sm) + p_lm * (mu_lm**2 + sigma_lm) - (p_sm * mu_sm + p_lm * mu_lm)**2 
    return sigma_sm_lm
        
#equation 4
def calculate_cue_weights(sigma_lm, sigma_sm): 
    w_lm = sigma_sm / (sigma_lm + sigma_sm)
    w_sm = 1 - w_lm 
    return w_lm, w_sm  

################################################################################################
                                    ### Full Model  ###
################################################################################################


def cue_integration_nardini2008_model(normalized_data):    
    #extract mean_lm, mean_sm, mean_conflict, sigma_lm, sigma_sm
    SDs = compute_SDs_per_condition(normalized_data)
    mean =  extract_mean_response_locations(SDs)
    sigma = extract_variance(SDs)
    
    #d_lm, d_sm = compute_d_chen_formula(mean)
    d_lm, d_sm = compute_d_nardini_formula(mean)

    #calculate relative proximity (equation 1)
    rprox_lm, rprox_sm = relative_proximity(d_lm, d_sm)
                              
    #calculate cue weights (equation 4)
    w_lm, w_sm = calculate_cue_weights(sigma['landmark'], sigma['self-motion'])
                              
    #predict reduced variance (equaton 2)
    sigma_sm_lm_combined = predicted_variance_cue_combination(sigma['landmark'], sigma['self-motion'], w_lm, w_sm)

    #predicted variance_cue alternation (equation 3)
    sigma_sm_lm_alternation = predicted_variance_cue_alternation(rprox_sm, rprox_lm, 0, sigma['self-motion'], 0.46, sigma['landmark'])
    
    sigma['cue-integration'] = sigma_sm_lm_combined
    sigma['cue-alternation'] = sigma_sm_lm_alternation
    
    d = {'d_lm' : d_lm, 'd_sm': d_sm}
    w = {'w_lm' : w_lm, 'w_sm' : w_sm}
    rprox = {'rprox_lm' : rprox_lm, 'r_prox' : rprox_sm}
    
    
    results = {'mean_response_location' : mean, 
               'sigmas' : sigma,
               'd' : d,
               'w' : w, 
               'rprox_' : rprox
              }

    return SDs, results