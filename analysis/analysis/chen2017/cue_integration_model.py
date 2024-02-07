import copy 
import numpy as np 
import pandas as pd 

from analysis.chen2017.utils import recover_relative_segment_pos

from analysis.chen2017.preprocessing import * 


trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}

################################################################################################
                                    ### Full Model  ###
################################################################################################

def chen_2017_cue_integration_model(data, remove_outliers = False):

    '''
    Preprocessing of the data according to Chen 2017
    '''
    data_recentered, data_left, data_right = preprocess_data_chen_2017(data, remove_outliers = remove_outliers, recenter = True, return_both_sides=True)
    S2s = compute_response_variance(data_recentered)

    rr_vision, rr_motion = compute_relative_reliability(S2s) #weights sum to 1.0

    '''
    The calculation of response relative proximity needs to take 
    into account any intrinsic bias in the response distribution. 
    Therefore we computed the response relative proximity for each side first, 
    then took the mean as the final estimate.
    '''

    rp_combined_total = []
    rp_conflict_total = []

    for data_side in [data_left, data_right]:
        mean_responses = extract_mean_response_locations(data_side)
        d = relative_distances(mean_responses)

        rp_combined, rp_conflict = relative_response_proximity(d)

        rp_combined_total.append(rp_combined)
        rp_conflict_total.append(rp_conflict)

    #final estimate 
    rp_combined = np.mean(rp_combined_total)
    rp_conflict = np.mean(rp_conflict_total)

    '''
    The response relative proximities computed for each side were then averaged across sides. 
    This index estimates the observed weight assigned to visual cues based upon mean stopping points 
    in the homing task. According to Bayesian theory, response relative proximity to the visually defined 
    location should be equal to cue relative reliability of visual cues; 
    that is, the observed visual weight should be equal to the predicted visual weight.
    '''


    S2s['cue-integration'] = optimal_cue_integration_prediction(S2s)

    #todo apply for both sides
    #Sd2s['cue-alternation'] = alternation_model_prediction(S2s, rr_vision, mean)
    
    results = {'S2' : S2s,
               'S' : {condition:np.sqrt(variance) for (condition, variance) in S2s.items()}, 
               'data' : {'recentered' : data_recentered, 'left' : data_left, 'right': data_right}, 
               'rr' : {'vision' : rr_vision, 'motion' : rr_motion},
               'rp' : {'combined' : rp_combined, 'conflict' : rp_conflict}
              }
    
    return results


def compute_sds_per_vp(dataset, environment, kind = 'Chen 2017'):
    
    #dataset_empirical.loc[:,'VPCode'] = 'None'
    VPs = dataset.VPCode.unique()

    dfs = []
    for VP in VPs:
        data = dataset[dataset.VPCode == VP]

        try: 
            results = chen_2017_cue_integration_model(data, remove_outliers=True)
        except:
            continue 

        order = {'self-motion' : 0, 'landmark' : 1, 'combined' : 2, 'conflict' : 3, 'cue-integration' : 4}
        for condition in results['S'].keys():
                dfs.append(pd.DataFrame({'environment' : environment , 'VPCode' : VP, 'condition' : condition, 'value' : results['S'][condition], 'metric' : 'SD', 'order' : order[condition]},  index = [VP]))


    sds_per_vp = pd.concat(dfs).reset_index()[['environment', 'VPCode', 'condition', 'value', 'order']]

    sds_obs = sds_per_vp
    sds_obs['metric'] = 'SD'
    sds_obs['kind'] = kind 
    sds_obs = sds_obs.sort_values('order', ascending = True)
    
    return sds_obs 


def compute_cue_weights(dataset):
    VPs = dataset.VPCode.unique()

    subjects = []
    predicted_weights = []
    empirical_weights = []
    
    for VP,data in dataset.groupby('VPCode'):
        try:
            results = chen_2017_cue_integration_model(data, remove_outliers=False)
        except:
            continue
        
        subjects.append(VP)
        predicted_weights.append(results['rr']['vision'])
        #empirical_weights.append(np.array([results['rp']['conflict'] + results['rp']['combined']]) / 2)
        empirical_weights.append(np.array(results['rp']['conflict']))
    
    
    return subjects, predicted_weights, empirical_weights

################################################################################################
                                    ### Key Equations  ###
################################################################################################

#compute response variability (equation 9)
def compute_response_variance(normalized_data):
    #Chen Formula 
    S2s = {}
    for condition, data in normalized_data.groupby('condition'):
        d = data['euclidean_distance'].values 
        n = d.shape[0]
        
        #must be n-1!!!
        S2s[condition] = np.sum(d**2)/(n-1)
        
    return S2s 

# relative relative reliability (equation 10)
def compute_relative_reliability(S2s):
    rr_vision = S2s['self-motion'] / (S2s['landmark'] +  S2s['self-motion']) #predicted weight on the visual cue 
    rr_motion = S2s['landmark'] / (S2s['landmark'] +  S2s['self-motion'])
    
    return rr_vision, rr_motion 

# compute mean responses 
def extract_mean_response_locations(normalized_data):
    mean_locations = {}
    for condition, data in normalized_data.groupby('condition'):
        mean_locations[condition] = np.mean(data[['x','y']].values, axis = 0)
    return mean_locations

'''
Euclidean distances between single-cue distributions and double cue distributions
''' 

def relative_distances(mean_response):
    #requires mean responses from uncentered data
    d = {}

    #combined condition
    d['motion_combined'] = np.linalg.norm(mean_response['self-motion'] - mean_response['combined'])
    d['vision_combined'] = np.linalg.norm(mean_response['landmark'] - mean_response['combined'])
    
    #conflict condition (this is wrong in my estimation)
    d['motion_conflict'] = np.linalg.norm(mean_response['self-motion'] - mean_response['conflict'])
    d['vision_conflict'] = np.linalg.norm(mean_response['landmark'] - mean_response['conflict'])
    
    #alternatively (#this however does not account for intrinsic biases)
    #conflict_target = np.array([0.55646095, -0.07325947])
    
    d['motion_conflict'] = np.linalg.norm(mean_response['self-motion'] - mean_response['conflict'])
    d['vision_conflict'] = np.linalg.norm((mean_response['landmark']  + np.array([0.55646095, -0.07325947])) - mean_response['conflict']) 
    
    #d['vision_conflict'] = mean_response['conflict']
    
    
    return d 

'''
Response relative proximity to the visually defined location was calculated as the
inverse of the relative distance,
'''
#response relative proximity to visually defined locations (equation 11 and 12)
def relative_response_proximity(d):
    rp_combined = d['motion_combined'] / (d['motion_combined'] + d['vision_combined'])
    rp_conflict = d['motion_conflict'] / (d['motion_conflict'] + d['vision_conflict'])
    
   
    #rp_conflict = recover_relative_segment_pos(d['vision_conflict'])
    #rp_conflict = np.clip(rp_conflict, 0,1)
    
    return rp_combined, rp_conflict


#equation 13 
def optimal_cue_integration_prediction(S2s):
    #optimal variance (equation 13) (apply to both conflict and combined condition)
    return S2s['landmark'] * S2s['self-motion'] /  (S2s['landmark'] + S2s['self-motion'])

#equation 15 
def alternation_model_prediction(S2s, rr_vision, mean_response):
    return ((1-rr_vision) * (mean_response['self-motion'])) + rr_vision * (mean_response['landmark'] + S2s['landmark']) - ((1-rr_vision) * mean_response['self-motion'] + rr_vision * mean_response['landmark'])**2 