import copy 
import numpy as np 
import pandas as pd 

from analysis.chen2017.utils import * 

trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}

################################################################################################
                                    ### Preprocessing  ###
################################################################################################


def preprocess_data_chen_2017(dataset, recenter = False, remove_outliers = True, return_both_sides = False, iqr = 3):
    #compute target_loc indices 
    target_loc = dict(zip(np.unique(dataset['target_loc']), (1,2,3,4)))

    '''
    First, for each of the four target locations, responses were transformed 
    into a spatial coordinate system with the target location as the origin, 
    the correct walking direction as the y-axis, and the orthogonal direction as the x-axis.
    '''

    normalized_data = normalize_by_post_location(dataset, target_loc, flip_conflict = False)


    '''
    Second, responses for the two target locations on the same side were pooled together. 
    Outliers were defined as responses whose distance from their respective response 
    centroids exceeded the 3rd quartile by 3 times the interquartile range 
    (because of the small numbers of responses collected on each side, 
    these quartiles were computed using responses from all four target locations)
    '''

    #recompute euclidean search distances for each side 
    dfs = []
    for side, side_data in normalized_data.groupby('target_loc'):
        dfs.append(compute_search_distance_by_condition(side_data))
        
    normalized_data = pd.concat(dfs)

    #remove outliers (3x IQR)
    if remove_outliers:
        dfs = []

        #compute quartiles across all 4 target locations (not sure if quantiles are computed within each condition ??)
        q = compute_quantiles(normalized_data, 'euclidean_distance')

        #outlier deletion for each side 
        for condition, condition_data in normalized_data.groupby('side'):
            #and each condition
            #for side, side_data in condition_data.groupby('condition'):
            
            #(because of the small numbers of responses collected on each side, 
            # these quartiles were computed using responses from all four target locations).
            q = compute_quantiles(condition_data, 'euclidean_distance')
            dfs.append(subset_by_iqr(condition_data, 'euclidean_distance', q = q, whisker_width=iqr))

        normalized_data = pd.concat(dfs)

    '''
    Third, before pooling responses from the two sides, 
    we corrected the bias in each side’s response distribution by centering each distribution over its 
    new centroid after outlier deletion. This was done to eliminate disparity in the centroids 
    of the distributions on the two sides, which would increase the variability of the pooled distribution artificially.
    '''
    left_side =  normalized_data[normalized_data.side == -1]
    right_side = normalized_data[normalized_data.side == 1]

    

    '''
    Fourth, after pooling the two sides’ response distributions, we computed the response variance
    '''
   
    if recenter:
        #recenter the distribution on both sides 
        normalized_data_left  = recenter_data_by_condition(left_side)
        normalized_data_right = recenter_data_by_condition(right_side)
        pooled_data = pd.concat([normalized_data_left, normalized_data_right])
        
        #pool the data 
        pooled_data[['x', 'y']] = pooled_data[['x_recentered', 'y_recentered']]
    else:
        pooled_data = normalized_data 

    pooled_data = pooled_data[['condition','VPCode', 'x', 'y', 'target_loc', 'side', 'euclidean_distance']]

    #recompute euclidean search distances 
    pooled_data = compute_search_distance_by_condition(pooled_data)
    
    if return_both_sides:
        return pooled_data, left_side, right_side 
    
    return pooled_data


def preprocess_per_participant(dataset):
    #preprocess data per participant 
    dfs_data = []
    dfs_sds= []

    #for each participant preprocess data 
    for VPCode, data in dataset.groupby('VPCode'):
        #if VPCode == 'Aaron': continue 
    
        pooled_data_vp = preprocess_data_chen_2017(data, recenter=False, return_both_sides = False)
        pooled_data_vp['VPCode'] = VPCode

        sd_vp = compute_SDs_Chen(pooled_data_vp)
        sd_vp['VPCode'] = VPCode

        dfs_data.append(pooled_data_vp)
        dfs_sds.append(sd_vp)


    #pooled vp_data
    pooled_data_per_vp = pd.concat(dfs_data)

    #response variability p
    sds_per_vp = pd.concat(dfs_sds)

    
    return pooled_data_per_vp, sds_per_vp