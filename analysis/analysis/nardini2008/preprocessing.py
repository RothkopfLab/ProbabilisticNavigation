
from analysis.nardini2008.utils import * 


def preprocess_nardini_2008(dataset, remove_outliers = True, normalize = True, flip_conclict = False):
    
    '''
    Before calculation of each participant’s RMSE for a condition (Figure 2A), 
    individual trial data were filtered to remove errors that were extreme outliers 
    in the distribution of all errors recorded for that age and condition.

    ‘‘Extreme outliers’’ were defined as errors greater than the third quartile 
    plus three times the interquartile range [30]; in a normal distribution, fewer than 0.0002% of values 
    would meet this criterion.
    '''
    
    #if remove_outliers:
    #    dataset.loc[:,'euclidean_distance'] = euclidean_dist(dataset)
    #    dataset= subset_by_iqr(dataset, 'euclidean_distance', whisker_width=3)
    
    '''
    To analyze response variance (Figure 2B), each participant’s four responses in a 
    condition were first standardised to account for the fact that they were aiming for 
    four different locations in the room (Figure 1D, lo- cations) 
    at which the following of landmarks (in the conflict condition)
    would predict different angles of shift relative to the room. 
    
    Response coordinates were transposed and rotated so that they 
    could be considered as responses around a single point, 
    at which the direction of the correct return path to the target is 
    ‘‘north’’ and the axis of landmark rotation is orthogonal.
    '''
    
    if normalize:
        normalized_data = normalize_data_by_condition(dataset, normalize_by = 'nardini')
    else:
        normalized_data = dataset
    
    '''
    For conflict trials, x coordinates of search were 
    inverted for anticlockwise-rotation participants, 
    so that the direction of landmark rotation was always ‘‘east.’’
    '''
    
    if flip_conclict: 
        #not valid if we do anti-clockwise rotations in our model too 
        normalized_data.loc[normalized_data.condition == 'conflict', 'x'] = -normalized_data.loc[normalized_data.condition == 'conflict', 'x']
    
    '''
    A participant’s standard deviation for a condition was 
    then calculated on the basis of the participant’s searches’ 
    distances from his or her mean search location for that condition.
    '''
    
    #recompute euclidean distances 
    normalized_data = compute_search_distance_by_condition(normalized_data)
    
    
    ''' 
    Extreme outliers were removed as in the analysis of RMSE, 
    based on distance from the grand mean response location for each age and condition. 
    '''
    
    if remove_outliers:
        #outlier deletion post normalization 
        print(len(normalized_data))
        dfs = []
        for condition, data in normalized_data.groupby('condition'):
            response_mean = np.mean(data[['x', 'y']].values ,axis = 0)
            data.loc[:,'euclidean_distance'] = np.sqrt((data['x'].values - response_mean[0])**2 + (data['y'].values - response_mean[1])**2)
            data = subset_by_iqr(data, 'euclidean_distance', whisker_width=3)
            dfs.append(data)

        normalized_data = pd.concat(dfs)
        print(len(normalized_data))
    
    SDs = compute_SDs(normalized_data, return_dict = False)
        
    return normalized_data, SDs  


def preprocress_per_participant(dataset):
    pass 