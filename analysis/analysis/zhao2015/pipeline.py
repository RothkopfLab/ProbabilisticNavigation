#import analysis functions 
from analysis.zhao2015.data_loading import load_zhao_data,load_zhao_data_simulated
from analysis.zhao2015.utils import * 
from analysis.zhao2015.preprocessing import fit_von_mises, fit_von_mises_per_condition_df, normalize_data_by_conflict_condition
from analysis.zhao2015.cue_integration_model import zhao_2015a_cue_integration_model

from analysis.key_plots import plot_endpoint_variability

import copy 
import matplotlib.pyplot as plt 
from svgutils.compose import *

colours = {'self-motion' : (0.8545098039215687, 0.1415686274509803, 0.14862745098039193),
           'landmark' : (0.24098039215686273, 0.49156862745098046, 0.6962745098039216),
           'combined' : (0.32058823529411773, 0.6664705882352941, 0.31000000000000005),
           'conflict' : (0.5837254901960787, 0.3225490196078431, 0.6225490196078431)}

colours['cue-integration'] = 'grey'
colours['a-integration'] = 'grey'
trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}

def normalize_data_per_side(side_data):
    data_empirical = []
              
    for condition, data in side_data.groupby('condition'):
        #flip x coordinates for conflict condition
        if condition == 'conflict':
            data_empirical.append(copy.deepcopy(data))
        else:
            data[['respx', 'respz']] = (data[['respx', 'respz']] - data[['respx', 'respz']].mean()) + data[['targetx', 'targetz']].values
            data_empirical.append(copy.deepcopy(data))
        
    return pd.concat(data_empirical)


def obtain_datasets(experiment, files):
    
    if experiment == 'proximal':
        dataset_empirical = load_zhao_data_simulated('../../../data/raw_data_Zhao2015a/' + 'zhao2015a_proximal_full_trial_data.csv')
    else:    
        dataset_empirical = load_zhao_data_simulated('../../../data/raw_data_Zhao2015a/' + 'zhao2015a_distal_full_trial_data.csv')
    dataset_empirical['target'] = dataset_empirical.apply(lambda x : int(x['trajectory_id'][0]), axis=1)

    #dataset_empirical = dataset_empirical[dataset_empirical['target'] == 1]


    datasets = []
    for file in files:
        dataset_simulated = load_zhao_data_simulated(file)
        datasets.append(dataset_simulated)
    
    dataset_simulated = pd.concat(datasets)
    dataset_simulated['target'] = dataset_simulated.apply(lambda x : int(x['trajectory_id'][0]), axis=1)
    
    
    return dataset_empirical, dataset_simulated 



def normalize_data_empirical(dataset_empirical):
    #Empirical 
    normalized_data_empirical = []
    for condition, data in dataset_empirical.groupby('condition'):
        #flip x coordinates for conflict condition
        if condition == 'conflict':
            normalized_left = normalize_data_by_conflict_condition(data[data.lmturn == 'left'], return_dataframe=True)
            normalized_left['x'] = - normalized_left['x']
            normalized_right = normalize_data_by_conflict_condition(data[data.lmturn == 'right'], return_dataframe=True)
            normalized = pd.concat([normalized_left, normalized_right])
        else:
            normalized = normalize_data_by_conflict_condition(data, return_dataframe=True)

        normalized_data_empirical.append(copy.deepcopy(normalized))

    normalized_data_empirical = pd.concat(normalized_data_empirical)

    return normalized_data_empirical

def normalize_data_simulated(dataset_simulated):
    normalize_by_side = True
    if normalize_by_side:
        dataset_simulated['lmturn'] = np.sign(dataset_simulated.landmark_rotation).apply(lambda x : 'left' if x == -1 else 'right')
        dataset_simulated['landmark_rotation'] = np.abs(dataset_simulated['landmark_rotation'])

        normalized_data_simulated = []
        for condition, data in dataset_simulated.groupby('condition'):
            #flip x coordinates for conflict condition
            if condition == 'conflict':
                normalized_left = normalize_data_by_conflict_condition(data[data.lmturn == 'left'], return_dataframe=True)
                normalized_left['x'] = - normalized_left['x']
                normalized_right = normalize_data_by_conflict_condition(data[data.lmturn == 'right'], return_dataframe=True)
                normalized = pd.concat([normalized_left, normalized_right])
            else:
                normalized = normalize_data_by_conflict_condition(data, return_dataframe=True)

            normalized_data_simulated.append(copy.deepcopy(normalized))

        normalized_data_simulated = pd.concat(normalized_data_simulated)
    else:
        normalized_data_simulated = normalize_data_by_conflict_condition(dataset_simulated, return_dataframe=True)
        
    return normalized_data_simulated



def normalize_data(dataset_empirical, dataset_simulated):
    #Empirical 
    normalized_data_empirical = []
    for condition, data in dataset_empirical.groupby('condition'):
        #flip x coordinates for conflict condition
        if condition == 'conflict':
            normalized_left = normalize_data_by_conflict_condition(data[data.lmturn == 'left'], return_dataframe=True)
            normalized_left['x'] = - normalized_left['x']
            normalized_right = normalize_data_by_conflict_condition(data[data.lmturn == 'right'], return_dataframe=True)
            normalized = pd.concat([normalized_left, normalized_right])
        else:
            normalized = normalize_data_by_conflict_condition(data, return_dataframe=True)

        normalized_data_empirical.append(copy.deepcopy(normalized))

    normalized_data_empirical = pd.concat(normalized_data_empirical)
  


    #Simulated
    normalize_by_side = True
    if normalize_by_side:
        dataset_simulated['lmturn'] = np.sign(dataset_simulated.landmark_rotation).apply(lambda x : 'left' if x == -1 else 'right')
        dataset_simulated['landmark_rotation'] = np.abs(dataset_simulated['landmark_rotation'])

        normalized_data_simulated = []
        for condition, data in dataset_simulated.groupby('condition'):
            #flip x coordinates for conflict condition
            if condition == 'conflict':
                normalized_left = normalize_data_by_conflict_condition(data[data.lmturn == 'left'], return_dataframe=True)
                normalized_left['x'] = - normalized_left['x']
                normalized_right = normalize_data_by_conflict_condition(data[data.lmturn == 'right'], return_dataframe=True)
                normalized = pd.concat([normalized_left, normalized_right])
            else:
                normalized = normalize_data_by_conflict_condition(data, return_dataframe=True)

            normalized_data_simulated.append(copy.deepcopy(normalized))

        normalized_data_simulated = pd.concat(normalized_data_simulated)
    else:
        normalized_data_simulated = normalize_data_by_conflict_condition(dataset_simulated, return_dataframe=True)
        
    return normalized_data_empirical, normalized_data_simulated

#remover per-side-bias 
def remove_per_side_bias(dataset):
    dataset_per_side  = [] 
    for vp, vp_data in dataset.groupby('VPCode'):
        for side, side_data in vp_data.groupby('target'):
            dataset_per_side.append(normalize_data_per_side(side_data))
    dataset_per_side = pd.concat(dataset_per_side)
    
    return dataset_per_side




def generate_endpoint_plot(normalized_data, experiment, path, limit = 5.5, center = False, flip = True):
    for kind in ['simulated']:
        curr_data = copy.deepcopy(normalized_data)
        curr_data['condition'] = curr_data['condition'].map(lambda x : str(x).split('_')[0]) + '_' + curr_data['landmark_rotation'].map(int).map(str)

        for condition, data in curr_data.groupby('condition'):
            plt.figure(figsize = (2,2))

            data = copy.deepcopy(data)

            #data = normalize_data_by_conflict_condition(data, return_dataframe=True)


            if center:
                data[['x', 'y']] = data[['x', 'y']] - np.mean(data[['x', 'y']], axis = 0)

            if kind == 'simulated':
                if flip:
                    if condition.split('_')[0] == 'conflict':
                        data['x'] = -data['x']


            variability_data = data[['x','y']] # - data[['x','y']].mean()
            axHistx,axHisty, axScatter = plot_endpoint_variability(variability_data.values, lim = limit, density_max=20, alpha = 0.5, color = colours[condition.split('_')[0]])

  
            axHistx.set_title('' + condition.split('_')[0])
           
            if condition != 'self-motion':
                axScatter.set_ylabel('')


            filename = path + 'zhao2015_' + experiment + '_' + kind + '_' + condition + '_variability.svg'
            plt.savefig(filename)
            plt.close()

            #plt.show()

    kind = 'simulated'
    zhao2015a_proximal_simulated = Figure("12.375cm", "5cm", 
          SVG(path + 'zhao2015_'+ experiment + '_' + kind +'_self-motion_0_variability.svg').move(0,0),
          SVG(path + 'zhao2015_'+ experiment + '_' + kind +'_landmark_0_variability.svg').move(0,0),
          SVG(path + 'zhao2015_'+ experiment + '_' + kind +'_combined_0_variability.svg').move(0,0)
                           ).tile(3,1)
    svg_file_1 = path + 'zhao2015_' + experiment + '_' + kind + '_variability_all.svg'
    zhao2015a_proximal_simulated.save(svg_file_1)


    kind = 'simulated'
    zhao2015a_proximal_simulated = Figure("16.5cm", "5cm", 
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_15_variability.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_30_variability.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_45_variability.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_90_variability.svg').move(0,0),
                           ).tile(4,1)
                           
    svg_file_2 = path + 'zhao2015_' + experiment + '_' + kind + '_variability_conflict.svg'
    zhao2015a_proximal_simulated.save(svg_file_2)

    
    return svg_file_1, svg_file_2







def generate_endpoint_plots(normalized_data_empirical, normalized_data_simulated, experiment, path, limit = 5.5, center = False, flip = True):
    for kind in ['simulated', 'empirical']: #'empirical'
        if kind == 'simulated':
            curr_data = copy.deepcopy(normalized_data_simulated)
        else:
            #curr_data = copy.deepcopy(normalized_data_empirical[normalized_data_empirical['VPCode'] == 'sub03'])
            curr_data = copy.deepcopy(normalized_data_empirical)

        curr_data['condition'] = curr_data['condition'].map(lambda x : str(x).split('_')[0]) + '_' + curr_data['landmark_rotation'].map(int).map(str)

        for condition, data in curr_data.groupby('condition'):
            plt.figure(figsize = (2,2))

            data = copy.deepcopy(data)

            #data = normalize_data_by_conflict_condition(data, return_dataframe=True)


            if center:
                data[['x', 'y']] = data[['x', 'y']] - np.mean(data[['x', 'y']], axis = 0)

            if kind == 'simulated':
                if flip:
                    if condition.split('_')[0] == 'conflict':
                        data['x'] = -data['x']


            variability_data = data[['x','y']] # - data[['x','y']].mean()
            axHistx,axHisty, axScatter = plot_endpoint_variability(variability_data.values, lim = limit, density_max=20, alpha = 0.5, color = colours[condition.split('_')[0]])

  
            if kind == 'empirical':
                axHistx.set_title('' + condition.split('_')[0])
            else: 
                axHistx.set_title('')

            if condition != 'self-motion':
                axScatter.set_ylabel('')


            filename = path + 'zhao2015_' + experiment + '_' + kind + '_' + condition + '_variability.svg'
            plt.savefig(filename)
            plt.close()

            #plt.show()












def generate_regular_plot(experiment, path):
    kind = 'empirical'
    zhao2015a_proximal_empirical = Figure("12.375cm", "5cm",
          SVG(path + 'zhao2015_'+ experiment + '_' + kind +'_self-motion_0_variability.svg').move(0,0),
          SVG(path + 'zhao2015_'+ experiment + '_' + kind +'_landmark_0_variability.svg').move(0,0),
          SVG(path + 'zhao2015_'+ experiment + '_' + kind +'_combined_0_variability.svg').move(0,0)
                           ).tile(3,1)
    svg_file = path + 'zhao2015_' + experiment + '_' + kind + '_variability_all.svg'
    zhao2015a_proximal_empirical.save(svg_file)

    kind = 'simulated'
    zhao2015a_proximal_simulated = Figure("12.375cm", "5cm", 
          SVG(path + 'zhao2015_'+ experiment + '_' + kind +'_self-motion_0_variability.svg').move(0,0),
          SVG(path + 'zhao2015_'+ experiment + '_' + kind +'_landmark_0_variability.svg').move(0,0),
          SVG(path + 'zhao2015_'+ experiment + '_' + kind +'_combined_0_variability.svg').move(0,0)
                           ).tile(3,1)
    svg_file = path + 'zhao2015_' + experiment + '_' + kind + '_variability_all.svg'
    zhao2015a_proximal_simulated.save(svg_file)

    zhao2015a_proximal = Figure("16.5cm", "9cm", 
          SVG(path + 'zhao2015_' + experiment + '_' + 'empirical' + '_variability_all.svg').move(100,0),
          SVG(path + 'zhao2015_' + experiment + '_' + 'simulated' + '_variability_all.svg').move(100,0)
                           ).tile(1,2)
    svg_file = path + 'zhao2015_' + experiment + '_' + 'both' + '_variability_all.svg'
    zhao2015a_proximal.save(svg_file)
    
    return svg_file

def generate_conflict_15_90_plot(experiment, path):
    kind = 'empirical'
    zhao2015a_proximal_empirical = Figure("16.5cm", "5cm", 
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_15_variability.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_30_variability.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_45_variability.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_90_variability.svg').move(0,0),
                           ).tile(4,1)
    svg_file = path + 'zhao2015_' + experiment + '_' + kind + '_variability_conflict.svg'
    zhao2015a_proximal_empirical.save(svg_file)

    kind = 'simulated'
    zhao2015a_proximal_simulated = Figure("16.5cm", "5cm", 
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_15_variability.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_30_variability.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_45_variability.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + kind + '_conflict_90_variability.svg').move(0,0),
                           ).tile(4,1)
                           
    svg_file = path + 'zhao2015_' + experiment + '_' + kind + '_variability_conflict.svg'
    zhao2015a_proximal_simulated.save(svg_file)

    #full 
    zhao2015a_proximal = Figure("16.5cm", "9cm", 
          SVG(path + 'zhao2015_' + experiment + '_' + 'empirical' + '_variability_conflict.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + 'simulated' + '_variability_conflict.svg').move(0,0)
                           ).tile(1,2)
    
    svg_file = path + 'zhao2015_' + experiment + '_' + 'both' + '_variability_conflict_1590.svg'
    zhao2015a_proximal.save(svg_file)

    return svg_file


def generate_conflict_115_135_plot(experiment, path):
    kind = 'empirical'
    zhao2015a_proximal_empirical = Figure("8.25cm", "6cm", 
          SVG(path + 'zhao2015_'+ experiment + '_' + kind + '_conflict_115_variability.svg').move(0,0),
          SVG(path + 'zhao2015_'+ experiment + '_' + kind + '_conflict_135_variability.svg').move(0,0),
                           ).tile(2,1)

    svg_file = path + 'zhao2015_' + experiment + '_' +  kind + '_variability_conflict.svg'
    zhao2015a_proximal_empirical.save(svg_file)

    kind = 'simulated'

    zhao2015a_proximal_simulated = Figure("8.25cm", "6cm", 
          SVG(path + 'zhao2015_' + experiment + '_' +  kind + '_conflict_115_variability.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' +  kind + '_conflict_135_variability.svg').move(0,0),
                           ).tile(2,1)

    svg_file = path + 'zhao2015_' + experiment + '_' + kind + '_variability_conflict.svg'
    zhao2015a_proximal_simulated.save(svg_file)

    zhao2015a_proximal = Figure("8.25cm", "9cm", 
          SVG(path + 'zhao2015_' + experiment + '_' + 'empirical' + '_variability_conflict.svg').move(0,0),
          SVG(path + 'zhao2015_' + experiment + '_' + 'simulated' + '_variability_conflict.svg').move(0,0)
                           ).tile(1,2)
    
    svg_file = path + 'zhao2015_' + experiment + '_' + 'both' + '_variability_conflict_115135.svg'
    zhao2015a_proximal.save(svg_file)
    
    return svg_file

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


    full_results['order'] = full_results.condition.apply(lambda x : order[x])
    full_results = full_results.sort_values('order', ascending = True)
    
    return full_results