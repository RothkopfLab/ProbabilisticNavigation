from analysis.chen2017.data_loading import load_chen_data, process_simulated_data
from analysis.chen2017.utils import * 
from analysis.chen2017.preprocessing import preprocess_data_chen_2017, preprocess_per_participant
from analysis.chen2017.cue_integration_model import chen_2017_cue_integration_model
from analysis.key_plots import *

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

from svgutils.compose import *

colours = {'self-motion' : (0.8545098039215687, 0.1415686274509803, 0.14862745098039193),
           'landmark' : (0.24098039215686273, 0.49156862745098046, 0.6962745098039216),
           'combined' : (0.32058823529411773, 0.6664705882352941, 0.31000000000000005),
           'conflict' : (0.5837254901960787, 0.3225490196078431, 0.6225490196078431)}

colours['cue-integration'] = 'grey'
colours['a-integration'] = 'grey'
trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}


def load_chen_datasets(file, environment_choice):
    ################################################################################################
                                            ### Empirical  ###
    ################################################################################################
    dataset_empirical = load_chen_data(path = '../../../data/raw_data_Chen_2017/', 
                                     file = 'chen2017_exp1a_2environments.csv', 
                                     environment_choice = environment_choice)



    #dataset_empirical['trajectory_id'] = dataset_empirical['trajectory_id'].astype(str)
    dataset_empirical['trajectory_id'] = dataset_empirical.apply(lambda x : str(x['post1']) + '-' + str(x['post2']) + '-' + str(x['post3']), axis=1)
    
    dataset_empirical['target'] = dataset_empirical.apply(lambda x : int(x['post1']), axis=1)
    dataset_empirical['target'] =  dataset_empirical[['targetx', 'targetz']].round(2).apply(lambda x : str(x['targetx']) + '-' + str(x['targetz']), axis = 1)



    ################################################################################################
                                            ### Simulated  ###
    ################################################################################################
    dataset_simulated = pd.read_csv(file)
    dataset_simulated = dataset_simulated.dropna() 
    dataset_simulated.loc[:,'condition'] = dataset_simulated.condition.apply(lambda x : int(x))
    dataset_simulated[['targetx', 'targetz']] = dataset_simulated[['targetx', 'targetz']].values - dataset_simulated[['jitterx', 'jitterz']].values
    
    #rezero dataset
    dataset_simulated['targetz'] += 0.75 
    dataset_simulated['startz'] += 0.75 
    dataset_simulated['respz'] += 0.75 
    
    dataset_simulated['target'] =  dataset_simulated[['targetx', 'targetz']].round(2).apply(lambda x : str(x['targetx']) + '-' + str(x['targetz']), axis = 1)
    dataset_simulated.loc[:,'target_loc'] = np.round(dataset_simulated['targetx'] - dataset_simulated['jitterx'], 3)
    dataset_simulated.condition = dataset_simulated.condition.apply(lambda x : trial_types[x])
    
    
    
    
    return dataset_empirical, dataset_simulated 




def generate_per_condition_plots(normalized_data_empirical, normalized_data_simulated, path, environment_choice):
    
    ################################################################################################
                                            ### Empirical  ###
    ################################################################################################
    for condition, data in normalized_data_empirical.groupby('condition'):
        plt.figure(figsize = (2,2))
        axHistx,axHisty, axScatter = plot_endpoint_variability(data[['x', 'y']].values, lim = 2.5, alpha = 0.5, color = colours[condition])
        axHistx.set_title('' + condition)


        if condition != 'self-motion':
            axScatter.set_ylabel('')

        axScatter.set_xlabel('')

        plt.tight_layout()
        plt.savefig(path + 'chen_2017_' + '2_environments' + '_' + environment_choice  + '_' + condition + '_variability.svg')

        plt.close()

    ################################################################################################
                                            ### Simulated  ###
    ################################################################################################
    for condition, data in normalized_data_simulated.groupby('condition'):
        plt.figure(figsize = (2,2))

        #if condition == 'conflict':
        #    data['x'] = -data['x']

        #print(data[['x', 'y']].mean())
        axHistx,axHisty, axScatter = plot_endpoint_variability(data[['x', 'y']].values, lim = 2.5, alpha = 0.5, color = colours[condition])
        #plt.title(condition)
        if condition != 'self-motion':
            axScatter.set_ylabel('')
        plt.savefig(path + 'output_' + condition + '_variability.svg')
        plt.close()


def generate_per_condition_plot(normalized_data, path):
    ################################################################################################
                                            ### Simulated  ###
    ################################################################################################
    for condition, data in normalized_data.groupby('condition'):
        plt.figure(figsize = (2,2))

        #if condition == 'conflict':
        #    data['x'] = -data['x']

        #print(data[['x', 'y']].mean())
        axHistx,axHisty, axScatter = plot_endpoint_variability(data[['x', 'y']].values, lim = 2.5, alpha = 0.5, color = colours[condition])
        #plt.title(condition)
        if condition != 'self-motion':
            axScatter.set_ylabel('')
        plt.savefig(path + 'output_' + condition + '_variability.svg')
        plt.close()

    output = Figure("17.5cm", "4cm", 
        SVG(path + 'output_self-motion_variability.svg').move(20.0,0),
        SVG(path + 'output_landmark_variability.svg').move(20.0,0),
        SVG(path + 'output_combined_variability.svg').move(20.0,0),
        SVG(path + 'output_conflict_variability.svg').move(20.0,0)).tile(4,1)

    svg_file = path + 'output_all_variability_colours.svg'
    output.save(path + 'output_all_variability_colours.svg')

    return svg_file  


        
        
        
#generate a main plot 
from svgutils.compose import *
def generate_chen_main_plot(path, environment_choice):
    ################################################################################################
                                            ### Main Plot (Both)  ###
    ################################################################################################


    chen2017_exp1a = Figure("17.5cm", "4cm", 
          SVG(path + 'chen_2017_' + '2_environments' + '_' + environment_choice  + '_' + 'self-motion' + '_variability.svg').move(20.0,0),
          SVG(path + 'chen_2017_' + '2_environments' + '_' + environment_choice  + '_' + 'landmark' + '_variability.svg').move(20.0,0),
          SVG(path + 'chen_2017_' + '2_environments' + '_' + environment_choice  + '_' + 'combined' + '_variability.svg').move(20.0,0),
          SVG(path + 'chen_2017_' + '2_environments' + '_' + environment_choice  + '_' + 'conflict' + '_variability.svg').move(20.0,0)).tile(4,1)

    svg_file = path + 'chen_2017_' + '2_environments' + '_' + environment_choice + '_variability.svg'
    chen2017_exp1a.save(svg_file)

    output = Figure("17.5cm", "4cm", 
          SVG(path + 'output_self-motion_variability.svg').move(20.0,0),
          SVG(path + 'output_landmark_variability.svg').move(20.0,0),
          SVG(path + 'output_combined_variability.svg').move(20.0,0),
          SVG(path + 'output_conflict_variability.svg').move(20.0,0)).tile(4,1)

    output.save(path + 'output_all_variability_colours.svg')


    output = Figure("17.5cm", "9cm", 
          SVG(svg_file).move(20.0,0),
          SVG(path + 'output_all_variability_colours.svg').move(20.0,0)).tile(1,2)

    svg_file = path + 'chen_2017_2environments_' + environment_choice + '_all_variability.svg'
    output.save(path + 'chen_2017_2environments_' + environment_choice + '_all_variability.svg')
    
    
    return svg_file