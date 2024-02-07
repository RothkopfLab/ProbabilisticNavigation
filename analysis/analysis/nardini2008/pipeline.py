#import analysis functions 
from analysis.nardini2008.utils import * 

from analysis.key_plots import plot_endpoint_variability

import matplotlib.pyplot as plt 
from svgutils.compose import *

colours = {'self-motion' : (0.8545098039215687, 0.1415686274509803, 0.14862745098039193),
           'landmark' : (0.24098039215686273, 0.49156862745098046, 0.6962745098039216),
           'combined' : (0.32058823529411773, 0.6664705882352941, 0.31000000000000005),
           'conflict' : (0.5837254901960787, 0.3225490196078431, 0.6225490196078431)}

colours['cue-integration'] = 'grey'
colours['a-integration'] = 'grey'
trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}



def plot_all_conditions(path, normalized_data):
    for condition, data in normalized_data.groupby('condition'):
        plt.figure(figsize = (2,2))
        axHistx,axHisty, axScatter = plot_endpoint_variability(data[['x', 'y']].values, lim = 1.5, alpha = 0.5, color = colours[condition])
        if condition != 'self-motion':
            axScatter.set_ylabel('')
        plt.savefig(path + 'nardini_' + condition + '_variability.svg')
        plt.close()

    output = Figure("17.5cm", "5cm", 
          SVG(path + 'nardini_self-motion_variability.svg').move(20.0,0),
          SVG(path + 'nardini_landmark_variability.svg').move(20.0,0),
          SVG(path + 'nardini_combined_variability.svg').move(20.0,0),
          SVG(path + 'nardini_conflict_variability.svg').move(20.0,0)).tile(4,1)

    output.save(path + 'nardini_all_variability_colours.svg')

    return output



def nardini_plot(path, normalized_data_empirical, normalized_data_simulated):    
    ################################################################################################
                                            ### Empirical  ###
    ################################################################################################
    for condition, data in normalized_data_empirical.groupby('condition'):
        plt.figure(figsize = (2,2))
        axHistx,axHisty, axScatter = plot_endpoint_variability(data[['x', 'y']].values, lim = 1.5, alpha = 0.5, color = colours[condition])
        axHistx.set_title('' + condition)

        if condition != 'self-motion':
            axScatter.set_ylabel('')

        axScatter.set_xlabel('')

        plt.tight_layout()
        plt.savefig(path + 'nardini2008_' + condition + '_variability.svg')

        plt.close()


    nardini2008_data = Figure("17.5cm", "5cm", 
          SVG(path + 'nardini2008_self-motion_variability.svg').move(20.0,0),
          SVG(path + 'nardini2008_landmark_variability.svg').move(20.0,0),
          SVG(path + 'nardini2008_combined_variability.svg').move(20.0,0),
          SVG(path + 'nardini2008_conflict_variability.svg').move(20.0,0)).tile(4,1)
    nardini2008_data.save(path + 'nardini_2008_all_variability_coloured.svg')

    ################################################################################################
                                            ### Simulated  ###
    ################################################################################################

    for condition, data in normalized_data_simulated.groupby('condition'):
        plt.figure(figsize = (2,2))
        axHistx,axHisty, axScatter = plot_endpoint_variability(data[['x', 'y']].values, lim = 1.5, alpha = 0.5, color = colours[condition])
        if condition != 'self-motion':
            axScatter.set_ylabel('')
        plt.savefig(path + 'model_' + condition + '_variability.svg')
        plt.close()

    output = Figure("17.5cm", "5cm", 
          SVG(path + 'model_self-motion_variability.svg').move(20.0,0),
          SVG(path + 'model_landmark_variability.svg').move(20.0,0),
          SVG(path + 'model_combined_variability.svg').move(20.0,0),
          SVG(path + 'model_conflict_variability.svg').move(20.0,0)).tile(4,1)
    output.save(path + 'model_all_variability_colours.svg')

    output = Figure("17.5cm", "9cm",
          #Grid(50,50),
          #Text("A", 15, 20, size=20, weight='bold'),
          SVG(path + 'nardini_2008_all_variability_coloured.svg').move(30.0,20),
          SVG(path + 'model_all_variability_colours.svg').move(30.0, 175.0))

    #output.save(path + 'nardini_vs_model_simulation_colours.svg')
    return output

def extract_SDs(model_results):
    df = pd.DataFrame.from_dict(model_results['sigmas'], orient='index')
    df = df.reset_index()
    df['condition'] = df['index']
    df['value'] = np.sqrt(df[0])
    df['metric'] = 'SD'
    df = df[['condition', 'metric', 'value']]
    df = df.sort_values('condition', ascending = False)
    df.reset_index()
    
    return df[df['condition'] != 'cue-alternation']