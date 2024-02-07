import matplotlib.pyplot as plt 
import numpy as np 
import copy 

from analysis.cue_integration_preprocessing import * 

from svgutils.compose import * 

colours = {'self-motion' : (0.8545098039215687, 0.1415686274509803, 0.14862745098039193),
           'landmark' : (0.24098039215686273, 0.49156862745098046, 0.6962745098039216),
           'combined' : (0.32058823529411773, 0.6664705882352941, 0.31000000000000005),
           'conflict' : (0.5837254901960787, 0.3225490196078431, 0.6225490196078431)}

colours['cue-integration'] = 'grey'
colours['a-integration'] = 'grey'
trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}


def plot_endpoint_variability(data, center = [0,0], lim = 4, alpha = 1.0, color = 'grey', density_max = 30, markersize = 100, axHistx = None, axHisty = None, axScatter = None): 
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels
    
    x = data[:,0]
    y = data[:,1]
    
    if not (axHistx or axHisty or axScatter):
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        # start with a rectangular Figure
        plt.figure(1, figsize=(5, 5))

        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)

        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y, s = 10, color = color, alpha = alpha)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    #lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins, color = color) 
    axHistx.axvline(0.0, color = 'red')
    
    axHisty.hist(y, bins=bins, orientation='horizontal', color = color)
    axHisty.axhline(0.0, color = 'red')

    axHistx.spines['top'].set_visible(False)
    axHistx.spines['right'].set_visible(False)

    axHisty.spines['top'].set_visible(False)
    axHisty.spines['right'].set_visible(False)

    axHistx.set_xlim(axScatter.get_xlim())
    axHistx.set_ylim(0, density_max)
    axHisty.set_ylim(axScatter.get_ylim())
    axHisty.set_xlim(0, density_max)
    
    axScatter.set_xlabel('x (m)')
    axScatter.set_ylabel('y (m)')
    
    axScatter.scatter(center[0], center[1], marker = '+', s = markersize, color = 'red')
    axScatter.scatter(np.mean(x), np.mean(y), marker = 'D', s = markersize/4, color = 'black')
    
    return axHistx,axHisty, axScatter

def plot_raw_endpoints(dataset, third_post_location = np.array([0,0]), title = '', limits = ((-4,4), (-4,4))):
    fig, axes = plt.subplots(3,4, figsize = (20,15), sharex = True, sharey=True)

    c_map = {'landmark' : 0,
             'self-motion' : 1,
             'combined' : 2,
             'conflict' : 3}

    dataset = copy.deepcopy(dataset)
    dataset['target'] = dataset.trajectory_id.apply(lambda x : int(x.split('-')[0]))

    for condition, condition_data in dataset.groupby('condition'):
        if condition == 'conflict': 
            continue 

        print(condition)
        for target, data in condition_data.groupby('target'):
            
            #dunno really how to deal with something as dumb as this...
            if target == 4:
                continue 

            
            ax = axes[c_map[condition], target]
            ax.scatter(data.respx, data.respz, color = colours[condition])
            ax.scatter(np.mean(data.respx), np.mean(data.respz), color = 'black', marker = 'D', s = 200)
            ax.scatter(data.targetx, data.targetz, color = 'black', marker = '*', s = 300)


            ax.plot((third_post_location[0],data.targetx.iloc[0]), (third_post_location[1],data.targetz.iloc[0]), ls = '--', lw = 1, color = 'black')

            #plt.scatter(data.respx - data.targetx, data.respz - data.targetz, color = 'black', alpha = .25)
            #plt.scatter(data.respx - np.mean(data.respx), data.respz - np.mean(data.respz), color = 'black', alpha = .25)

            #print(np.linalg.norm(np.zeros(2) - data[['targetx', 'targetz']].values))

            ax.scatter(third_post_location[0],third_post_location[1], color = 'red')


            #plt.scatter(data.startx, data.startz)
            ax.set_xlim(limits[0][0],limits[0][1])
            ax.set_ylim(limits[1][0],limits[1][1])
            ax.set_aspect('equal')

    
    fig.suptitle(title, y = 1.0125, size = 25)
    plt.tight_layout()
    plt.show()

def plot_cue_integration_comparison(sds_obs_empirical, sds_obs_simulated, order = ['Chen 2017', 'Model'], figsize = (3.25,2)): 
    import seaborn as sns 
    plt.figure(figsize = figsize)    
    p = sns.barplot(x = 'kind', y = 'value', hue = 'condition', data = pd.concat([sds_obs_empirical, sds_obs_simulated]), order = order, palette=colours)
    plt.suptitle('Response Variability')
    plt.ylabel('sd (m)')
    plt.xlabel('')
    plt.ylim(0,1.5)
    #plt.yticks(np.arange(0,1.1,0.1))
    #p.legend(fontsize=10)
    p.legend_.remove()
    sns.despine()

    return p 


def annotate_plot(p, kind_1, kind_2, data, posthoc_tests, show_non_significant, comparisons_correction): 
    import statannotations
    from statannotations.Annotator import Annotator
    import pingouin as pg 


    x = 'kind'
    y = 'SD'
    hue = 'condition'

    order = [kind_1, kind_2]
    hue_order = ['self-motion', 'landmark', 'combined', 'conflict', 'cue-integration']

    pairs = []
    p_values = []

    posthoc_nardini = pg.pairwise_ttests(data = data[data.kind == kind_1], dv = 'SD', between= 'condition', subject = 'VPCode')
    posthoc_model = pg.pairwise_ttests(data = data[data.kind == kind_2], dv = 'SD', between= 'condition', subject = 'VPCode')


    pairs += [((kind_1, i[1]["A"]), (kind_1, i[1]["B"])) for i in posthoc_nardini.iterrows()]
    p_values += ([i[1]["p-unc"] for i in posthoc_nardini.iterrows()])

    pairs += [((kind_2, i[1]["A"]), (kind_2, i[1]["B"])) for i in posthoc_model.iterrows()]
    p_values += ([i[1]["p-unc"] for i in posthoc_model.iterrows()])


    interaction_tests = posthoc_tests[posthoc_tests.Contrast == 'condition * kind']
    pairs += [((i[1]["A"], i[1]["condition"]), (i[1]["B"], i[1]["condition"])) for i in interaction_tests.iterrows()]
    p_values += ([i[1]["p-unc"] for i in interaction_tests.iterrows()])


    pvalue_format = [[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]

    annot = Annotator(data =  data,ax = p.axes, pairs=pairs, x = x, y = y, hue = hue, order = order, hue_order=hue_order, show_non_significant=show_non_significant)
    annot.new_plot(p.axes, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annot.configure(test=None, comparisons_correction=comparisons_correction, verbose=2)
    annot.set_pvalues(p_values)
    annot.annotate()

