''' 
Experiment Factory for creating Experiment Factors which can be used in Runs 
''' 

def get_experiment(experiment_name):
    if experiment_name == 'nardini2008':
        from experiment.nardini2008.params import nardini_2008_parameters
        experiment = nardini_2008_parameters()
    
    if experiment_name == 'zhao2015a':
        from experiment.zhao2015a.params import zhao_2015a_parameters 
        experiment = zhao_2015a_parameters()

    if experiment_name == 'chen2017':  
        from experiment.chen2017.params import chen_2017_parameters
        experiment = chen_2017_parameters()
    

    return experiment
    