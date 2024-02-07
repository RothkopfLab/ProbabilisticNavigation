import os 
import pandas as pd 

trial_types = {1: 'landmark', 2: 'self-motion', 3: 'combined', 4: 'conflict'}

################################################################################################
                                        ### Empirical  ###
################################################################################################

def load_zhao_data(path = '../../../data/raw_data_Zhao2015a/'):
    files = os.listdir(path)
    files = [file for file in files if ".csv" in file[-4:]]

    dataset = []
    
    for file in files:
        condition = file.split('_')
        data = pd.read_csv(path + file)
        data = data[['x','y']] / 100

        data['condition'] = condition[2].split('.')[0]
        

        if len(condition) == 4:
            #data['condition'] += '_'  + condition[3].split('.')[0]
            data['landmark_rotation'] = condition[3].split('.')[0]
        else:
            data['landmark_rotation'] = 0 

        data = data.dropna()

        dataset.append(data)

    dataset = pd.concat(dataset)
    dataset['VPCode'] = 'None'
    
    return dataset


################################################################################################
                                        ### Simulated  ###
################################################################################################
#Simulated Data 
def load_zhao_data_simulated(file):
    dataset_simulated =  pd.read_csv(file)
    dataset_simulated = dataset_simulated[dataset_simulated.task_finished.fillna(False)]
    dataset_simulated = dataset_simulated.dropna()


    #apply this to the conflict conditions 
    dataset_simulated['condition'] = dataset_simulated['condition'].apply(lambda x: trial_types[int(x)])
    dataset_simulated['condition'] = dataset_simulated.apply(lambda x : ('conflict' + '_' + str(int(x['landmark_rotation']))) if x['condition'] == 'conflict' else x['condition'], axis = 1)
    dataset_simulated = dataset_simulated.sort_values(by = 'condition')
    

    if 'VPCode' not in dataset_simulated:
        dataset_simulated['VPCode'] = 'None'


    dataset_simulated['condition'] = dataset_simulated.condition.apply(lambda x : x.split('_')[0])
    
    #set all other rotation conditions to 0 
    dataset_simulated.loc[dataset_simulated.condition != 'conflict','landmark_rotation'] = 0
    
    return dataset_simulated
