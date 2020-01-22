import numpy as np
import pandas as pd

def importDrugs():
    '''
    Imports Drug Data and separates it by compound

    Returns:
            List of length 24 where each element defines a single compound as a 2D numpy array
    '''
    drugData = pd.read_csv('data/DrugData.csv', header = 0, index_col = False).values
    drugIdx = np.unique(drugData[:,2], return_index = True)[1]
    drugArr = np.split(drugData, drugIdx[1:])
    
    return drugArr
    