import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import tqdm
from synapseclient import Synapse, File
from dataOrg import filterData
from fixCutDataScript import cutMissingValues


def DataWorkFlow (username, password, threshold):
    
    #import data from synapse
    syn = Synapse()
    syn.login(username, password)
    meth0 = pd.read_csv(syn.get('syn21533071').path, index_col = 0)
    copy0 = pd.read_csv(syn.get('syn21533091').path, index_col = 0)
    gene0 = pd.read_csv(syn.get('syn21533090').path, index_col = 0)
    
    #run filter - aligns all 3 data sets to the same size
    meth1, gene1, copy1 = filterData(meth0, gene0, copy0)
    
    #cut missing values from methylation data
    meth1_5 = cutMissingValues(meth1, threshold)
    
    #redo filter to make the data sets the same size again
    meth2, gene2, copy2 = filterData(meth1_5, gene1, copy1)
    
    return meth2, gene2, copy2


def normalize(data):
    data_1 = scale(data[0, :, :])
    data_2 = scale(data[1, :, :])
    data_3 = scale(data[2, :, :])
    return np.array((data_1, data_2, data_3))
