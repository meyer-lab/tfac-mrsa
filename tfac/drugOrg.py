import numpy as np
import pandas as pd


def importDrugs():
    '''
    Imports Drug Data and separates it by compound

    Returns:
            List of length 24 where each element defines a single compound as a 2D numpy array
    '''
    drugData = pd.read_csv('data/DrugData.csv', header=0, index_col=False).values
    drugIdx = np.unique(drugData[:, 2], return_index=True)[1]
    drugArr = np.split(drugData, drugIdx[1:])

    return drugArr

def tempFilter(drugData):
    '''temporarily uses known cell lines and factors for initial regression testing
    Inputs: one compound (e.g. drugArr[0]) from the drugArr (a 2d numpy array)
    
    Outputs:
    two 2d numpy arrays containing the drugArr and factors with common cell lines
    '''
    factCells = pd.read_csv('data/cellLines(aligned,precut).csv', header=None, index_col=False).values
    factors = getCellLineComps()
    factFiltered, drugFiltered = filterCells(factCells, factors, drugData)
    return factFiltered, drugFiltered

def filterCells(factCells, factors, drugData):
    '''aligns factors and drug data by common cell lines'''
    commonCL = reduce(np.intersect1d, (factCells, drugData[:,0]))
    factIdx = np.where(np.in1d(factCells, commonCL))[0]
    drugIdx = np.where(np.in1d(drugData[:,0], commonCL))[0]
    factFiltered = factors[factIdx, :]
    drugFiltered = drugData[drugIdx, :]
    return factFiltered, drugFiltered