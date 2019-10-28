'''Functions for handling gene alignment'''
import numpy as np
import pandas as pd

def extractData(filename, columns=None, row=0, col=None):
    '''Pullling Data from excel file on server'''
    return pd.read_excel(filename, header=row, index_col=col, usecols=columns)

def extractCopy(dupes=False):
    '''
    Extracts out all duplicates data using excel file of gene names

    Returns:
            Order: Methylation, Gene Expression, Copy Number
            List of length 3 containing 3D arrays with
            duplicate gene names, indices, and # of dupes corresponding to each name
            Also returns # of duplicates in each data set
    '''
    data = extractData('data/GeneData_All.xlsx', 'A:C')
    data = data.to_numpy()

    methylation = np.append(data[:12158, 0], data[12159:21338, 0])
    geneExp = data[:, 1]
    copyNum = data[:23316, 2]
    data = [methylation.astype(str), geneExp.astype(str), copyNum.astype(str)]

    if dupes:
        duplicates = np.zeros(3)

    returnVal = [] #creates list of 3 2D numpy arrays containing names and indices
    for i in range(len(data)):
        uData = np.unique(data[i], return_index=True, return_counts=True)

        if dupes:
            duplicates[i] = data[i].size - uData[0].size

        copyData = []
        idxData = []
        count = []
        for j in range(uData[0].size):
            if uData[2][j] != 1:
                copyData.append(uData[0][j])
                idxData.append(uData[1][j])
                count.append(uData[2][j])
        returnVal.append(np.array([copyData, idxData, count]))

    if dupes:
        return returnVal, duplicates
    return returnVal
