'''Contains functions used in data preprocessing'''
from functools import reduce
import numpy as np
import pandas as pd


def extractData(filename, columns=None, row=0, col=None):
    '''useless -- to be deleted'''
    return pd.read_excel(filename, header=row, index_col=col, usecols=columns)


def extractGeneNames():
    '''
    Extracts sorted gene names from all data sets

    Returns:
            Order: Methylation, Gene Expression, Copy Number
            Returns three numpy arrays with gene names from aforementioned datasets
    '''
    data = extractData('data/GeneData_All.xlsx', 'A:C')
    data = data.to_numpy()

    methylation = data[:13493, 0].astype(str)
    geneExp = data[:, 1].astype(str)
    copyNum = data[:23316, 2].astype(str)

    return methylation, geneExp, copyNum


def extractCellLines():
    '''
    Extracts sorted cell lines from all data sets

    Returns:
            Order: Methylation, Gene Expression, Copy Number
            Returns three numpy arrays with cell lines from aforementioned datasets
    '''
    data = extractData('data/CellLines_All.xlsx', 'A:C')
    data = data.to_numpy()
    methylation = data[:843, 0].astype(str)
    geneExp = data[:1019, 1].astype(str)
    copyNum = data[:, 2].astype(str)

    return methylation, geneExp, copyNum


def findCommonGenes(methIdx, geneIdx, copyIdx):
    '''
    Finds the set of unique gene names from the copy number, methylation, and gene expression dataset

    Returns:
            Numpy array of unique common gene names
    '''
    commonGenes = reduce(np.intersect1d, (methIdx, geneIdx, copyIdx))
    return commonGenes


def findCommonCellLines(methCL, geneCL, copyCL):
    '''
    Finds the set of unique cell lines from the copy number, methylation, and gene expression dataset

    Returns:
            Numpy array of unique common cell lines
    '''
    commonCellLines = reduce(np.intersect1d, (methCL, geneCL, copyCL))
    return commonCellLines


def filterData(methData, geneData, copyData):
    '''
    Pushes the filtered data to synapse :D
    '''

    methValues = np.array(methData.values)
    geneValues = np.array(geneData.values)
    copyValues = np.array(copyData.values)

    methIdx = np.array(methData.index)
    geneIdx = np.array(geneData.index)
    copyIdx = np.array(copyData.index)

    methCL = np.array(methData.columns)
    geneCL = np.array(geneData.columns)
    copyCL = np.array(copyData.columns)


#   methG, geneG, copyG = extractGeneNames()
#   methCL, geneCL, copyCL = extractCellLines(cut, methCell)
    commonG = findCommonGenes(methIdx, geneIdx, copyIdx)
    commonCL = findCommonCellLines(methCL, geneCL, copyCL)

    # Find indices of common genes in full dataset
    methGIndices = np.where(np.in1d(methIdx, commonG))[0]
    geneGIndices = np.where(np.in1d(geneIdx, commonG))[0]
    copyGIndices = np.where(np.in1d(copyIdx, commonG))[0]

    # Find indices of common cell lines in full dataset
    methCLIndices = np.where(np.in1d(methCL, commonCL))[0]
    geneCLIndices = np.where(np.in1d(geneCL, commonCL))[0]
    copyCLIndices = np.where(np.in1d(copyCL, commonCL))[0]

    methFiltered = methValues[methGIndices, :]
    methFiltered = methFiltered[:, methCLIndices]

    geneFiltered = geneValues[geneGIndices, :]
    geneFiltered = geneFiltered[:, geneCLIndices]

    copyFiltered = copyValues[copyGIndices, :]
    copyFiltered = copyFiltered[:, copyCLIndices]

    methDF = pd.DataFrame(data=methFiltered, index=methIdx[methGIndices], columns=commonCL)
    geneDF = pd.DataFrame(data=geneFiltered, index=geneIdx[geneGIndices], columns=commonCL)
    copyDF = pd.DataFrame(data=copyFiltered, index=copyIdx[copyGIndices], columns=commonCL)

#     exportData('NilayShah', 'nilayisthebest', methDF, 'MethAligned')
#     exportData('NilayShah', 'nilayisthebest', geneDF, 'GeneAligned')
#     exportData('NilayShah', 'nilayisthebest', copyDF, 'CopyAligned')

    return methDF, geneDF, copyDF


def extractCopy(dupes=False, cellLines=False):
    '''
    Extracts out all duplicates data using excel file of gene names

    Returns:
            Order: Methylation, Gene Expression, Copy Number
            List of length 3 containing 3D arrays with
            duplicate EGIDs, indices, and # of dupes corresponding to each EGID
            Also returns # of duplicates in each data set
    '''
    if cellLines:
        methylation, geneExp, copyNum = extractCellLines()
    else:
        methylation, geneExp, copyNum = extractGeneNames()

    data = [methylation, geneExp, copyNum]

    if dupes:
        duplicates = np.zeros(3)

    returnVal = []  # creates list of 3 2D numpy arrays containing names and indices
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
    else:
        return returnVal
