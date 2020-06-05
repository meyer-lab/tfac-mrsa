"""Functions for importing and manipulating data from the Cancer Cell Line Encyclopedia"""
import os
from os.path import join
from functools import reduce
import numpy as np
import pandas as pd
from synapseclient import Synapse
from sklearn.preprocessing import scale

path_here = os.path.dirname(os.path.abspath(__file__))
################################ Summary Function for All Preprocessing #################################################


def DataWorkFlow(username, password, threshold):
    """Contain the full preprocessing procedure in one function"""
    # import data from synapse
    syn = Synapse()
    syn.login(username, password)
    meth0 = pd.read_csv(syn.get('syn21303732').path, index_col=0)
    copy0 = pd.read_csv(syn.get('syn21303730').path, index_col=0)
    gene0 = pd.read_csv(syn.get('syn21303731').path, index_col=0)

    # run filter - aligns all 3 data sets to the same size
    meth1, gene1, copy1 = filterData(meth0, gene0, copy0)

    # cut missing values from methylation data
    meth1_5 = cutMissingValues(meth1, threshold)

    # redo filter to make the data sets the same size again
    meth2, gene2, copy2 = filterData(meth1_5, gene1, copy1)

    return meth2, gene2, copy2

################################ Individual Data Import Functions ###############################################################


def importData(username, password, dataType=None):
    '''Data Import from synapse
    ----------------------------------------------
    Parameters:
        username: string
            Synapse username
        password: string
            Synapse password
        data: string
            'Copy Number', 'Methylation', or 'Gene Expression'

    Returns:
        df: DataFrame
            Data from the CCLE in data frame format
    '''

    # Input Checking
    if dataType is None:
        print('Invalid Data Set')
        print('For Raw Data Enter:', '"Copy Number All",', '"Methylation All",', 'or "Gene Expression All"')
        print('For Processed Data Enter:', '"Copy Number"', '"Methylation"', 'or "Gene Expression"')
        return None
    syn = Synapse()
    try:
        syn.login(username, password)
    except BaseException:
        print('Bad Username or Password')
        return None

    # Find Data -- TODO: FIGURE OUT WHAT THESE ALL SPECIFICALLY REPRESENT
    if dataType == 'Copy Number All':
        df = pd.read_excel(syn.get('syn21033823').path)
    elif dataType == 'Methylation All':
        df = pd.read_excel(syn.get('syn21033929').path)
    elif dataType == 'Gene Expression All':
        df = pd.read_excel(syn.get('syn21033805').path)
    elif dataType == 'Copy Number':
        df = pd.read_csv(syn.get('syn21303730').path, index_col=0, header=0)
    elif dataType == 'Methylation':
        df = pd.read_csv(syn.get('syn21303732').path, index_col=0, header=0)
    elif dataType == 'Gene Expression':
        df = pd.read_csv(syn.get('syn21303731').path, index_col=0, header=0)

    syn.logout()
    return df


def makeTensor(username, password):
    '''Generate correctly aligned tensor for factorization'''
    syn = Synapse()
    syn.login(username, password)

    # Get Data
    copy_number = importData(username, password, 'Copy Number')
    methylation = importData(username, password, 'Methylation')
    gene_expression = importData(username, password, 'Gene Expression')

    # Create final tensor
    arr = normalize(np.stack((gene_expression.values, copy_number.values, methylation.values)))

    syn.logout()
    return arr


def cellLineNames():
    """Get a Full List of Cell Lines for a plot legend
    ------------------------------------------------------------
    ***Calling np.unique(ls) yields the 23 different cancer types***
    """
    filename = join(path_here, "./data/cellLines(aligned,precut).csv")
    df = pd.read_csv(filename)
    names = np.insert(df.values, 0, "22RV1_PROSTATE")
    ls = [x.split('_', maxsplit=1)[1] for x in names]
    return ls


def geneNames():
    '''Get a full list of the ordered gene names in the tensor (names are EGID's)'''
    genes = importData("robertt", "LukeKuechly59!", "Gene Expression")
    return np.array(genes.index)

################################ Tensor Data Preprocessing ######################################################################


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
    for i, _ in enumerate(data):
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
    return returnVal, None


def cutMissingValues(data, threshold):
    ''' Function takes in data and cuts rows & columns
    that have more missing values than the threshold set.
    (Currently only used for methylation)
    Inputs: data to be cut, threshold to keep (as a fraction)
    Returns: cut data set '''

    uncutData = data
    rows = uncutData.index
    cols = uncutData.columns
    data_size = data.shape

    print(data_size)

    masked_rows = []
    for _ in range(len(rows)):
        masked_rows.append(0)
    masked_cols = []
    for _ in range(len(cols)):
        masked_cols.append(0)

    uncutData = pd.DataFrame.to_numpy(data)

    cutData = uncutData
    data_size = tuple(uncutData.shape)

    limit_rows = (1 - threshold) * data_size[1]
    limit_cols = (1 - threshold) * data_size[0]

    cut_row_count = 0
    cut_col_count = 0

    # cut along genes
    for row_name in range(data_size[0]):
        count = 0
        for col_name in range(data_size[1]):
            if np.isnan(uncutData[row_name, col_name]):
                count += 1
        if count >= limit_rows:
            cutData = np.delete(cutData, cut_row_count, 0)
            masked_rows[row_name] = 1
            cut_row_count -= 1
        cut_row_count += 1

    # cut along cell lines
    data_size = cutData.shape
    freshlyChopped = cutData

    for col_name in range(data_size[1]):
        count = 0
        for row_name in range(data_size[0]):
            if np.isnan(cutData[row_name, col_name]):
                count += 1
        if count >= limit_cols:
            freshlyChopped = np.delete(freshlyChopped, cut_col_count, 1)
            masked_cols[col_name] = 1
            cut_col_count -= 1
        cut_col_count += 1

    rows = np.ma.masked_array(rows, masked_rows)
    rows = rows.compressed()
    cols = np.ma.masked_array(cols, masked_cols)
    cols = cols.compressed()

    df_labeled = pd.DataFrame(data=freshlyChopped, columns=cols, index=rows)
    print(df_labeled.shape)

    return df_labeled


def normalize(data):
    """Scale the data along cell lines"""
    data_1 = scale(data[0, :, :])
    data_2 = scale(data[1, :, :])
    data_3 = scale(data[2, :, :])
    return np.array((data_1, data_2, data_3))

################################ Drug Data Processing ############################################################


def importDrugs():
    '''
    Imports Drug Data and separates it by compound
    Returns:
            List of length 24 where each element defines a single compound as a 2D numpy array
    '''
    filename = os.path.join(path_here, './data/DrugData.csv')
    drugData = pd.read_csv(filename, header=0, index_col=False).values
    drugs = np.unique(drugData[:, 2])
    drugList = []
    for drug in drugs:
        drugIdx = np.where(np.in1d(drugData[:, 2], drug))[0]
        drugList.append(drugData[drugIdx, :])

    return drugList


def tempFilter(drugData, factors):
    '''temporarily uses known cell lines and factors for initial regression testing
    Inputs: one compound (e.g. drugArr[0]) from the drugArr (a 2d numpy array)
            Factorization output (cell line components -- serves as the X in regression)
    Outputs:
    two 2d numpy arrays containing the drugArr and factors with common cell lines
    '''
    filename = os.path.join(path_here, "./data/cellLines(aligned,precut).csv")
    factCells = pd.read_csv(filename, header=None, index_col=False).values
    factFiltered, drugFiltered = filterCells(factCells, factors, drugData)
    return factFiltered, drugFiltered


def filterCells(factCells, factors, drugData):
    '''aligns factors and drug data by common cell lines'''
    commonCL = reduce(np.intersect1d, (factCells, drugData[:, 0]))
    factIdx = np.where(np.in1d(factCells, commonCL))[0]
    drugIdx = np.where(np.in1d(drugData[:, 0], commonCL))[0]
    factFiltered = factors[factIdx, :]
    drugFiltered = drugData[drugIdx, :]
    return factFiltered, drugFiltered
