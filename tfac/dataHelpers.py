'''Contains function for importing data from and sending data to synapse'''
from os.path import join, dirname
import numpy as np
import pandas as pd
import tqdm
from synapseclient import Synapse
from .dataProcess import normalize

path_here = dirname(dirname(__file__))


def importLINCSprotein():
    """ Import protein characterization from LINCS. """
    dataA = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_A.csv"))
    dataB = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_B.csv"))
    dataC = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_C.csv"))

    dataA["File"] = "A"
    dataB["File"] = "B"
    dataC["File"] = "C"

    return pd.concat([dataA, dataB, dataC])


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
        print('Enter:', 'Copy Number All,', 'Methylation All,', 'or Gene Expression All')
        return None
    syn = Synapse()
    try:
        syn.login(username, password)
    except BaseException:
        print('Bad Username or Password')
        return None

    # Find Data -- TODO: FIGURE OUT WHAT THESE ALL SPECIFICALLY REPRESENT
    if dataType == 'Copy Number All':
        data = syn.get('syn21089502')             # Insert Non-Processed Data
    elif dataType == 'Methylation All':
        data = syn.get('syn21089540')             # Insert Non-Processed Data
    elif dataType == 'Gene Expression All':
        data = syn.get('syn21089539')             # Insert Non-Processed Data
    elif dataType == 'Copy Number':
        data = syn.get('syn21303730')             # Insert Processed Data
    elif dataType == 'Methylation':
        data = syn.get('syn21303732')             # Insert Processed Data
    elif dataType == 'Gene Expression':
        data = syn.get('syn21303731')             # Insert Processed Data

    df = pd.read_csv(data.path, index_col=0, header=0)
    syn.logout()
    return df


def makeTensor(username, password, returndf=False):
    '''Generate correctly aligned tensor for factorization'''
    syn = Synapse()
    syn.login(username, password)

    # Setup Data Carriers
    copy_number = pd.DataFrame()
    methylation = pd.DataFrame()
    gene_expression = pd.DataFrame()

    # Get Data
    for chunk1 in tqdm.tqdm(pd.read_csv(syn.get('syn21303730').path, chunksize=150), ncols=100, total=87):
        copy_number = pd.concat((copy_number, chunk1))
    for chunk2 in tqdm.tqdm(pd.read_csv(syn.get('syn21303732').path, chunksize=150), ncols=100, total=87):
        methylation = pd.concat((methylation, chunk2))
    for chunk3 in tqdm.tqdm(pd.read_csv(syn.get('syn21303731').path, chunksize=150), ncols=100, total=87):
        gene_expression = pd.concat((gene_expression, chunk3))

    if returndf:
        return gene_expression, copy_number, methylation

    arr = normalize(np.stack((gene_expression.values[:, 1:], copy_number.values[:, 1:], methylation.values[:, 1:])))

    # Create final tensor
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
