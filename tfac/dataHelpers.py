'''Contains function for importing data from and sending data to synapse'''
import numpy as np
import pandas as pd
import tqdm
from synapseclient import Synapse, File


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

    # Find Data
    if dataType == 'Copy Number All':
        data = syn.get('syn21089502')
    elif dataType == 'Methylation All':
        data = syn.get('syn21089540')
    elif dataType == 'Gene Expression All':
        data = syn.get('syn21089539')
    elif dataType == 'Copy Number':
        data = syn.get('syn21299981')
    elif dataType == 'Methylation':
        data = syn.get('syn21299983')
    elif dataType == 'Gene Expression':
        data = syn.get('syn21299979')

    df = pd.read_excel(data.path, index_col=0)
    syn.logout()
    return df


def exportData(username, password, data, nm):
    '''Pandas Data Frame to upload back to synapse as an excel file
    ---------------------------------------------------------------
    Parameters:
        username: String
            Your synapse username
        password: String
            You synapse password
        data: Data Frame
            Pandas object containing the data to upload to synapse as an excel file
        name: String
            A name for the file in synapse
    '''
    syn = Synapse()
    syn.login(username, password)
    proj = syn.get('syn21032722')
    data.to_csv('data/file.csv')
    syn.store(File(path='file.csv', name=nm, parent=proj))
    syn.logout()


def makeTensor(username, password):
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

    # FIX: This replaces nans with zeros -- we need to either cut them or justifiably impute them
    methyl = methylation.values[:, 1:]
    mk = np.isnan(methyl, where=True)
    methyl[mk] = 0

    # Create final tensor
    syn.logout()
    return np.stack((gene_expression.values[:, 1:], copy_number.values[:, 1:], methyl))
