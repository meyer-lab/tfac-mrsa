'''Contains function for importing data from and sending data to synapse'''

import pandas as pd
from synapseclient import Synapse


def importData(username, password, data=None):
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

    ## Input Checking
    if data is None:
        print('Invalid Data Set')
        print('Enter:', 'Copy Number,', 'Methylation,', 'or Gene Expression')
        return None
    syn = Synapse()
    try:
        syn.login(username, password)
    except:
        print('Bad Username or Password')
        return None

    ## Find Data
    if data == 'Copy Number':
        data = syn.get('syn21033823')
    elif data == 'Methylation':
        data = syn.get('syn21033929')
    elif data == 'Gene Expression':
        data = syn.get('syn21033805')

    df = pd.read_excel(data.path)
    syn.logout()
    return df
