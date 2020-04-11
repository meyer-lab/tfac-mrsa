'''Contains function for importing data from and sending data to synapse'''
from os.path import join, dirname
import numpy as np
import pandas as pd
from synapseclient import Synapse

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
