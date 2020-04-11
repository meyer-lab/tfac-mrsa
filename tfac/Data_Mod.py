import pandas as pd
import numpy as np
from .dataHelpers import importLINCSprotein

def data_mod(x):
    '''Creates a slice of the data tensor corresponding to the inputted treatment'''
    df = importLINCSprotein()
    spec_df = df.loc[(df['Treatment'] == 'Control') | (df['Treatment'] == x)]
    spec_df = spec_df.drop(columns = ['Sample description', 'Treatment', 'File'])
    y = spec_df.to_numpy()
    return y, spec_df
