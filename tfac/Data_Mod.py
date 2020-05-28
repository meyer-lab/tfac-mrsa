"""Data pre-processing and tensor formation"""
import pandas as pd
import numpy as np
from .dataHelpers import importLINCSprotein


def data_mod(x, df=None):
    '''Creates a slice of the data tensor corresponding to the inputted treatment'''
    if not isinstance(df, pd.core.frame.DataFrame):
        df = importLINCSprotein()
    spec_df = df.loc[(df['Treatment'] == 'Control') | (df['Treatment'] == x)]
    times = spec_df['Time'].to_numpy().tolist()
    spec_df = spec_df.drop(columns=['Sample description', 'Treatment', 'Time'])
    y = spec_df.to_numpy()
    return y, spec_df, times


def form_tensor():
    '''Creates tensor in numpy array form and returns tensor, treatments, and time'''
    df = importLINCSprotein()
    tempindex = df["Sample description"]
    tempindex = tempindex[:36]
    i = 0
    for a in tempindex:
        tempindex[i] = a[3:]
        i += 1
    treatments = df["Treatment"][0:36]
    times = df["Time"][0:36]
    df = df.drop(["Sample description"], axis=1)
    by_row_index = df.groupby(df.index)
    df_means = by_row_index.mean()
    df_means.insert(0, "Treatment", value=treatments)
    df_means.insert(0, "Sample description", tempindex)
    unique_treatments = np.unique(df_means['Treatment'].values).tolist()
    unique_treatments.remove('Control')

    slices = []
    for treatment in unique_treatments:
        array, _, times = data_mod(treatment, df_means)
        slices.append(array)
    tensor = np.stack(slices)
    return tensor, unique_treatments, times
