import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


def cutMissingValues(data, threshold):
    ''' Function takes in data and cuts rows & columns
    that have more missing values than the threshold set.

    Inputs: data to be cut, threshold to keep (as a fraction)

    Returns: cut data set '''

    uncutData = data
    rows = uncutData.index
    cols = uncutData.columns

    uncutData = pd.DataFrame.to_numpy(data)
    #uncutData = uncutData1.values

    cutData = uncutData
    data_size = uncutData.shape

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
            #print(cutData.shape, rows[row_name] )
            cutData = np.delete(cutData, cut_row_count, 0)
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
            cut_col_count -= 1
        cut_col_count += 1

    return (freshlyChopped)


def normalize(data):
    data_1 = scale(data[0, :, :])
    data_2 = scale(data[1, :, :])
    data_3 = scale(data[0, :, :])
