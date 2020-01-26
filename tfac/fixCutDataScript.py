"""Functions for cutting missing values in methylation data"""
import numpy as np
import numpy.ma as ma
import pandas as pd


def cutMissingValues(data, threshold):
    ''' Function takes in data and cuts rows & columns
    that have more missing values than the threshold set.

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

    rows = ma.masked_array(rows, masked_rows)
    rows = rows.compressed()
    cols = ma.masked_array(cols, masked_cols)
    cols = cols.compressed()

    df_labeled = pd.DataFrame(data=freshlyChopped, columns=cols, index=rows)
    print(df_labeled.shape)

    return df_labeled
