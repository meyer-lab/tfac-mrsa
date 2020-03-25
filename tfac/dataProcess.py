"""Container for the entire data preprocessing procedure"""
import numpy as np
import pandas as pd
from synapseclient import Synapse
from sklearn.preprocessing import scale
from .dataOrg import filterData


def DataWorkFlow(username, password, threshold):
    """Contain the full preprocessing procedure in one function"""
    # import data from synapse
    syn = Synapse()
    syn.login(username, password)
    meth0 = pd.read_csv(syn.get('syn21533071').path, index_col=0)
    copy0 = pd.read_csv(syn.get('syn21533091').path, index_col=0)
    gene0 = pd.read_csv(syn.get('syn21533090').path, index_col=0)

    # run filter - aligns all 3 data sets to the same size
    meth1, gene1, copy1 = filterData(meth0, gene0, copy0)

    # cut missing values from methylation data
    meth1_5 = cutMissingValues(meth1, threshold)

    # redo filter to make the data sets the same size again
    meth2, gene2, copy2 = filterData(meth1_5, gene1, copy1)

    return meth2, gene2, copy2


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
