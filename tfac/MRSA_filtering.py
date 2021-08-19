import pandas as pd
import numpy as np

from .dataImport import form_tensor, import_rna
from .tensor import perform_CMTF
from .figures.figure4 import OPTIMAL_SCALING


def export_mrsa_inputfile(OutputPath):
    """Select top 10% genes per component and export csv file to run clusterProfiler in R"""
    tensor, matrix, _ = form_tensor(OPTIMAL_SCALING)
    rna = import_rna()

    d = perform_CMTF(tensor, matrix)
    d = pd.DataFrame(d.mFactor)
    d.columns = range(1, 10)
    d.loc[:, 'ID'] = rna.index

    d = pd.melt(frame=d, id_vars='ID', value_vars=d.columns[:-1], var_name="Components", value_name="Weights")

    dfs = []
    for ii in set(d["Components"]):
        c = d[d["Components"] == ii]
        up = np.percentile(c["Weights"], 90)
        low = np.percentile(c["Weights"], 10)
        c = c[(c["Weights"] >= up) | (c["Weights"] <= low)]
        dfs.append(c)

    pd.concat(dfs).to_csv(OutputPath)
