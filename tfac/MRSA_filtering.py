import pandas as pd
import numpy as np 

def export_mrsa_inputfile(PathToFile, OutputPath):
    """Select top 10% genes per component and export csv file to run clusterProfiler in R"""
    d = pd.read_csv(PathToFile)
    d = pd.melt(frame=d, id_vars="Unnamed: 0", value_vars=d.columns[1:], var_name="Components", value_name="Weights")

    dfs = []
    for ii in set(d["Components"]):
        c = d[d["Components"] == ii]
        up = np.percentile(c["Weights"], 90)
        low = np.percentile(c["Weights"], 10)
        c = c[(c["Weights"] >= up) | (c["Weights"] <= low)]
        dfs.append(c)

    pd.concat(dfs).to_csv(OutputPath)