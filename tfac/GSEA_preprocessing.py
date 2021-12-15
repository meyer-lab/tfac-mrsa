import pandas as pd
import numpy as np
import mygene
from .dataImport import form_tensor, import_rna
from tensorpack import perform_CMTF

path = "tfac/data/mrsa/"


def export_mrsa_inputfile(export=False):
    """ Select top 10% genes per component and export csv file to run clusterProfiler in R """
    tensor, matrix, _ = form_tensor()
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

    out = pd.concat(dfs)

    if export:
        out.to_csv(path + "MRSA_gsea_input_ENSEMBL.csv")

    return out


def translate_geneIDs(toID="entrezgene", export=False):
    """ Translate gene accessions. In this case to ENTREZID by default. """
    d = export_mrsa_inputfile()
    mg = mygene.MyGeneInfo()
    gg = mg.getgenes(d["ID"], fields=toID, as_dataframe=True)
    d[str(toID)] = gg[toID].values
    out = d.dropna()

    if export:
        out.to_csv(path + "MRSA_gsea_input_ENTREZ.csv")

    return out[[toID, "Components"]]
