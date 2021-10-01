import pandas as pd
import numpy as np
import mygene
from .dataImport import form_tensor, import_rna
from tensorpac import perform_CMTF
from .figures.figureCommon import OPTIMAL_SCALING

path = "tfac/data/mrsa/"


def export_mrsa_inputfile(export=False):
    """ Select top 10% genes per component and export csv file to run clusterProfiler in R """
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


def translate_gene_sets(gs, path, f):
    mg = mygene.MyGeneInfo()
    for r in range(gs.shape[0]):
        xx = gs.iloc[r, :].dropna()[2:]
        genes = list(gs.iloc[r, :].dropna()[2:])
        gIDX = list(xx.index)
        gg = mg.querymany(genes, scopes="symbol", fields="entrezgene", species="human", returnall=False, as_dataframe=True)
        aa = dict(zip(list(gg.index), list(gg["entrezgene"])))
        gs.iloc[r, gIDX] = [aa[g] for g in genes]
        gs.to_csv(path + f + ".csv")
