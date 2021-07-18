"""Data import and processing for the MRSA data"""
from os.path import join, dirname
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


path_here = dirname(dirname(__file__))


def import_deconv():
    """ Imports and returns cell deconvolution data. """
    return (
        pd.read_csv(join(path_here, "tfac/data/mrsa/deconvo_cibersort_APMB.csv"), delimiter=",", index_col="sample")
        .sort_index()
        .drop(["gender", "cell_type"], axis=1)
    )


def get_patient_info():
    """ Return specific patient information. """
    columns = {'sid': int, 'status': float, 'cohort': int}

    c1 = pd.read_csv("tfac/data/mrsa/mrsa_s1s2_clin+cyto_073018.csv",
                     usecols=columns.keys(), dtype=columns, index_col="sid")
    c1 = c1.loc[c1['cohort'] == 1]

    c3 = pd.read_csv("tfac/data/mrsa/metadata_cohort3.csv",
                     na_values="Unknown", dtype=columns, index_col="sid")

    cs = pd.concat([c1, c3], axis=0)
    cs.sort_values("sid", inplace=True)
    return cs


def form_missing_tensor(variance1: float = 1.0):
    """ Create list of normalized data matrices: cytokines from serum, cytokines from plasma, RNAseq. """
    cyto_list, cytokines, dfExp, geneIDs = full_import()
    # Assign data to matrices
    dfCyto_serum = cyto_list[0].T
    dfCyto_plasma = cyto_list[1].T

    # Arrange and gather patient info
    patInfo = get_patient_info()

    serumNumpy = dfCyto_serum.reindex(patInfo.index).to_numpy(dtype=float)
    plasmaNumpy = dfCyto_plasma.reindex(patInfo.index).to_numpy(dtype=float)
    expNumpy = dfExp.T.reindex(patInfo.index).to_numpy(dtype=float)

    assert serumNumpy.shape[0] == plasmaNumpy.shape[0]
    assert serumNumpy.shape[0] == expNumpy.shape[0]

    patInfo["type"] = ""
    for col in range(patInfo.shape[0]):
        if np.isfinite(serumNumpy[col, 0]):
            patInfo["type"].iloc[col] += "0Serum"
        if np.isfinite(plasmaNumpy[col, 0]):
            patInfo["type"].iloc[col] += "1Plasma"
        if np.isfinite(expNumpy[col, 0]):
            patInfo["type"].iloc[col] += "2RNAseq"

    # Eliminate normalization bias
    cytokVar = np.linalg.norm(np.nan_to_num(serumNumpy)) + np.linalg.norm(np.nan_to_num(plasmaNumpy))
    serumNumpy /= cytokVar
    plasmaNumpy /= cytokVar
    expVar = np.linalg.norm(np.nan_to_num(expNumpy))
    expNumpy /= expVar * variance1
    tensor_slices = [serumNumpy, plasmaNumpy, expNumpy]
    return tensor_slices, cytokines, geneIDs, patInfo


def full_import():
    """ Imports raw cytokine and RNAseq data for all 3 cohorts. Performs normalization and fixes bad values. """
    # Import cytokines
    dfCyto = import_cyto()
    # Import RNAseq
    dfExp_c1 = importCohort1Expression()
    dfExp_c3 = importCohort3Expression()

    # normalize separately and extract cytokines
    # Make initial data slices
    dfCyto_serum = dfCyto.loc[dfCyto["stype"] == "serum"].drop("stype", axis=1)
    dfCyto_plasma = dfCyto.loc[dfCyto["stype"] == "plasma"].drop("stype", axis=1)
    cyto_list = [dfCyto_serum.T, dfCyto_plasma.T]
    for idx, df in enumerate(cyto_list):
        df = df.transform(np.log)
        df -= df.mean(axis=0)
        df = df.sub(df.mean(axis=1), axis=0)
        cyto_list[idx] = df
    cytokines = dfCyto_serum.index.to_list()

    # Removes duplicate genes from cohort 1 data. There are only a few (approx. 10) out of ~50,000, 
    # they are possibly different isoforms. The ones similar to cohort 3 are kept.
    dfExp_c1 = dfExp_c1[~dfExp_c1.index.duplicated(keep="first")]
    # Drop genes not shared
    dfExp_c1.sort_values("Geneid", inplace=True)
    dfExp_c3.sort_values("Geneid", inplace=True)
    dfExp = pd.concat([dfExp_c1, dfExp_c3], axis=1, join="inner")
    # Remove those with very few reads on average
    dfExp = dfExp.loc[np.mean(dfExp.to_numpy(), axis=1) > 2]
    # Store info for later
    pats = dfExp.columns.astype(int)
    genes = dfExp.index
    # Normalize
    dfExp = scale(dfExp, axis=0)
    dfExp = scale(dfExp, axis=1)
    dfExp = pd.DataFrame(dfExp, index=genes, columns=pats)
    geneIDs = dfExp.index.to_list()
    return cyto_list, cytokines, dfExp, geneIDs


def import_cyto():
    """ Import cytokine data. """
    dfCyto_c1 = pd.read_csv("tfac/data/mrsa/mrsa_s1s2_clin+cyto_073018.csv", index_col="sid")
    dfCyto_c1 = dfCyto_c1[dfCyto_c1["cohort"] == 1]
    dfCyto_c1 = pd.concat([dfCyto_c1["stype"], dfCyto_c1.iloc[:, -38:]], axis=1).sort_values(by="sid")
    dfCyto_c3 = pd.read_csv("tfac/data/mrsa/CYTOKINES.csv", index_col="sid")
    dfCyto_c1.columns = dfCyto_c3.columns
    return pd.concat([dfCyto_c1, dfCyto_c3], axis=0)


def importCohort1Expression():
    """ Import expression data. """
    df = pd.read_table(join(path_here, "tfac/data/mrsa/expression_counts_cohort1.txt.xz"), compression="xz")
    df.drop(["Chr", "Start", "End", "Strand", "Length"], inplace=True, axis=1)
    df["Geneid"] = [val[: val.index(".")] for val in df["Geneid"]]
    return df.set_index("Geneid")


def importCohort3Expression():
    """ Imports RNAseq data for cohort 3, sorted by patient ID. """
    dfExp_c3 = pd.read_csv("tfac/data/mrsa/Genes_cohort3.csv", index_col=0)
    return dfExp_c3.reindex(sorted(dfExp_c3.columns), axis=1)
