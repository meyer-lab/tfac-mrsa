"""Data import and processing for the MRSA data"""
from os.path import join, dirname, abspath

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

PATH_HERE = dirname(dirname(abspath(__file__)))


def import_patient_metadata(drop_validation=False):
    """
    Returns patient meta data, including cohort and outcome.

    Parameters:
        drop_validation (bool, default:False): drop validation samples

    Returns:
        patient_data (pandas.DataFrame): Patient outcomes and cohorts
    """
    patient_data = pd.read_csv(
        join(PATH_HERE, 'tfac', 'data', 'mrsa', 'patient_metadata.txt'),
        delimiter=',',
        index_col=0
    )

    if drop_validation:
        patient_data = patient_data.loc[patient_data['status'] != 'Unknown']

    return patient_data


def import_cytokines(scale_cyto=True):
    """
    Return plasma and serum cytokine data.

    Parameters:
        scale_cyto (bool, default:True): scale cytokine values

    Returns:
        plasma_cyto (pandas.DataFrame): plasma cytokine data
        serum_cyto (pandas.DataFrame): serum cytokine data
    """
    plasma_cyto = pd.read_csv(
        join(PATH_HERE, 'tfac', 'data', 'mrsa', 'plasma_cytokines.txt'),
        delimiter=',',
        index_col=0
    )
    serum_cyto = pd.read_csv(
        join(PATH_HERE, 'tfac', 'data', 'mrsa', 'serum_cytokines.txt'),
        delimiter=',',
        index_col=0
    )
    patient_data = import_patient_metadata()

    cohort_1 = patient_data.loc[patient_data['cohort'] == 1].index
    c1_plasma = set(cohort_1) & set(plasma_cyto.index)
    c1_serum = set(cohort_1) & set(serum_cyto.index)

    plasma_p70 = [val * 123000000 if val < 1 else val for val in plasma_cyto.loc[c1_plasma, 'IL-12(p70)']]
    serum_p70 = [val * 123000000 if val < 1 else val for val in serum_cyto.loc[c1_serum, 'IL-12(p70)']]
    plasma_cyto.loc[c1_plasma, 'IL-12(p70)'] = plasma_p70
    serum_cyto.loc[c1_serum, 'IL-12(p70)'] = serum_p70

    if scale_cyto:
        plasma_cyto = scale_cytokines(plasma_cyto)
        serum_cyto = scale_cytokines(serum_cyto)

    return plasma_cyto.T, serum_cyto.T


def scale_cytokines(cyto):
    """
    Scales provided cytokine data--performs a log-transform, then
    zero-mean centers.

    Parameters:
        cyto (pandas.DataFrame): cytokine data

    Returns:
        cyto (pandas.DataFrame): scaled cytokine data
    """
    cyto = cyto.transform(np.log)
    cyto -= cyto.mean(axis=0)
    cyto = cyto.sub(cyto.mean(axis=1), axis=0)

    return cyto


def import_rna(trim_low=True, scale_rna=True):
    """
    Return RNA expression data.

    Parameters:
        trim_low (bool, default:True): remove genes with low expression counts
        scale_rna (bool, default:True): zero-mean scale RNA expression

    Returns:
        rna (pandas.DataFrame): RNA expression counts
    """
    rna = pd.read_csv(
        join(PATH_HERE, 'tfac', 'data', 'mrsa', 'rna_expression.txt.zip'),
        delimiter=',',
        index_col=0
    )

    if trim_low:
        high_means = rna.mean(axis=0) > 1.0
        rna = rna.loc[:, high_means]

    if scale_rna:
        columns = rna.columns
        rna = rna.apply(scale, axis=1, result_type='expand')
        rna.columns = columns
        rna = rna.apply(scale, axis=0)

    rna = rna.T.sort_index()

    return rna


def add_missing_columns(data, patients):
    """
    Adds patients that do not appear in data as empty columns (all NaNs).
    
    Parameters:
        data (pandas.DataFrame): cytokine/RNA data
        patients (iterable): patients that must appear in data
        
    Returns:
        data (pandas.DataFrame): cytokine/RNA data with missing columns
            added; sorted by patient numbers
    """
    missing = patients.difference(data.columns)
    data = pd.concat(
        [
            data,
            pd.DataFrame(
                data=np.nan,
                index=data.index,
                columns=missing
            )
        ],
        axis=1
    )
    data = data.sort_index(axis=1)

    return data


def form_tensor(variance_scaling: float = 1.0, drop_validation=False):
    """
    Forms a tensor of cytokine data and a matrix of RNA expression data for
    CMTF decomposition.

    Parameters:
        variance_scaling (float, default:1.0): RNA/cytokine variance scaling
        drop_validation (bool, default:False): drop validation samples

    Returns:
        tensor (numpy.array): tensor of cytokine data
        matrix (numpy.array): matrix of RNA expression data
        patient_data (pandas.DataFrame): patient data, including status, data
            types, and cohort
    """
    plasma_cyto, serum_cyto = import_cytokines()
    rna = import_rna()
    patient_data = import_patient_metadata(drop_validation=drop_validation)
    patients = set(patient_data.index)

    if drop_validation:
        serum_cyto = serum_cyto.loc[:, set(serum_cyto.columns) & set(patients)]
        plasma_cyto = plasma_cyto.loc[:, set(plasma_cyto.columns) & set(patients)]
        rna = rna.loc[:, set(rna.columns) & set(patients)]

    serum_cyto = add_missing_columns(serum_cyto, patients)
    plasma_cyto = add_missing_columns(plasma_cyto, patients)
    rna = add_missing_columns(rna, patients)

    cyto_var = np.linalg.norm(serum_cyto.fillna(0)) + \
        np.linalg.norm(plasma_cyto.fillna(0))
    serum_cyto /= cyto_var
    plasma_cyto /= cyto_var

    rna_var = np.linalg.norm(rna.fillna(0))
    rna /= rna_var
    rna /= variance_scaling
    matrix = rna.to_numpy(dtype=float)
    matrix = matrix.T

    tensor = np.stack(
        (serum_cyto, plasma_cyto)
    ).T

    return tensor, matrix, patient_data


# OLD IMPORTS


def get_C1C2_patient_info():
    """ Return specific patient information for cohorts 1 and 2. """
    dataCohort = pd.read_csv(join(PATH_HERE, "tfac/data/mrsa/mrsa_s1s2_clin+cyto_073018.csv"))
    return dataCohort[["sid", "status", "stype", "cohort"]].sort_values("sid")


def get_C3_patient_info():
    """ Return specific patient information for cohort 3. """
    c3 = pd.read_csv(join(PATH_HERE, "tfac/data/mrsa/metadata_cohort3.csv"))
    known = c3[c3["status"].str.contains("0|1")].astype(int)
    unknown = c3[~c3["status"].str.contains("0|1")]
    c3 = pd.concat([known, unknown])
    return c3.sort_values("sid")


def form_missing_tensor(variance1: float = 1.0):
    """ Create list of normalized data matrices: cytokines from serum, cytokines from plasma, RNAseq. """
    cyto_list, cytokines, dfExp, geneIDs = full_import()
    # Add in NaNs
    temp = pd.concat([cyto_list[0], cyto_list[1], dfExp])
    # Arrange and gather patient info
    temp = temp.append(pd.DataFrame(index=["type"], columns=temp.columns, data=""))
    for col in temp.columns:
        if np.isfinite(temp[col][0]):
            temp.loc["type"][col] += "0Serum"
        if np.isfinite(temp[col][38]):
            temp.loc["type"][col] += "1Plasma"
        if np.isfinite(temp[col][76]):
            temp.loc["type"][col] += "2RNAseq"

    c1_c2_info = get_C1C2_patient_info().set_index("sid").T.drop("stype")
    c3_info = get_C3_patient_info().set_index("sid").T
    c1_info = c1_c2_info.loc[:, c1_c2_info.loc['cohort'] != 2]

    patInfo = pd.concat([c1_info, c3_info], axis=1)
    temp = temp.append(patInfo).sort_values(by=["cohort", "type", "status"], axis=1)
    patInfo = temp.loc[["type", "status", "cohort"]]
    temp = temp.sort_index(axis=1)

    # Assign data to matrices
    dfCyto_serum = temp.iloc[:38, :]
    dfCyto_plasma = temp.iloc[38:76, :]
    dfExp = temp.iloc[76:-3, :]

    serumNumpy = dfCyto_serum.to_numpy(dtype=float)
    plasmaNumpy = dfCyto_plasma.to_numpy(dtype=float)
    expNumpy = dfExp.to_numpy(dtype=float)
    # Eliminate normalization bias
    cytokVar = np.linalg.norm(np.nan_to_num(serumNumpy)) + np.linalg.norm(np.nan_to_num(plasmaNumpy))
    serumNumpy /= cytokVar
    plasmaNumpy /= cytokVar
    expVar = np.linalg.norm(np.nan_to_num(expNumpy))
    expNumpy /= expVar * variance1
    tensor_slices = [serumNumpy, plasmaNumpy, expNumpy]

    # TODO: Combine slices into tensor and matrix
    return tensor_slices, cytokines, geneIDs, patInfo


def full_import():
    """ Imports raw cytokine and RNAseq data for all 3 cohorts. Performs normalization and fixes bad values. """
    # Import cytokines
    dfCyto_c1 = import_C1_cyto()
    dfCyto_c1 = dfCyto_c1.set_index("sid")  # TODO: Move to import function
    dfCyto_c3_serum, dfCyto_c3_plasma = import_C3_cyto()
    # Import RNAseq
    dfExp_c1 = importCohort1Expression()  # TODO: Combine these
    dfExp_c3 = importCohort3Expression()

    # Modify cytokines
    dfCyto_c1.columns = dfCyto_c3_serum.columns  # TODO: Move to import function
    # Fix limit of detection error - bring to next lowest value
    dfCyto_c1["IL-12(p70)"] = [val * 123000000 if val < 1 else val for val in
                               dfCyto_c1["IL-12(p70)"]]  # TODO: Move to import function
    # normalize separately and extract cytokines
    # Make initial data slices
    patInfo = get_C1C2_patient_info()
    dfCyto_c1["type"] = patInfo[patInfo["cohort"] == 1].stype.to_list()
    dfCyto_serum = pd.concat(
        [dfCyto_c1[dfCyto_c1["type"] == "Serum"].T.drop("type"), dfCyto_c3_serum.T], axis=1
    ).astype('float')
    dfCyto_plasma = pd.concat(
        [dfCyto_c1[dfCyto_c1["type"] == "Plasma"].T.drop("type"), dfCyto_c3_plasma.T], axis=1
    ).astype('float')
    cyto_list = [dfCyto_serum, dfCyto_plasma]
    for idx, df in enumerate(cyto_list):
        df = df.transform(np.log)
        df -= df.mean(axis=0)
        df = df.sub(df.mean(axis=1), axis=0)
        cyto_list[idx] = df
    cytokines = dfCyto_serum.index.to_list()

    # Modify RNAseq
    dfExp_c1 = removeC1_dupes(dfExp_c1)
    # Drop genes not shared
    # TODO: WE HERE
    dfExp_c1.sort_values("Geneid", inplace=True)
    dfExp_c3.sort_values("Geneid", inplace=True)
    dfExp = pd.concat([dfExp_c1, dfExp_c3], axis=1, join="inner")
    # Remove those with very few reads on average
    # TODO: Decide if this is right.
    dfExp = dfExp[np.mean(dfExp.to_numpy(), axis=1) > 1.0]
    # Store info for later
    pats = dfExp.columns.astype(int)
    genes = dfExp.index
    # Normalize

    dfExp = scale(dfExp, axis=0)
    dfExp = scale(dfExp, axis=1)
    dfExp = pd.DataFrame(dfExp, index=genes, columns=pats)
    geneIDs = dfExp.index.to_list()
    return cyto_list, cytokines, dfExp, geneIDs


# Combine C1 and C3 cyto import funcs--keep data separate
def import_C1_cyto():
    """ Import cytokine data from clinical data set. """
    coh1 = pd.read_csv(join(PATH_HERE, "tfac/data/mrsa/mrsa_s1s2_clin+cyto_073018.csv"))
    coh1 = coh1[coh1["cohort"] == 1]
    coh1 = pd.concat([coh1.iloc[:, 3], coh1.iloc[:, -38:]], axis=1).sort_values(by="sid")
    return coh1


def import_C3_cyto():
    """Imports the cohort 3 cytokine data, giving separate dataframes for serum and plasma data. They overlap on many paitents."""
    dfCyto_c3 = pd.read_csv(join(PATH_HERE, "tfac/data/mrsa/CYTOKINES.csv"))
    dfCyto_c3 = dfCyto_c3.set_index("sample ID")
    dfCyto_c3 = dfCyto_c3.rename_axis("sid")
    dfCyto_c3_serum = dfCyto_c3[dfCyto_c3["sample type"] == "serum"].copy()
    dfCyto_c3_plasma = dfCyto_c3[dfCyto_c3["sample type"] == "plasma"].copy()
    dfCyto_c3_serum.drop("sample type", axis=1, inplace=True)
    dfCyto_c3_plasma.drop("sample type", axis=1, inplace=True)
    return dfCyto_c3_serum, dfCyto_c3_plasma


# Remove
def importCohort1Expression():
    """ Import expression data. """
    df = pd.read_table(join(PATH_HERE, "tfac/data/mrsa/expression_counts_cohort1.txt.xz"), compression="xz")
    df.drop(["Chr", "Start", "End", "Strand", "Length"], inplace=True, axis=1)
    nodecimals = [val[: val.index(".")] for val in df["Geneid"]]
    df["Geneid"] = nodecimals
    return df.set_index("Geneid")


def importCohort3Expression():
    """ Imports RNAseq data for cohort 3, sorted by patient ID. """
    dfExp_c3 = pd.read_csv(join(PATH_HERE, "tfac/data/mrsa/Genes_cohort3.csv"), index_col=0)
    return dfExp_c3.reindex(sorted(dfExp_c3.columns), axis=1)


def removeC1_dupes(df):
    """ Removes duplicate genes from cohort 1 data. There are only a few (approx. 10) out of ~50,000, they are possibly different isoforms. The ones similar to cohort 3 are kept. """
    return df[~df.index.duplicated(keep="first")]
