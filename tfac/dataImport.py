"""Data import and processing for the MRSA data"""
from os.path import join, dirname, abspath
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

PATH_HERE = dirname(dirname(abspath(__file__)))


@lru_cache
def import_patient_metadata():
    """
    Returns patient meta data, including cohort and outcome.

    Returns:
        patient_data (pandas.DataFrame): Patient outcomes and cohorts
    """
    patient_data = pd.read_csv(
        join(PATH_HERE, 'tfac', 'data', 'mrsa', 'patient_metadata.txt'),
        delimiter=',',
        index_col=0
    )

    return patient_data


@lru_cache
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


@lru_cache
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
        index_col=0,
        engine="c"
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
    Adds patients that do not appear in data as empty columns (all NaNs);
    removes any patients in data not present in patients.

    Parameters:
        data (pandas.DataFrame): cytokine/RNA data
        patients (iterable): patients that must appear in data

    Returns:
        data (pandas.DataFrame): cytokine/RNA data with missing columns
            added; sorted by patient numbers
    """
    # Remove patients who are missing outcome labels
    shared = set(data.columns) & set(patients)
    data = data.loc[:, shared]

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


def form_tensor(variance_scaling: float = 1.0):
    """
    Forms a tensor of cytokine data and a matrix of RNA expression data for
    CMTF decomposition.

    Parameters:
        variance_scaling (float, default:1.0): RNA/cytokine variance scaling

    Returns:
        tensor (numpy.array): tensor of cytokine data
        matrix (numpy.array): matrix of RNA expression data
        patient_data (pandas.DataFrame): patient data, including status, data
            types, and cohort
    """
    plasma_cyto, serum_cyto = import_cytokines()
    rna = import_rna()
    patient_data = import_patient_metadata()
    patients = set(patient_data.index)

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
