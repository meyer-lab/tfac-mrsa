"""Data import and processing for the MRSA data"""
from os.path import join, dirname, abspath
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

PATH_HERE = dirname(dirname(abspath(__file__)))

OPTIMAL_SCALING = 2 ** 0.5


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
def import_validation_patient_metadata():
    """
    Returns validation patient meta data, including cohort and outcome.

    Returns:
        patient_data (pandas.DataFrame): Validation patient outcomes and cohorts
    """
    patient_data = pd.read_csv(
        join(PATH_HERE, 'tfac', 'data', 'mrsa', 'validation_patient_metadata.txt'),
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

    plasma_cyto['IL-12(p70)'] = np.clip(plasma_cyto['IL-12(p70)'], 1.0, np.inf)
    serum_cyto['IL-12(p70)'] = np.clip(serum_cyto['IL-12(p70)'], 1.0, np.inf)

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
    return cyto


def import_rna(scale_rna=False):
    """
    Return RNA expression data.

    Parameters:
        trim_low (bool, default:True): remove genes with low expression counts
        scale_rna (bool, default:True): zero-mean scale RNA expression

    Returns:
        rna (pandas.DataFrame): RNA expression counts
    """
    rna = pd.read_csv(
        join(PATH_HERE, 'tfac', 'data', 'mrsa', 'rna_modules_combat.txt'),
        delimiter=',',
        index_col=0,
        engine="c",
        dtype="float64"
    )
    rna.index = rna.index.astype("int32")

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


def form_limit_tensor():
    """
    Forms a tensor of the limits of detection for each cytokine across cytokine
    sources and cohorts.

    Returns:
        limit_tensor (pandas.DataFrame): detection limits for each cytokine
            across serum/plasma sources; set to be highest detection limit
            across cohorts
    """
    limit_tensor = np.load(
        join(PATH_HERE, 'tfac', 'data', 'mrsa', 'LoD_tensor.pkl'),
        allow_pickle=True
    )

    return np.copy(limit_tensor)


@lru_cache
def form_tensor(variance_scaling: float = OPTIMAL_SCALING):
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

    serum_cyto = add_missing_columns(serum_cyto, patients).to_numpy(dtype=float)
    plasma_cyto = add_missing_columns(plasma_cyto, patients).to_numpy(dtype=float)
    rna = add_missing_columns(rna, patients).to_numpy(dtype=float)

    tensor = np.stack(
        (serum_cyto, plasma_cyto)
    ).T

    tensor /= np.nanvar(tensor)
    rna /= np.nanvar(rna)

    return np.copy(tensor * variance_scaling), np.copy(rna.T), patient_data
