"""Data import and processing for the MRSA data"""
from os.path import join, dirname, abspath
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from tensorpack import perform_CMTF

PATH_HERE = dirname(dirname(abspath(__file__)))

OPTIMAL_SCALING = 2 ** 3.0


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

    # Drop patients with only RNAseq
    patient_data = patient_data.loc[patient_data["type"] != "2RNAseq", :]

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
def import_cytokines(scale_cyto=True, transpose=True, fix_lod=True):
    """
    Return plasma and serum cytokine data.

    Parameters:
        scale_cyto (bool, default:True): scale cytokine values
        transpose (bool): transpose resulting cytokine matrices
        fix_lod (bool): sets consistent LoD across cohorts

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

    if fix_lod:
        limits = form_limit_tensor()
        limits = np.nan_to_num(limits, 0)
        minimums = limits.max(axis=0)
        plasma_cyto.clip(lower=minimums[:, 1], axis=1, inplace=True)
        serum_cyto.clip(lower=minimums[:, 0], axis=1, inplace=True)

    plasma_cyto['IL-12(p70)'] = np.clip(plasma_cyto['IL-12(p70)'], 1.0, np.inf)
    serum_cyto['IL-12(p70)'] = np.clip(serum_cyto['IL-12(p70)'], 1.0, np.inf)

    # IL-3 is almost entirely missing
    plasma_cyto.drop("IL-3", axis=1, inplace=True)
    serum_cyto.drop("IL-3", axis=1, inplace=True)

    if scale_cyto:
        plasma_cyto = plasma_cyto.transform(np.log)
        plasma_cyto -= plasma_cyto.mean(axis=0)
        serum_cyto = serum_cyto.transform(np.log)
        serum_cyto -= serum_cyto.mean(axis=0)

    # If a sample isn't in the metadata, remove it from the cytokines
    patients = set(import_patient_metadata().index)
    plasma_cyto = plasma_cyto.reindex(set(plasma_cyto.index).intersection(patients))
    serum_cyto = serum_cyto.reindex(set(serum_cyto.index).intersection(patients))

    if transpose:
        plasma_cyto = plasma_cyto.T
        serum_cyto = serum_cyto.T

    return plasma_cyto, serum_cyto


def import_rna():
    """
    Return RNA expression modules.

    Returns:
        rna (pandas.DataFrame): RNA expression modules
    """
    rna = pd.read_csv(
        join(PATH_HERE, 'tfac', 'data', 'mrsa', 'tpm_modules.txt'),
        delimiter=',',
        index_col=0,
        engine="c",
        dtype="float64"
    )
    rna.index = rna.index.astype("int32")

    # Always scale
    rna.loc[:, :] = scale(rna.to_numpy())

    return rna


def form_limit_tensor():
    """
    Forms a tensor of the limits of detection for each cytokine across cytokine
    sources and cohorts.

    Returns:
        limit_tensor (numpy.ndarray): tensor of detection limits for each
        cytokine and patient; same shape as the tensor produced via form_tensor
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
    plasma_cyto, serum_cyto = import_cytokines(transpose=False)
    rna = import_rna()
    patient_data = import_patient_metadata()

    serum_cyto = serum_cyto.reindex(patient_data.index).to_numpy(dtype=float).T
    plasma_cyto = plasma_cyto.reindex(patient_data.index).to_numpy(dtype=float).T
    rna = rna.reindex(patient_data.index).to_numpy(dtype=float)

    tensor = np.stack(
        (serum_cyto, plasma_cyto)
    ).T

    # Put on similar scale
    tensor = tensor / np.nanvar(tensor) * variance_scaling
    rna /= np.nanvar(rna)

    assert tensor.shape[0] == rna.shape[0]
    assert tensor.shape[2] == 2
    assert tensor.ndim == 3
    return np.copy(tensor), np.copy(rna), patient_data


@lru_cache
def get_factors(variance_scaling: float = OPTIMAL_SCALING, r=8):
    """
    Return the factorization results.

    Parameters:
        variance_scaling (float, default:1.0): RNA/cytokine variance scaling

    Returns:
        tfac (tl.CP): The factorization results
        patient_data (pandas.DataFrame): patient data, including status, data
            types, and cohort
    """
    tensor, rna, patient_data = form_tensor(variance_scaling)
    t_fac = perform_CMTF(tensor, rna, r=r, maxiter=800, progress=False)
    return t_fac, patient_data
