"""
Recreate plot 1 of figureS1 without local imports
"""
from os.path import join, dirname, abspath
import pandas as pd
import numpy as np
from scipy import sparse
# plot 1 is the pearson coefficients of serum and plasma cytokine levels

#start by importing cytokine data

"""
Since we will be grabbing data from files present in tfac-mrsa,
we need to define our starting point. home/jamesp/Playground/tfac-mrsa
"""
PATH_HERE = dirname(dirname(dirname(abspath(__file__))))
OPTIMAL_SCALING = 2 ** 7.0

def import_cytokines(scale_cyto=True):
    """
    Return 2 matrices containing 1) plasma_ctyo data and 2) serum_cyto data
    
    Parameters:
        scale default:True log scale the concentrations
    
    Returns:
        plasma_ctyo (pandas.DataFrame)
        serum_cyto (pandas.DataFrame)
    """
    # Read plasma data from txt file defined in path, using "," as delimiter and axis 0 as labels
    plasma_cyto = pd.read_csv(
        join(PATH_HERE, "tfac", "data", "mrsa", "plasma_cytokines.txt"),
        delimiter=",",
        index_col=0
        )
    serum_cyto = pd.read_csv(
        join(PATH_HERE, "tfac", "data", "mrsa", "serum_cytokines.txt"),
        delimiter=",",
        index_col=0
        )
    
    # Why are we setting a concentration floor for IL-12(p70)?
    plasma_cyto["IL-12(p70)"] = np.clip(plasma_cyto["IL-12(p70)"], 1.0, np.inf)
    serum_cyto["IL-12(p70)"] = np.clip(serum_cyto["IL-12(p70)"], 1.0, np.inf)
    # Seems like IL-12(p70) is the only one in scientific notation in miniscule quantities

    # plasma_cyto_percents = plasma_cyto.mean(axis=0)
    # plasma_cyto_percents = ((plasma_cyto_percents - plasma_cyto["IL-3"].mean(axis=0)) / plasma_cyto["IL-3"].mean(axis=0))*100
    # print(plasma_cyto_percents)

    # Drop IL-3 since it's low relative to others, see commented block just above
    plasma_cyto.drop("IL-3", axis=1, inplace=True)
    serum_cyto.drop("IL-3", axis=1, inplace=True)

    if scale_cyto:
        plasma_cyto = plasma_cyto.transform(np.log)
        plasma_cyto -= plasma_cyto.mean(axis=0)
        serum_cyto = serum_cyto.transform(np.log)
        serum_cyto -= serum_cyto.mean(axis=0)

    return plasma_cyto, serum_cyto

# debug line
plasma_cyto, serum_cyto = import_cytokines()

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
#debug
print(import_patient_metadata())

def form_tensor(variance_scaling: float = OPTIMAL_SCALING):
    """
    Form a tensor of all the cytokine data with an RNA expression data
    for analysis and graphing

    Params:
        variance_scaling default:OPTIMAL_SCALING determined to be 2^7

    Returns:
        tensor (numpy.array): tensor of cytokine data
    """
    #import relevant datasets, not doing RNA yet
    plasma_cyto, serum_cyto = import_cytokines()
    patient_data = import_patient_metadata()
    print(f"plasma_cyto shape before reindex: {plasma_cyto.shape} (Patient, Cytokine)")
    print(f"serum_cyto shape before reindex: {serum_cyto.shape} (Patient, Cytokine)")
    print(f"patient data shape: {patient_data.shape} (Patient, Metadata)")
   
    # change our cytokine indexes to patients and flip so we index by cytokine
    plasma_cyto = plasma_cyto.reindex(patient_data.index).to_numpy(dtype=float).T
    serum_cyto = serum_cyto.reindex(patient_data.index).to_numpy(dtype=float).T
    print(f"plasma_cyto shape after reindexing on patients: {plasma_cyto.shape} (Cytokine, Patient)")
    print(f"serum_cyto after reindexing on patients: {serum_cyto.shape} (Cytokine, Patient)")

    # stack into a tensor on unspecified new axis. Why transpose again?
    tensor = np.stack(
        (serum_cyto, plasma_cyto)
    ).T
    print(tensor.shape)

    # scale tensor to perform CMTF
    tensor = tensor / np.nanvar(tensor) * variance_scaling

    # make sure the tensor is the right shape for operations
    assert tensor.shape[2] == 2
    assert tensor.ndim == 3
    return np.copy(tensor), patient_data

def fig_S1_setup():
    tensor, patInfo = form_tensor()

#debug
form_tensor()
