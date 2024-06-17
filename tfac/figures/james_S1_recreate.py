"""
Recreate plot 1 of figureS1 without local imports
"""
from os.path import join, dirname, abspath
import pandas as pd
import numpy as np
from scipy import sparse

from scipy.stats import pearsonr
import seaborn as sns

from tfac.figures.common import getSetup

"""
Since we will be grabbing data from files present in tfac-mrsa,
we need to define our starting point. home/jamesp/Playground/tfac-mrsa
"""
PATH_HERE = dirname(dirname(dirname(abspath(__file__))))
OPTIMAL_SCALING = 2 ** 7.0

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
# #debug
# print(import_patient_metadata())

def import_cytokines(scale_cyto=True, transpose=True):
    """
    Return 2 matrices containing 1) plasma_ctyo data and 2) serum_cyto data
    
    Parameters:
        scale default:True | log scale the concentrations
        transpose default:True | want cytokine labels as indeces (axis=0) 
    
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

    """
    Not only is IL-3 low relative to the others (x3 smaller minimum)
    it is also suspected that it was below the detection limit of the
    machine.
    """
    plasma_cyto.drop("IL-3", axis=1, inplace=True)
    serum_cyto.drop("IL-3", axis=1, inplace=True)

    if scale_cyto:
        plasma_cyto = plasma_cyto.transform(np.log)
        plasma_cyto -= plasma_cyto.mean(axis=0)
        serum_cyto = serum_cyto.transform(np.log)
        serum_cyto -= serum_cyto.mean(axis=0)

    """
    If sample isn't represnted in patients, remove it from cytokines.
    Do this by reindexing cyto data by the intersection of patient index
    and cyto index
    """
    patients = set(import_patient_metadata().index)
    plasma_cyto = plasma_cyto.reindex(set(plasma_cyto.index).intersection(patients))
    serum_cyto = serum_cyto.reindex(set(serum_cyto.index).intersection(patients))
    print(f"plasma_cyto shape post-importation, pre-transpose: {plasma_cyto.shape}")
    print(f"serum_cyto shape post-importation, pre-transpose: {serum_cyto.shape}")


    # transpose by default so that we expect cytokine labels as indeces (axis=0)
    if transpose:
        plasma_cyto = plasma_cyto.T
        serum_cyto = serum_cyto.T

    return plasma_cyto, serum_cyto

# # debug line
# plasma_cyto, serum_cyto = import_cytokines()


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
    plasma_cyto, serum_cyto = import_cytokines(transpose=False)
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
    # we're only interested in plasma cytokines so we shove RNA into _
    tensor, patInfo = form_tensor()
    plasma, _ = import_cytokines() # we just need cytokine list

    # collect cytokines from the labels of the plasma data and the number
    cytokines = plasma.index
    n_cytokines = len(cytokines)

    print(f"tensor is shape {tensor.shape} after creation")
    tensor = tensor.T
    patInfo = patInfo.T
    serum_slice = tensor[0, :, :]
    plasma_slice = tensor[1, :, :]
    print(f"tensor is shape {tensor.shape} during slice collection (mode 1)")

    # concatenate the serum and plasma slices across the index (37,177)+(37,177)=(74, 177)
    test = pd.concat([pd.DataFrame(serum_slice), pd.DataFrame(plasma_slice)])
    print(f"concatenated serum+plasma slices shape: {test.shape}")

    # drop any patients (axis=1) that only have either serum or plasma cytokine (axis=0) data
    test = test.dropna(axis=1)
    # we are trying to generate the pearson coefficients between serum and plasma

    """
    Setup a pearson list, then for every i in 34 cytokines append the
    cytokine string and pearsonr calc of test row i (index i) with
    test row i+n_cytokines (+n_cyto to get to plasma concat section). Do
    this while ensuring everything is a numpy float in the pears list.
    Turn pears back into a pandas DataFrame
    """
    pears = []
    for i in range(n_cytokines):
        print(f"appending {cytokines[i]}: {pearsonr(test.iloc[i, :].to_numpy(dtype=float), test.iloc[i + n_cytokines, :].to_numpy(dtype=float))[0]}")
        pears.append([cytokines[i], pearsonr(test.iloc[i, :].to_numpy(dtype=float), test.iloc[i + n_cytokines, :].to_numpy(dtype=float))[0]])
    pears = pd.DataFrame(pears).sort_values(1) # sort by incerasing pearson correlation (column 1)
    print(f"shape of pears: {pears.shape}")

    return pears, serum_slice, plasma_slice, cytokines, patInfo

# #debug
# fig_S1_setup()

def cytokine_boxplot(cyto_slice, cytokines, patInfo:pd.DataFrame, axx):
    ser = pd.DataFrame(cyto_slice, index=cytokines, columns=patInfo.columns).T
    print(f"Shape of ser before drop of na and concat with patInfo: {ser.shape}")
    patInfo = patInfo.T["status"] # strips patInfo of everything but "status" index, then transposes

    """
    We're going to add patient status into the ser dataFrame,
    remove NaN values (patients 177->129), reset_index to 0->... 
    and adding the old index into new column, then melt columns except
    identifiers "sid" and "status" (expands the data across cytokines
    [129*37=4773])
    """
    ser = ser.join(patInfo).dropna().reset_index().melt(id_vars=["sid", "status"])
    print(f"shape of ser post-op: {ser.shape}")

    # feed ser into seaborn boxplot 
    b = sns.boxplot(
        data=ser,
        x="variable",
        y="value",
        hue="status",
        linewidth=1,
        ax=axx
    )
    b.set_xticklabels(b.get_xticklabels(), rotation=30, ha="right")
    b.set_xlabel("Cytokine")
    b.set_ylabel("Normalized cytokine level")

def makeFigure():
    """Now we are adding the boxplots in"""

    # list of axis objects (plots)
    fig_size = (8, 8)
    layout = {
        "ncols": 1,
        "nrows": 3
    } # adding 2 more rows for the boxplot subplots.

    """
    getSetup() takes in fig_size (in inchest) and layout
    (how subplots are organized). It makes a figure with those dimensions
    using plt.figure() and gridspec.Gridspec(). Then, if x (index) is not 
    in empts or multz (leave space empty or span subplots), it makes an 
    ax list object that runs through nrows*ncols=total_subplots,
    adding a figure subplot for each by indexing Gridspec instance
    to access SubplotSpec (location in the grid nrows(i), ncols(i)).
    I still do not understand multz functionality.
    """
    ax, f, _ = getSetup(
        fig_size,
        layout
    ) # returns ax (storing SubplotSpec[location of subplot]) and figure f

    pears, serum_slice, plasma_slice, cytokines, patInfo = fig_S1_setup()
    a = sns.pointplot(data=pears, x=0, y=1, join=False, ax=ax[0])
    a.set_xticklabels(a.get_xticklabels(), rotation=30, ha="right")
    a.set_xlabel("Cytokine")
    a.set_ylabel("Pearson's correlation")
    a.set_title("Serum-Plasma Cytokine Level Correlation")

    
    cytokine_boxplot(serum_slice, cytokines, patInfo, ax[1])
    ax[1].set_title("Normalized Serum Cytokine Level by Outcome")

    cytokine_boxplot(plasma_slice, cytokines, patInfo, ax[2])
    ax[2].set_title("Normalized Plasma Cytokine Level by Outcome")

    return f

"""
path is relative to the current path in the running terminal,
which may be anywhere to run the .py file of choice using VScode.
It is not necessarily within the same folder as the running .py file.
PATH_HERE is a convenient way to get to the 'root' of the project.
"""
fig = makeFigure()
fig.savefig(f"{PATH_HERE}/output/james/JamesS1ByHand_Full_tfacTest.svg", format="svg")
