"""
This creates Figure 4 - Cytokine weights.
"""
import pickle
import pandas as pd
import seaborn as sns
from tensorly.parafac2_tensor import apply_parafac2_projections
from .figureCommon import subplotLabel, getSetup
from ..dataImport import form_MRSA_tensor
from ..explore_factors import label_points


def fig_4_setup():
    patient_matrices, _, _, _ = pickle.load(open("MRSA_pickle.p", "rb"))
    patient_mats_applied = apply_parafac2_projections(patient_matrices)
    _, cytos, _ = form_MRSA_tensor(1, 1)
    cytoA = patient_mats_applied[1][1][0].T[8]
    cytoB = patient_mats_applied[1][1][0].T[32]
    cyto_df = pd.DataFrame([cytoA, cytoB, cytos]).T
    cyto_df.index = cytos
    cyto_df.columns = ["Component A", "Component B", "Cytokines"]
    return cyto_df


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    df = fig_4_setup()
    # Get list of axis objects
    ax, f = getSetup((15, 8), (1, 1))
    b = sns.scatterplot(data=df, x="Component A", y="Component B", ax=ax[0])  # blue
    b.set_xlabel("Component A", fontsize=20)
    b.set_ylabel("Component B", fontsize=20)
    b.tick_params(labelsize=14)
    label_points(df, "Cytokines", ax[0])

    # Add subplot labels
    subplotLabel(ax)

    return f
