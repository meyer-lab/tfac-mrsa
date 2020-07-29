"""
This creates Figure 4 - Cytokine weights.
"""
import pickle
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import form_MRSA_tensor


patient_mats_applied = pickle.load(open("Factors.p", "rb"))
_, cytos, _ = form_MRSA_tensor(1, 1)
cytoA = patient_mats_applied[1][1][0].T[8]
cytoB = patient_mats_applied[1][1][0].T[32]
cyto_df = pd.DataFrame([cytoA, cytoB, cytos]).T
cyto_df.index = cytos
cyto_df.columns = ["Component A", "Component B", "Cytokines"]

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 8), (1, 1))
    b = sns.scatterplot(data=cyto_df, x='Component A', y='Component B', ax=ax[0]) # blue
    b.set_xlabel("Component A", fontsize=20)
    b.set_ylabel("Component B", fontsize=20)
    b.tick_params(labelsize=14)
    label_point(cyto_df, ax[0])

    # Add subplot labels
    subplotLabel(ax)

    return f

def label_point(df, ax):
    """Labels cytokines on plot"""
    for _, point in df.iterrows():
        ax.text(point['Component A']+.002, point['Component B'], str(point['Cytokines']), fontsize=15)
