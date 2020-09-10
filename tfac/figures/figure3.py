"""
This creates Figure 3 - Deconvolution-Component Correlation
"""
import pickle
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import get_patient_info, produce_outcome_bools

cell_df = fig_4_setup()

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((18, 9), (1, 2))
    b = sns.scatterplot(data=cell_df, x='Component A', y='A', hue="Outcomes", ax=ax[0], s=70)
    b.set_xlabel("Component A", fontsize=25)
    b.set_ylabel("Active CD4+ Ts and M2 Macrophage minus T regs", fontsize=25)
    b.tick_params(labelsize=20)
    ax[0].legend(fontsize=25, loc='lower left')
    b = sns.scatterplot(data=cell_df, x='Component B', y='B', hue="Outcomes", ax=ax[1], s=70)
    b.set_xlabel("Component B", fontsize=25)
    b.set_ylabel("Active Mast Cells, Plasma Cells, and M1 Macrophages", fontsize=25)
    b.tick_params(labelsize=20)
    ax[1].legend(fontsize=25, loc='upper left')
    # Add subplot labels
    subplotLabel(ax)

    return f


def fig_4_setup():
    patient_matrices, _, _, deconv = pickle.load(open("MRSA_pickle.p", "rb"))
    cohort_ID, statusID = get_patient_info()
    outcomes = produce_outcome_bools(statusID)
    cytoA = patient_matrices[1][2].T[8]
    cytoB = patient_matrices[1][2].T[32]
    cyto_df = pd.DataFrame([cytoA, cytoB, outcomes]).T
    cyto_df.columns = ["Component A", "Component B", "Outcomes"]
    ids = [i[-4:] for i in cohort_ID]
    cyto_df["ID"] = ids
    outs = []
    for idx in range(cyto_df.shape[0]):
        if cyto_df.iloc[idx, 2] == 0:
            outs.append("Persister",)
        else:
            outs.append("Resolver")
    cyto_df["Outcomes"] = outs
    cyto_df = cyto_df.drop([25, 43, 46])
    T_mem_active = (deconv["T cell CD4+ memory activated"].values - deconv["T cell CD4+ memory activated"].mean())/deconv["T cell CD4+ memory activated"].std()
    T_regs = (deconv["T cell regulatory (Tregs)"].values - deconv["T cell regulatory (Tregs)"].mean())/deconv["T cell regulatory (Tregs)"].std()
    Mast_active = (deconv["Mast cell activated"].values - deconv["Mast cell activated"].mean())/deconv["Mast cell activated"].std()
    B_plasma = (deconv["B cell plasma"].values - deconv["B cell plasma"].mean())/deconv["B cell plasma"].std()
    Mac_1 = (deconv["Macrophage M1"].values - deconv["Macrophage M1"].mean())/deconv["Macrophage M1"].std()
    Mac_2 = (deconv["Macrophage M2"].values - deconv["Macrophage M2"].mean())/deconv["Macrophage M2"].std()
    cyto_df["A"] = T_mem_active - T_regs + Mac_2
    cyto_df["B"] = Mast_active + B_plasma + Mac_1
    
    return cyto_df
