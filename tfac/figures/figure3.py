"""
This creates Figure 3 - Deconvolution-Component Correlation
"""
import pickle
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import get_patient_info, produce_outcome_bools


def fig_3_setup():
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
    Cytotoxic_T = (deconv["T cell CD8+"].values - deconv["T cell CD8+"].mean())/deconv["T cell CD8+"].std()
    T_mem_active = (deconv["T cell CD4+ memory activated"].values - deconv["T cell CD4+ memory activated"].mean())/deconv["T cell CD4+ memory activated"].std()
    T_regs = (deconv["T cell regulatory (Tregs)"].values - deconv["T cell regulatory (Tregs)"].mean())/deconv["T cell regulatory (Tregs)"].std()
    T_naive = (deconv["T cell CD4+ naive"].values - deconv["T cell CD4+ naive"].mean())/deconv["T cell CD4+ naive"].std()
    T_mem_resting = (deconv["T cell CD4+ memory resting"].values - deconv["T cell CD4+ memory resting"].mean())/deconv["T cell CD4+ memory resting"].std()
    T_follicular = (deconv["T cell follicular helper"].values - deconv["T cell follicular helper"].mean())/deconv["T cell follicular helper"].std()
    Mast_active = (deconv["Mast cell activated"].values - deconv["Mast cell activated"].mean())/deconv["Mast cell activated"].std()
    Mac_0 = (deconv["Macrophage M0"].values - deconv["Macrophage M0"].mean())/deconv["Macrophage M0"].std()
    Mac_1 = (deconv["Macrophage M1"].values - deconv["Macrophage M1"].mean())/deconv["Macrophage M1"].std()
    cyto_df["A"] = T_mem_active + Cytotoxic_T + T_follicular - T_naive - T_mem_resting - T_regs
    cyto_df["B"] = Mast_active + Mac_0 + Mac_1
    
    return cyto_df


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    cell_df = fig_3_setup()
    # Get list of axis objects
    ax, f = getSetup((18, 9), (1, 2))
    b = sns.scatterplot(data=cell_df, x='Component A', y='A', hue="Outcomes", ax=ax[0], s=70)
    b.set_xlabel("Component A", fontsize=25)
    b.set_ylabel("Active T Cells minus T regs/Inactive T cells", fontsize=25)
    b.tick_params(labelsize=20)
    ax[0].legend(fontsize=25, loc='lower left')
    b = sns.scatterplot(data=cell_df, x='Component B', y='B', hue="Outcomes", ax=ax[1], s=70)
    b.set_xlabel("Component B", fontsize=25)
    b.set_ylabel("Active Mast Cells, M0, and M1 Macrophages", fontsize=25)
    b.tick_params(labelsize=20)
    ax[1].legend(fontsize=25, loc='upper left')
    # Add subplot labels
    subplotLabel(ax)

    return f
