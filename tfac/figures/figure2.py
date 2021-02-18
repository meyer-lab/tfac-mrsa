"""
This creates Figure 2 - MRSA R2X
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, pyplot as plt
from string import ascii_lowercase
from .figureCommon import subplotLabel, getSetup
from ..dataImport import form_missing_tensor, get_C1_patient_info, produce_outcome_bools
from ..tensor import perform_TMTF


def fig_2_setup():
    """Import and organize R2X and heatmaps"""
    #R2X
    tensor_slices, cytokines, _, cohortID = form_missing_tensor()
    tensor = np.stack((tensor_slices[0], tensor_slices[1])).T
    matrix = tensor_slices[2].T
    components = 5
    all_tensors = []
    #Run factorization at each component number up to chosen limit
    for component in range(1, components + 1):
        print(f"Starting decomposition with {component} components.")
        all_tensors.append(perform_TMTF(tensor, matrix, r=component))

    AllR2X = [all_tensors[x][2] for x in range(0, components)]
    R2X = pd.DataFrame({"Number of Components": np.arange(1, components + 1), "R2X": AllR2X})
    #Heatmaps
    # TODO: Change once determined by SVC
    factors = perform_TMTF(tensor, matrix, r=2)[0]

    colnames = [f"Cmp. {i}" for i in np.arange(1, factors.rank + 1)]
    subs = pd.DataFrame(factors.factors[0], columns=colnames, index=[str(x) for x in cohortID])
    cytos = pd.DataFrame(factors.factors[1], columns=colnames, index=cytokines)
    sour = pd.DataFrame(factors.factors[2], columns=colnames, index=["Serum", "Plasma"])

    return R2X, subs, cytos, sour


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    R2X, subs, cytos, sour = fig_2_setup()
    # Get list of axis objects
    f = plt.figure(figsize=(20, 7))
    #Width corresponds to plots as such: [R2X, spacer, cohortcbar, spacer, outcomecbar, spacer, cohort, outcome, subs, spacer, cbar, spacer, cyto, spacer, source]
    gs = gridspec.GridSpec(1, 15, width_ratios=[35, 4, 1.5, 4, 1.5, 1, 1, 1, 25, 1, 2, 12, 25, 8, 25], wspace=0)
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[2])
    ax5 = plt.subplot(gs[4])
    ax7 = plt.subplot(gs[6])
    ax8 = plt.subplot(gs[7])
    ax9 = plt.subplot(gs[8])
    ax11 = plt.subplot(gs[10])
    ax13 = plt.subplot(gs[12])
    ax15 = plt.subplot(gs[14])

    vmin = min(subs.values.min(), cytos.values.min(), sour.values.min()) * .6
    vmax = max(subs.values.max(), cytos.values.max(), sour.values.max()) * .6

    sns.set(rc={'axes.facecolor':'whitesmoke'})
    sns.scatterplot(data=R2X, x="Number of Components", y="R2X", ax=ax1)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, ls="--")
    sns.heatmap(subs, cmap="PRGn", center=0, xticklabels=True, yticklabels=False, cbar_ax=ax11, vmin=vmin, vmax=vmax, ax=ax9)
    sns.heatmap(cytos, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=vmin, vmax=vmax, ax=ax13)
    sns.heatmap(sour, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=vmin, vmax=vmax, ax=ax15)
    ax15.set_yticklabels(["Serum", "Plasma"], rotation = 0)

    outcome_colors = ["gray", "green", "red"]
    cohort_colors = ["navy", "skyblue"]
    outcome_cmap = sns.color_palette(outcome_colors)
    cohort_cmap = sns.color_palette(cohort_colors)

    cohort = ["Cohort 1"] * 61 + ["Cohort 3"] * 132
    cohort = pd.DataFrame(cohort)
    cohort = cohort.set_index([0])
    cohort["Cohort"] = 0
    cohort[cohort.index == "Cohort 1"] = 1
    cohort[cohort.index == "Cohort 3"] = 0

    sns.heatmap(
        cohort, ax=ax7, cbar_ax=ax3, yticklabels=False, xticklabels=True, cmap=cohort_cmap
    )
    ax7.set_xticklabels(cohort.columns, rotation=90)
    colorbar = ax7.collections[0].colorbar
    colorbar.set_ticks([0.25, .75])
    ax3.yaxis.set_ticklabels(["Cohort 3", "Cohort 1"], va="center")
    ax3.yaxis.set_ticks_position('left')
    ax3.yaxis.set_tick_params(rotation=90)
    ax7.set_ylabel("")


    df = get_C1_patient_info()
    outs = produce_outcome_bools(df["outcome_txt"])
    outs = outs.tolist() + ["Unknown"] * 132
    outs = pd.DataFrame(outs)
    outs = outs.set_index([0])
    outs["Outcome"] = 0
    outs[outs.index == 0] = 2
    outs[outs.index == 1] = 1
    outs[outs.index == "Unknown"] = 0

    sns.heatmap(
        outs, ax=ax8, cbar_ax=ax5, yticklabels=False, xticklabels=True, cmap=outcome_cmap
    )
    ax8.set_xticklabels(outs.columns, rotation=90)
    colorbar = ax8.collections[0].colorbar
    colorbar.set_ticks([.33, 1, 1.66])
    ax5.yaxis.set_ticklabels(["Unknown", "Resolver", "Persister"], va='center')
    ax5.yaxis.set_ticks_position('left')
    ax5.yaxis.set_tick_params(rotation=90)
    ax8.set_ylabel("")

    ax1.set_title("R2X", fontsize=15)
    ax9.set_title("Subjects", fontsize=15)
    ax13.set_title("Cytokines", fontsize=15)
    ax15.set_title("Source", fontsize=15)

    for ii, ax in enumerate([ax1, ax9, ax13, ax15]):
        ax.text(-0.2, 1.1, ascii_lowercase[ii], transform=ax.transAxes, fontsize=25, fontweight="bold", va="top")

    return f
