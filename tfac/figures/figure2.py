"""
This creates Figure 2 - MRSA R2X
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, pyplot as plt
from string import ascii_lowercase
from ..dataImport import form_tensor, import_cytokines
from ..tensor import perform_CMTF


def fig_2_setup():
    """Import and organize R2X and heatmaps"""
    # R2X
    tensor, matrix, patInfo = form_tensor()
    plasma, _ = import_cytokines()
    cytokines = plasma.index

    patInfo.loc[:, 'sorted'] = range(patInfo.shape[0])
    patInfo = patInfo.sort_values(['cohort', 'type', 'status'])
    tensor = tensor[patInfo.loc[:, 'sorted'], :]
    patInfo = patInfo.drop('sorted', axis=1)
    patInfo = patInfo.T

    components = 12
    AllR2X = []
    # Run factorization at each component number up to chosen limit
    for component in range(1, components + 1):
        print(f"Starting decomposition with {component} components.")
        AllR2X.append(perform_CMTF(tensor, matrix, r=component).R2X)

    R2X = pd.DataFrame({"Number of Components": np.arange(1, components + 1), "R2X": AllR2X})

    # Heatmaps
    factors = perform_CMTF(tensor, matrix)

    colnames = [f"Cmp. {i}" for i in np.arange(1, factors.rank + 1)]
    subs = pd.DataFrame(factors.factors[0], columns=colnames, index=[str(x) for x in patInfo.columns])
    cytos = pd.DataFrame(factors.factors[1], columns=colnames, index=cytokines)
    sour = pd.DataFrame(factors.factors[2], columns=colnames, index=["Serum", "Plasma"])

    return R2X, subs, cytos, sour, patInfo


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    R2X, subs, cytos, sour, patInfo = fig_2_setup()
    f = plt.figure(figsize=(20, 7))
    # Width corresponds to plots as such: [R2X, spacer, typecbar, spacer, cohortcbar, spacer, outcomecbar, spacer, type, cohort, outcome, subs, spacer, cbar, spacer, cyto, spacer, source]
    gs = gridspec.GridSpec(1, 18, width_ratios=[35, 9, 1, 6, 1, 6, 1, 1.5, 1.5, 1.5, 1.5, 25, 1, 2, 12, 25, 8, 25], wspace=0)
    # Create axes that will have plots
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[2])
    ax5 = plt.subplot(gs[4])
    ax7 = plt.subplot(gs[6])
    ax9 = plt.subplot(gs[8])
    ax10 = plt.subplot(gs[9])
    ax11 = plt.subplot(gs[10])
    ax12 = plt.subplot(gs[11])
    ax14 = plt.subplot(gs[13])
    ax16 = plt.subplot(gs[15])
    ax18 = plt.subplot(gs[17])
    # Determine scale
    vmin = min(subs.values.min(), cytos.values.min(), sour.values.min()) * .6
    vmax = max(subs.values.max(), cytos.values.max(), sour.values.max()) * .6
    # Plot main graphs
    sns.set(rc={'axes.facecolor': 'whitesmoke'})
    sns.scatterplot(data=R2X, x="Number of Components", y="R2X", ax=ax1)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, ls="--")
    sns.heatmap(subs, cmap="PRGn", center=0, xticklabels=True, yticklabels=False, cbar_ax=ax14, vmin=vmin, vmax=vmax, ax=ax12)
    sns.heatmap(cytos, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=vmin, vmax=vmax, ax=ax16)
    sns.heatmap(sour, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=vmin, vmax=vmax, ax=ax18)
    ax18.set_yticklabels(["Serum", "Plasma"], rotation=0)
    # Set up subject colorbars
    outcome_colors = ["gray", "lightgreen", "brown"]
    cohort_colors = ["deeppink", "orchid", "pink"]
    type_colors = ["black", "orange", "purple", "green", "red", "yellow", "blue"]
    outcome_cmap = sns.color_palette(outcome_colors)
    cohort_cmap = sns.color_palette(cohort_colors)
    type_cmap = sns.color_palette(type_colors)
    # Data types bar
    types = pd.DataFrame(patInfo.loc["type"]).set_index("type")
    types["Type"] = 0
    types[types.index == "0Serum"] = 6
    types[types.index == "1Plasma"] = 5
    types[types.index == "2RNAseq"] = 4
    types[types.index == "0Serum1Plasma"] = 3
    types[types.index == "0Serum2RNAseq"] = 2
    types[types.index == "1Plasma2RNAseq"] = 1
    types[types.index == "0Serum1Plasma2RNAseq"] = 0

    sns.heatmap(
        types, ax=ax9, cbar_ax=ax3, yticklabels=False, xticklabels=True, cmap=type_cmap
    )
    ax9.set_xticklabels(types.columns, rotation=90)
    colorbar = ax9.collections[0].colorbar
    colorbar.set_ticks(np.linspace(.4, 5.6, 7))
    ax3.yaxis.set_ticklabels(["All types", "Plasma/RNAseq", "Serum/RNAseq", "Serum/Plasma", "RNAseq", "Plasma", "Serum"], va="center")
    ax3.yaxis.set_ticks_position('left')
    ax3.yaxis.set_tick_params(rotation=60)
    ax9.set_ylabel("")

    # Cohort bar
    cohort = pd.DataFrame(patInfo.loc["cohort"]).set_index("cohort")
    cohort["Cohort"] = 0
    cohort[cohort.index == 1] = 2
    cohort[cohort.index == 2] = 1
    cohort[cohort.index == 3] = 0

    sns.heatmap(
        cohort, ax=ax10, cbar_ax=ax5, yticklabels=False, xticklabels=True, cmap=cohort_cmap
    )
    ax10.set_xticklabels(cohort.columns, rotation=90)
    colorbar = ax10.collections[0].colorbar
    colorbar.set_ticks([.33, 1, 1.66])
    ax5.yaxis.set_ticklabels(["Cohort 3", "Cohort 2", "Cohort 1"], va="center")
    ax5.yaxis.set_ticks_position('left')
    ax5.yaxis.set_tick_params(rotation=60)
    ax10.set_ylabel("")

    # Outcome bar
    outs = pd.DataFrame(patInfo.loc["status"]).set_index("status")
    outs["Outcome"] = 0
    outs[outs.index == "0"] = 2
    outs[outs.index == "1"] = 1
    outs[outs.index == "Unknown"] = 0

    sns.heatmap(
        outs, ax=ax11, cbar_ax=ax7, yticklabels=False, xticklabels=True, cmap=outcome_cmap
    )
    ax11.set_xticklabels(outs.columns, rotation=90)
    colorbar = ax11.collections[0].colorbar
    colorbar.set_ticks([.33, 1, 1.66])
    ax7.yaxis.set_ticklabels(["Unknown", "Resolver", "Persister"], va='center')
    ax7.yaxis.set_ticks_position('left')
    ax7.yaxis.set_tick_params(rotation=60)
    ax11.set_ylabel("")
    # Titles/labeling
    ax1.set_title("R2X", fontsize=15)
    ax12.set_title("Subjects", fontsize=15)
    ax16.set_title("Cytokines", fontsize=15)
    ax18.set_title("Source", fontsize=15)

    for ii, ax in enumerate([ax1, ax12, ax16, ax18]):
        ax.text(-0.2, 1.1, ascii_lowercase[ii], transform=ax.transAxes, fontsize=25, fontweight="bold", va="top")

    return f
