"""
Creates Figure 5 -- Components vs. Subjects
"""
from os.path import abspath, dirname

import numpy as np
import pandas as pd
import seaborn as sns

from .figureCommon import getSetup, OPTIMAL_SCALING
from ..dataImport import form_tensor, import_cytokines
from tensorpack import perform_CMTF

N_BOOTSTRAP = 30
PATH_HERE = dirname(dirname(abspath(__file__)))
TARGETS = ['status', 'gender', 'race', 'age']


def tfac_setup():
    """
    Import cytokine data and correlate tfac components to cytokines and
    data sources.

    Parameters:
        None

    Returns:
        subjects (pandas.DataFrame): patient correlations to tfac components
        cytos (pandas.DataFrame): cytokine correlations to tfac components
        source (pandas.DataFrame): cytokine source correlations to tfac
            components
        pat_info (pandas.DataFrame): patient meta-data
    """
    tensor, matrix, pat_info = form_tensor(OPTIMAL_SCALING)
    plasma, _ = import_cytokines()
    cytokines = plasma.index

    pat_info.loc[:, 'sorted'] = range(pat_info.shape[0])
    pat_info = pat_info.sort_values(['cohort', 'type', 'status'])
    sort_idx = pat_info.loc[:, 'sorted']
    pat_info = pat_info.drop('sorted', axis=1)
    pat_info = pat_info.T

    factors = perform_CMTF(tensor, matrix)
    col_names = [f"Cmp. {i}" for i in np.arange(1, factors.rank + 1)]
    subjects = pd.DataFrame(
        factors.factors[0][sort_idx, :],
        columns=col_names,
        index=[str(x) for x in pat_info.columns]
    )
    cytos = pd.DataFrame(
        factors.factors[1],
        columns=col_names,
        index=cytokines
    )
    source = pd.DataFrame(
        factors.factors[2],
        columns=col_names,
        index=["Serum", "Plasma"]
    )

    return subjects, cytos, source, pat_info


def plot_results(subjects, pat_info):
    """
    Plots component associations to subjects and metadata characteristics.

    Parameters:
        subjects (pandas.DataFrame): patient correlations to tfac components
        pat_info (pandas.DataFrame): patient meta-data

    Returns:
        fig (plt.figure): figure containing heatmaps of CMTF components vs.
            patients
    """
    fig_size = (5, 5)
    layout = {
        'ncols': 1,
        'nrows': 1,
        'wspace': 0
    }
    axs, fig, gs = getSetup(
        fig_size,
        layout,
        style=None
    )
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    subgrid = gs[0].subgridspec(
        ncols=13,
        nrows=1,
        width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 30, 1, 1],
        wspace=0
    )

    for col in range(subgrid.ncols):
        fig.add_subplot(subgrid[col])

    axs = fig.axes
    for ax in axs:
        ax.set_frame_on(False)

    spacers = [1, 2, 3, 5, 7, 13]
    for spacer in spacers:
        axs[spacer].set_xticks([])
        axs[spacer].set_yticks([])

    sns.heatmap(
        subjects,
        cmap="PRGn",
        center=0,
        xticklabels=True,
        yticklabels=False,
        cbar_ax=axs[12],
        vmin=subjects.values.min(),
        vmax=subjects.values.max(),
        ax=axs[11]
    )

    # Set up subject colorbars
    outcome_colors = ["gray", "lightgreen", "brown"]
    cohort_colors = ["deeppink", "orchid", "pink"]
    type_colors = \
        ["black", "orange", "purple", "green", "red", "yellow", "blue"]
    outcome_cmap = sns.color_palette(outcome_colors)
    cohort_cmap = sns.color_palette(cohort_colors)
    type_cmap = sns.color_palette(type_colors)

    # Data types bar
    types = pd.DataFrame(pat_info.loc["type"]).set_index("type")
    types["Type"] = 0
    types[types.index == "0Serum"] = 6
    types[types.index == "1Plasma"] = 5
    types[types.index == "2RNAseq"] = 4
    types[types.index == "0Serum1Plasma"] = 3
    types[types.index == "0Serum2RNAseq"] = 2
    types[types.index == "1Plasma2RNAseq"] = 1
    types[types.index == "0Serum1Plasma2RNAseq"] = 0

    sns.heatmap(
        types,
        ax=axs[8],
        cbar_ax=axs[2],
        yticklabels=False,
        xticklabels=False,
        cmap=type_cmap
    )
    colorbar = axs[8].collections[0].colorbar
    colorbar.set_ticks(np.linspace(.4, 5.6, 7))
    axs[2].set_yticklabels(
        [
            "All types", "Plasma\nRNAseq", "Serum\nRNAseq", "Serum\nPlasma",
            "RNAseq", "Plasma", "Serum"
        ],
        va="center",
        rotation=90
    )
    axs[2].yaxis.set_ticks_position('left')
    axs[8].set_ylabel("")

    # Cohort bar
    cohort = pd.DataFrame(pat_info.loc["cohort"]).set_index("cohort")
    cohort["Cohort"] = 0
    cohort[cohort.index == 1] = 2
    cohort[cohort.index == 2] = 1
    cohort[cohort.index == 3] = 0

    sns.heatmap(
        cohort,
        ax=axs[9],
        cbar_ax=axs[4],
        yticklabels=False,
        xticklabels=False,
        cmap=cohort_cmap
    )
    colorbar = axs[9].collections[0].colorbar
    colorbar.set_ticks([.33, 1, 1.66])
    axs[4].yaxis.set_ticklabels(
        ["Cohort 3", "Cohort 2", "Cohort 1"],
        va="center")
    axs[4].yaxis.set_ticks_position('left')
    axs[4].yaxis.set_tick_params(rotation=90)
    axs[9].set_ylabel("")

    # Outcome bar
    outs = pd.DataFrame(pat_info.loc["status"]).set_index("status")
    outs["Outcome"] = 0
    outs[outs.index == "0"] = 1
    outs[outs.index == "1"] = 2
    outs[outs.index == "Unknown"] = 0

    sns.heatmap(
        outs,
        ax=axs[10],
        cbar_ax=axs[6],
        yticklabels=False,
        xticklabels=False,
        cmap=outcome_cmap
    )
    colorbar = axs[10].collections[0].colorbar
    colorbar.set_ticks([.33, 1, 1.66])
    axs[6].yaxis.set_ticklabels(
        ["Unknown", "Resolver", "Persister"],
        va='center'
    )
    axs[6].yaxis.set_ticks_position('left')
    axs[6].yaxis.set_tick_params(rotation=90)
    axs[10].set_ylabel("")

    return fig


def makeFigure():
    subjects, _, _, pat_info = tfac_setup()
    fig = plot_results(subjects, pat_info)

    return fig
