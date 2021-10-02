"""
Creates Figure 4 -- Model Interpretation
"""
from os.path import abspath, dirname
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.utils import resample

from .figureCommon import getSetup, OPTIMAL_SCALING
from ..dataImport import form_tensor, import_cytokines
from ..predict import run_model, predict_regression
from tensorpac import perform_CMTF

N_BOOTSTRAP = 30
PATH_HERE = dirname(dirname(abspath(__file__)))
TARGETS = ['status', 'gender', 'race', 'age']


def bootstrap_weights():
    """
    Predicts samples with unknown outcomes.

    Parameters:
        None

    Returns:
        weights (pandas.DataFrame): mean and StD of component weights w/r to
            prediction targets
    """
    tensor, matrix, patient_data = form_tensor(OPTIMAL_SCALING)
    patient_data = patient_data.reset_index(drop=True)
    patient_data = patient_data.loc[patient_data['status'] != 'Unknown']

    components = perform_CMTF(tensor, matrix)
    components = components[1][0]
    components = components[patient_data.index, :]

    stats = ['Mean', 'StD']
    index = pd.MultiIndex.from_product([TARGETS, stats])
    weights = pd.DataFrame(
        index=index,
        columns=list(range(1, components.shape[1] + 1))
    )

    for target in TARGETS:
        coef = []
        for sample in range(N_BOOTSTRAP):
            data, labels = resample(components, patient_data.loc[:, target])
            if target == 'age':
                _, _coef = predict_regression(data, labels, return_coef=True)
            else:
                _, _, _coef = run_model(data, labels, return_coef=True)

            coef.append(_coef)

        coef = scale(coef, axis=1)
        weights.loc[(target, 'Mean'), :] = np.mean(coef, axis=0)
        weights.loc[(target, 'StD'), :] = np.std(coef, axis=0, ddof=1)

    return weights


def tfac_setup():
    """
    Import cytokine data and correlate tfac components to cytokines and
    data sources.

    Parameters:
        None

    Returns:
        subs (pandas.DataFrame): patient correlations to tfac components
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
    subs = pd.DataFrame(
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

    return subs, cytos, source, pat_info


def plot_results(weights, subs, cytos, source, pat_info):
    """
    Plots component weights and interpretation.

    Parameters:
        weights (pandas.DataFrame): mean and StD of component weights w/r to
            prediction targets
        subs (pandas.DataFrame): patient correlations to tfac components
        cytos (pandas.DataFrame): cytokine correlations to tfac components
        source (pandas.DataFrame): cytokine source correlations to tfac
            components
        pat_info (pandas.DataFrame): patient meta-data
    """
    fig_size = (20, 7)
    layout = {
        'ncols': 18,
        'nrows': 1,
        'width_ratios': [35, 6, 1, 4, 1, 4, 1, 1.5, 1.5, 1.5, 1.5, 25, 1, 1, 16, 25, 8, 25],
        'wspace': 0
    }
    axs, fig = getSetup(
        fig_size,
        layout,
        style=None
    )
    axs[0].set_frame_on(True)

    # Determine scale
    vmin = min(subs.values.min(), cytos.values.min(), source.values.min())
    vmax = max(subs.values.max(), cytos.values.max(), source.values.max())

    # Plot main graphs
    sns.heatmap(subs, cmap="PRGn", center=0, xticklabels=True, yticklabels=False, cbar_ax=axs[13], vmin=vmin, vmax=vmax, ax=axs[11])
    sns.heatmap(cytos, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=vmin, vmax=vmax, ax=axs[15])
    sns.heatmap(source, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=vmin, vmax=vmax, ax=axs[17])
    axs[17].set_yticklabels(["Serum", "Plasma"], rotation=0)

    # Set up subject colorbars
    outcome_colors = ["gray", "lightgreen", "brown"]
    cohort_colors = ["deeppink", "orchid", "pink"]
    type_colors = ["black", "orange", "purple", "green", "red", "yellow", "blue"]
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
        types, ax=axs[8], cbar_ax=axs[2], yticklabels=False, xticklabels=True, cmap=type_cmap
    )
    axs[8].set_xticklabels(types.columns, rotation=90)
    colorbar = axs[8].collections[0].colorbar
    colorbar.set_ticks(np.linspace(.4, 5.6, 7))
    axs[2].yaxis.set_ticklabels(
        ["All types", "Plasma\nRNAseq", "Serum\nRNAseq", "Serum\nPlasma", "RNAseq", "Plasma", "Serum"],
        va="center"
    )
    axs[2].yaxis.set_ticks_position('left')
    axs[2].yaxis.set_tick_params(rotation=90)
    axs[8].set_ylabel("")

    # Cohort bar
    cohort = pd.DataFrame(pat_info.loc["cohort"]).set_index("cohort")
    cohort["Cohort"] = 0
    cohort[cohort.index == 1] = 2
    cohort[cohort.index == 2] = 1
    cohort[cohort.index == 3] = 0

    sns.heatmap(
        cohort, ax=axs[9], cbar_ax=axs[4], yticklabels=False, xticklabels=True, cmap=cohort_cmap
    )
    axs[9].set_xticklabels(cohort.columns, rotation=90)
    colorbar = axs[9].collections[0].colorbar
    colorbar.set_ticks([.33, 1, 1.66])
    axs[4].yaxis.set_ticklabels(["Cohort 3", "Cohort 2", "Cohort 1"], va="center")
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
        outs, ax=axs[10], cbar_ax=axs[6], yticklabels=False, xticklabels=True, cmap=outcome_cmap
    )
    axs[10].set_xticklabels(outs.columns, rotation=90)
    colorbar = axs[10].collections[0].colorbar
    colorbar.set_ticks([.33, 1, 1.66])
    axs[6].yaxis.set_ticklabels(["Unknown", "Resolver", "Persister"], va='center')
    axs[6].yaxis.set_ticks_position('left')
    axs[6].yaxis.set_tick_params(rotation=90)
    axs[10].set_ylabel("")
    # Titles/labeling
    axs[11].set_title("Subjects")
    axs[15].set_title("Cytokines")
    axs[17].set_title("Source")

    for ii, ax in enumerate([axs[0], axs[11], axs[15], axs[17]]):
        ax.text(
            -0.2,
            1.05,
            ascii_uppercase[ii],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top"
        )

    for offset, target in enumerate(TARGETS):
        axs[0].errorbar(
            weights.loc[(target, 'Mean')],
            range(offset, (len(TARGETS) + 5) * weights.shape[1], len(TARGETS) + 5),
            marker='.',
            xerr=weights.loc[(target, 'StD')],
            linestyle='',
            capsize=2
        )

    axs[0].legend(
        ['Persistence', 'Sex', 'Race', 'Age']
    )

    axs[0].plot(
        [0, 0],
        [-100, 100],
        color='k',
        linestyle='--'
    )

    axs[0].set_xlabel('Model Coefficient')
    axs[0].set_xlim(-3, 3)
    axs[0].set_ylim(-1, 76)
    axs[0].set_yticks(np.arange(1.5, 80, 9))
    axs[0].set_yticklabels(
        [f'Cmp. {i}' for i in range(1, 10)]
    )

    plt.subplots_adjust(left=0.05, right=0.975, top=0.95)

    return fig


def makeFigure():
    weights = bootstrap_weights()
    subs, cytos, source, pat_info = tfac_setup()

    fig = plot_results(weights, subs, cytos, source, pat_info)

    return fig
