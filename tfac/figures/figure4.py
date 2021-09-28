"""
Creates Figure 4 -- Model Interpretation
"""
from os.path import abspath, dirname
from string import ascii_uppercase

from matplotlib import gridspec, pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.utils import resample

from .figureCommon import OPTIMAL_SCALING
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
        weights (pandas.Series): bootstrapped coefficient weights
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
    """Import and organize R2X and heatmaps"""
    # R2X
    tensor, matrix, patInfo = form_tensor(OPTIMAL_SCALING)
    plasma, _ = import_cytokines()
    cytokines = plasma.index

    patInfo.loc[:, 'sorted'] = range(patInfo.shape[0])
    patInfo = patInfo.sort_values(['cohort', 'type', 'status'])
    sortIDX = patInfo.loc[:, 'sorted']
    patInfo = patInfo.drop('sorted', axis=1)
    patInfo = patInfo.T

    components = 9
    AllR2X = []
    # Run factorization at each component number up to chosen limit
    for component in range(1, components + 1):
        print(f"Starting decomposition with {component} components.")
        AllR2X.append(perform_CMTF(tensor, matrix, r=component).R2X)

    R2X = pd.DataFrame({"Number of Components": np.arange(1, components + 1), "R2X": AllR2X})

    # Heatmaps
    factors = perform_CMTF(tensor, matrix)

    colnames = [f"Cmp. {i}" for i in np.arange(1, factors.rank + 1)]
    subs = pd.DataFrame(factors.factors[0][sortIDX, :], columns=colnames, index=[str(x) for x in patInfo.columns])
    cytos = pd.DataFrame(factors.factors[1], columns=colnames, index=cytokines)
    sour = pd.DataFrame(factors.factors[2], columns=colnames, index=["Serum", "Plasma"])

    return R2X, subs, cytos, sour, patInfo


def plot_results(weights):
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    # R2X, subs, cytos, sour, patInfo = tfac_setup()

    # R2X.to_pickle('R2X.pkl')
    # subs.to_pickle('subs.pkl')
    # cytos.to_pickle('cytos.pkl')
    # sour.to_pickle('sour.pkl')
    # patInfo.to_pickle('patInfo.pkl')
    R2X = pd.read_pickle('R2X.pkl')
    subs = pd.read_pickle('subs.pkl')
    cytos = pd.read_pickle('cytos.pkl')
    sour = pd.read_pickle('sour.pkl')
    patInfo = pd.read_pickle('patInfo.pkl')

    fig = plt.figure(figsize=(20, 7))
    # Width corresponds to plots as such: [R2X, spacer, typecbar, spacer, cohortcbar, spacer, outcomecbar, spacer, type, cohort, outcome, subs, spacer, cbar, spacer, cyto, spacer, source]
    gs = gridspec.GridSpec(1, 18, width_ratios=[35, 9, 1, 6, 1, 6, 1, 1.5, 1.5, 1.5, 1.5, 25, 1, 2, 12, 25, 8, 25],
                           wspace=0)
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
    vmin = min(subs.values.min(), cytos.values.min(), sour.values.min())
    vmax = max(subs.values.max(), cytos.values.max(), sour.values.max())
    # Plot main graphs
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
    ax3.yaxis.set_ticklabels(
        ["All types", "Plasma\nRNAseq", "Serum\nRNAseq", "Serum\nPlasma", "RNAseq", "Plasma", "Serum"],
        va="center"
    )
    ax3.yaxis.set_ticks_position('left')
    ax3.yaxis.set_tick_params(rotation=90)
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
    ax5.yaxis.set_tick_params(rotation=90)
    ax10.set_ylabel("")

    # Outcome bar
    outs = pd.DataFrame(patInfo.loc["status"]).set_index("status")
    outs["Outcome"] = 0
    outs[outs.index == "0"] = 1
    outs[outs.index == "1"] = 2
    outs[outs.index == "Unknown"] = 0

    sns.heatmap(
        outs, ax=ax11, cbar_ax=ax7, yticklabels=False, xticklabels=True, cmap=outcome_cmap
    )
    ax11.set_xticklabels(outs.columns, rotation=90)
    colorbar = ax11.collections[0].colorbar
    colorbar.set_ticks([.33, 1, 1.66])
    ax7.yaxis.set_ticklabels(["Unknown", "Resolver", "Persister"], va='center')
    ax7.yaxis.set_ticks_position('left')
    ax7.yaxis.set_tick_params(rotation=90)
    ax11.set_ylabel("")
    # Titles/labeling
    ax12.set_title("Subjects")
    ax16.set_title("Cytokines")
    ax18.set_title("Source")

    for ii, ax in enumerate([ax1, ax12, ax16, ax18]):
        ax.text(-0.2, 1.1, ascii_uppercase[ii], transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    for offset, target in enumerate(TARGETS):
        ax1.errorbar(
            weights.loc[(target, 'Mean')],
            range(offset, (len(TARGETS) + 5) * weights.shape[1], len(TARGETS) + 5),
            marker='.',
            xerr=weights.loc[(target, 'StD')],
            linestyle='',
            capsize=2
        )

    ax1.legend(
        ['Persistence', 'Sex', 'Race', 'Age']
    )

    ax1.plot(
        [0, 0],
        [-100, 100],
        color='k',
        linestyle='--'
    )

    ax1.set_xlabel('Model Coefficient')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1, 76)
    ax1.set_yticks(np.arange(1.5, 80, 9))
    ax1.set_yticklabels(
        [f'Cmp. {i}' for i in range(1, 10)]
    )

    plt.subplots_adjust(left=0.05, right=0.975)

    return fig


def makeFigure():
    # weights = bootstrap_weights()
    weights = pd.read_csv('weights_scaled.csv', index_col=[0, 1])
    fig = plot_results(weights)

    return fig
