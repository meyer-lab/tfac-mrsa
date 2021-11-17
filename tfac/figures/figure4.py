"""
Creates Figure 4 -- Bootstrapped Model Weights
"""
from os.path import abspath, dirname

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.utils import resample

from .figureCommon import getSetup, OPTIMAL_SCALING
from ..dataImport import form_tensor, import_cytokines
from ..predict import run_model
from tensorpack import perform_CMTF

N_BOOTSTRAP = 30
PATH_HERE = dirname(dirname(abspath(__file__)))
TARGETS = ['status', 'gender']


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


def plot_results(weights):
    """
    Plots range of weights for each CMTF component.

    Parameters:
        weights (pandas.DataFrame): mean and StD of component weights w/r to
            prediction targets

    Returns:
        fig (plt.figure): figure containing component weight range plot
    """
    fig_size = (3, 3)
    layout = {
        'ncols': 1,
        'nrows': 1,
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout,
        style=None
    )
    ax = axs[0]

    for offset, target in enumerate(TARGETS):
        ax.errorbar(
            weights.loc[(target, 'Mean')],
            range(
                offset,
                (len(TARGETS) + 10) * weights.shape[1],
                len(TARGETS) + 10
            ),
            marker='.',
            xerr=weights.loc[(target, 'StD')],
            linestyle='',
            capsize=2
        )

    ax.legend(
        ['Persistence', 'Sex', 'Race', 'Age']
    )
    ax.plot(
        [0, 0],
        [-100, 100],
        color='k',
        linestyle='--'
    )

    ax.set_yticks(np.arange(1.5, 100, 12))
    ax.set_yticklabels(
        [f'Cmp. {i}' for i in range(1, 10)]
    )
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 100)

    ax.set_xlabel('Model Coefficient')
    plt.subplots_adjust(left=0.2, right=0.925, top=0.95)

    return fig


def makeFigure():
    weights = bootstrap_weights()
    fig = plot_results(weights)

    return fig
