"""
Creates Figure 5 -- Components vs. Cytokines
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


def plot_results(subjects, cytos, source):
    """
    Plots component weights and interpretation.

    Parameters:
        subjects (pandas.DataFrame): patient correlations to tfac components
        cytos (pandas.DataFrame): cytokine correlations to tfac components
        source (pandas.DataFrame): cytokine source correlations to tfac
            components
    """
    fig_size = (5, 5)
    layout = {
        'ncols': 2,
        'nrows': 1,
        'wspace': 0
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout,
        style=None
    )

    v_min = min(subjects.values.min(), cytos.values.min(), source.values.min())
    v_max = max(subjects.values.max(), cytos.values.max(), source.values.max())

    sns.heatmap(
        cytos,
        cmap="PRGn",
        center=0,
        yticklabels=True,
        cbar=False,
        vmin=v_min,
        vmax=v_max,
        ax=axs[0]
    )
    sns.heatmap(
        source,
        cmap="PRGn",
        center=0,
        yticklabels=True,
        cbar=False,
        vmin=v_min,
        vmax=v_max,
        ax=axs[1]
    )

    axs[0].set_yticklabels(cytos.index, fontsize=7)
    axs[1].set_yticklabels(["Serum", "Plasma"], rotation=0)
    axs[1].set_xticks(np.arange(0.5, source.shape[1]))
    axs[1].set_xticklabels([f'Cmp. {i}' for i in range(1, source.shape[1] + 1)])

    return fig


def makeFigure():
    subjects, cytos, source, _ = tfac_setup()
    fig = plot_results(subjects, cytos, source)

    return fig
