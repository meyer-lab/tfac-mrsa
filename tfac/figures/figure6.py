"""
Creates Figure 6 -- Components vs. Cytokines
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


def cytokine_setup():
    """
    Import cytokine data and CMTF component correlations to cytokine data.

    Parameters:
        None

    Returns:
        cytokines (pandas.DataFrame): cytokine correlations to tfac components
        source (pandas.DataFrame): cytokine source correlations to tfac
            components
    """
    tensor, matrix, _ = form_tensor(OPTIMAL_SCALING)
    plasma, _ = import_cytokines()

    factors = perform_CMTF(tensor, matrix)
    col_names = [f"Cmp. {i}" for i in np.arange(1, factors.rank + 1)]
    cytokines = pd.DataFrame(
        factors.factors[1],
        columns=col_names,
        index=plasma.index
    )
    source = pd.DataFrame(
        factors.factors[2],
        columns=col_names,
        index=["Serum", "Plasma"]
    )

    return cytokines, source


def plot_results(cytokines, source):
    """
    Plots component associations to cytokines and cytokine sources.

    Parameters:
        cytokines (pandas.DataFrame): cytokine correlations to tfac components
        source (pandas.DataFrame): cytokine source correlations to tfac
            components

    Returns:
        fig (plt.figure): figure containing heatmaps of CMTF components vs.
            cytokines and cytokine sources
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

    v_min = min(cytokines.values.min(), source.values.min())
    v_max = max(cytokines.values.max(), source.values.max())

    sns.heatmap(
        cytokines,
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

    axs[0].set_yticklabels(cytokines.index, fontsize=7)
    axs[1].set_yticklabels(["Serum", "Plasma"], rotation=0)
    axs[1].set_xticks(np.arange(0.5, source.shape[1]))
    axs[1].set_xticklabels([f'Cmp. {i}' for i in range(1, source.shape[1] + 1)])

    return fig


def makeFigure():
    cytokines, source = cytokine_setup()
    fig = plot_results(cytokines, source)

    return fig
