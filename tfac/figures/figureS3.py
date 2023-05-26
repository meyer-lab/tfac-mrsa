from os.path import abspath, dirname, join

import pandas as pd
import seaborn as sns

from .common import getSetup
from ..dataImport import import_cibersort_results

PATH_HERE = dirname(dirname(dirname(abspath(__file__))))


def makeFigure():
    cs_results = import_cibersort_results()
    fig = plot_results(cs_results)

    return fig


def plot_results(cs_results):
    """
    Plots cibersort immune cell mixtures.

    Parameters:
        cs_results (pandas.DataFrame): component associations to LM22 cells

    Returns:
        fig (matplotlib.Figure): heatmap of components associations to immune
            cell types
    """
    fig_size = (2, 6)
    layout = {
        'ncols': 1,
        'nrows': 1
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )
    ax = axs[0]

    scaled = cs_results / cs_results.max(axis=0)
    sns.heatmap(
        scaled,
        ax=ax,
        vmin=0,
        vmax=1,
        annot=cs_results.round(2),
        cmap="PRGn",
        cbar=False
    )

    ax.set_xticklabels(
        [f'Cmp. {i}' for i in cs_results.columns],
        rotation=45,
        ha='right',
        ma='center',
        va='top'
    )
    ax.set_xlabel('')
    ax.set_yticklabels(
        cs_results.index,
        ha='right',
        ma='right',
        va='center',
        rotation=0
    )
    ax.set_ylabel('Immune Cell')

    return fig
