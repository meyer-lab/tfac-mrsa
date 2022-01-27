"""
This creates Figure S4 - Cytokine Correlation Plots
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns

from .common import getSetup
from ..dataImport import import_cytokines


def correlate_cytokines():
    """Creates correlation matrices for both cytokine sources."""
    plasma, serum = import_cytokines()
    corr_plasma = pd.DataFrame(
        index=plasma.index[1:],
        columns=plasma.index[:-1],
        dtype=float
    )
    corr_serum = corr_plasma.copy()

    for source, corr_df in zip([plasma, serum], [corr_plasma, corr_serum]):
        for i in range(source.shape[0]):
            for j in range(i + 1, source.shape[0]):
                col = source.index[i]
                row = source.index[j]

                corr, _ = spearmanr(
                    source.loc[row, :],
                    source.loc[col, :]
                )

                corr_df.loc[row, col] = corr

    return corr_plasma, corr_serum


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    corr_plasma, corr_serum = correlate_cytokines()

    fig_size = (10, 5)
    layout = {
        'ncols': 2,
        'nrows': 1
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )

    sns.heatmap(
        corr_plasma,
        ax=axs[0],
        cbar=False,
        center=0,
        vmin=-1,
        vmax=1
    )
    sns.heatmap(
        corr_serum,
        ax=axs[1],
        center=0,
        vmin=-1,
        vmax=1
    )

    for ax, corr in zip(axs, [corr_plasma, corr_serum]):
        ax.set_xticks(
            np.arange(0.5, corr.shape[0], 1)
        )
        ax.set_xticklabels(
            corr.index,
            ha='center',
            va='top'
        )
        ax.set_yticks(
            np.arange(0.5, corr.shape[0], 1)
        )
        ax.set_yticklabels(
            corr.index,
            ha='right',
            va='center'
        )

    axs[0].text(
        corr_plasma.shape[0] * 0.6,
        corr_plasma.shape[1] * 0.4,
        'Plasma',
        fontsize=16
    )
    axs[1].text(
        corr_serum.shape[0] * 0.6,
        corr_serum.shape[1] * 0.4,
        'Serum',
        fontsize=16
    )

    return fig
