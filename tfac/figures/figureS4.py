"""
This creates Figure S4 - Cytokine Correlation Plots
"""
import scipy.cluster.hierarchy as sch
import numpy as np
import seaborn as sns

from .common import getSetup
from ..dataImport import import_cytokines


def correlate_cytokines():
    """Creates correlation matrices for both cytokine sources."""
    plasma, serum = import_cytokines(scale_cyto=False)
    corr_plasma = plasma.T.corr()
    corr_serum = serum.T.corr()

    return corr_plasma, corr_serum


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    corr_plasma, corr_serum = correlate_cytokines()

    pairwise_distances = sch.distance.pdist(corr_plasma)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    corr_plasma = corr_plasma.iloc[idx, :].T.iloc[idx, :]
    corr_serum = corr_serum.iloc[idx, :].T.iloc[idx, :]

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
            corr.columns,
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
