"""
Creates Figure 7 -- Imputation Effectiveness
"""
import matplotlib
from matplotlib.patches import Patch
import numpy as np
from os.path import abspath, dirname
import pandas as pd
from scipy.stats import pearsonr
from tensorpack import perform_CMTF
from tensorly import cp_to_tensor

from .common import getSetup
from ..dataImport import form_tensor, import_cytokines

COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][2:]
PATH_HERE = dirname(dirname(abspath(__file__)))


def impute_cv():
    """
    Imputes cytokines for each subject via cross-validation.

    Parameters:
        None

    Returns:
        tensor (numpy.array): cytokine measurements
        imputed (numpy.array): cytokine measurements predicted via CMTF
    """
    tensor, matrix, patient_data = form_tensor()
    both_cyto = patient_data.loc[:, 'type'].str.contains('0Serum1Plasma')

    imputed = np.full(tensor.shape, np.nan)
    both_idx = np.where(both_cyto)[0]
    print(both_idx)

    for k in range(tensor.shape[2]):
        for i in both_idx:
            dropped = tensor.copy()
            dropped[i, :, k] = np.nan
            t_fac = perform_CMTF(
                tensor,
                matrix,
                r=8,
                maxiter=400,
                progress=True
            )

            _imputed = cp_to_tensor(t_fac)
            imputed[i, :, k] = _imputed[i, :, k]
    
    # Subset for subjects that have both samples
    tensor = tensor[both_cyto, :, :]
    imputed = imputed[both_cyto, :, :]

    return tensor, imputed


def get_correlations(tensor, imputed):
    """
    Gets correlations between imputed and measured values.

    Parameters:
        tensor (numpy.array): cytokine measurements
        imputed (numpy.array): cytokine measurements predicted via CMTF

    Returns:
        correlations (pandas.DataFrame): correlations between imputed and
            measured cytokines
    """
    serum, _ = import_cytokines()
    correlations = pd.DataFrame(
        index=serum.index,
        columns=['Imputed Serum', 'Imputed Plasma', 'Measured']
    )

    for cyto in range(tensor.shape[1]):
        serum_imputed, _ = pearsonr(
            imputed[:, cyto, 0],
            tensor[:, cyto, 0]
        )
        plasma_imputed, _ = pearsonr(
            imputed[:, cyto, 1],
            tensor[:, cyto, 1]
        )
        measured, _ = pearsonr(
            tensor[:, cyto, 0],
            tensor[:, cyto, 1]
        )

        correlations.iloc[cyto, :] = [serum_imputed, plasma_imputed, measured]

    return correlations


def plot_results(tensor, imputed, correlations):
    """
    Plots prediction model performance.

    Parameters:
        tensor (numpy.array): cytokine measurements
        imputed (numpy.array): cytokine measurements predicted via CMTF
        correlations (pandas.DataFrame): correlations between imputed and
            measured cytokines

    Returns:
        fig (matplotlib.Figure): figure depicting imputed and actual cytokine
            measurements
    """
    fig_size = (8, 6)
    layout = {
        'ncols': 1,
        'nrows': 3,
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )
    serum, plasma = import_cytokines()

    imputed_top = imputed[:, :18, :]
    imputed_bottom = imputed[:, 18:, :]
    tensor_top = tensor[:, :18, :]
    tensor_bottom = tensor[:, 18:, :]
    cyto_top = serum.index[:18]
    cyto_bottom = serum.index[18:]

    axs[0].boxplot(
        imputed_top[:, :, 0],
        positions=np.arange(0, 5 * imputed_top.shape[1], 5),
        sym='d',
        patch_artist=True,
        boxprops={
            'facecolor': COLOR_CYCLE[0],
        },
        flierprops={
            'markersize': 3,
            'markerfacecolor': COLOR_CYCLE[0]
        }
    )
    axs[0].boxplot(
        tensor_top[:, :, 0],
        positions=np.arange(1, 5 * tensor_top.shape[1], 5),
        sym='d',
        patch_artist=True,
        boxprops={
            'facecolor': COLOR_CYCLE[1],
        },
        flierprops={
            'markersize': 3,
            'markerfacecolor': COLOR_CYCLE[1]
        }
    )
    axs[0].boxplot(
        imputed_top[:, :, 1],
        positions=np.arange(2, 5 * imputed_top.shape[1], 5),
        sym='d',
        patch_artist=True,
        boxprops={
            'facecolor': COLOR_CYCLE[2],
        },
        flierprops={
            'markersize': 3,
            'markerfacecolor': COLOR_CYCLE[2]
        }
    )
    axs[0].boxplot(
        tensor_top[:, :, 1],
        positions=np.arange(3, 5 * tensor_top.shape[1], 5),
        sym='d',
        patch_artist=True,
        boxprops={
            'facecolor': COLOR_CYCLE[3],
        },
        flierprops={
            'markersize': 3,
            'markerfacecolor': COLOR_CYCLE[3]
        }
    )

    axs[1].boxplot(
        imputed_bottom[:, :, 0],
        positions=np.arange(0, 5 * imputed_bottom.shape[1], 5),
        sym='d',
        patch_artist=True,
        boxprops={
            'facecolor': COLOR_CYCLE[0],
        },
        flierprops={
            'markersize': 3,
            'markerfacecolor': COLOR_CYCLE[0]
        }
    )
    axs[1].boxplot(
        tensor_bottom[:, :, 0],
        positions=np.arange(1, 5 * tensor_bottom.shape[1], 5),
        sym='d',
        patch_artist=True,
        boxprops={
            'facecolor': COLOR_CYCLE[1],
        },
        flierprops={
            'markersize': 3,
            'markerfacecolor': COLOR_CYCLE[1]
        }
    )
    axs[1].boxplot(
        imputed_bottom[:, :, 1],
        positions=np.arange(2, 5 * imputed_bottom.shape[1], 5),
        sym='d',
        patch_artist=True,
        boxprops={
            'facecolor': COLOR_CYCLE[2],
        },
        flierprops={
            'markersize': 3,
            'markerfacecolor': COLOR_CYCLE[2]
        }
    )
    axs[1].boxplot(
        tensor_bottom[:, :, 1],
        positions=np.arange(3, 5 * tensor_bottom.shape[1], 5),
        sym='d',
        patch_artist=True,
        boxprops={
            'facecolor': COLOR_CYCLE[3],
        },
        flierprops={
            'markersize': 3,
            'markerfacecolor': COLOR_CYCLE[3]
        }
    )

    axs[0].set_xticks(
        np.arange(1.5, 5 * tensor_top.shape[1], 5)
    )
    axs[0].set_xticklabels(
        cyto_top,
        rotation=45,
        ha='right',
        va='top'
    )
    axs[1].set_xticks(
        np.arange(1.5, 5 * tensor_bottom.shape[1], 5)
    )
    axs[1].set_xticklabels(
        cyto_bottom,
        rotation=45,
        ha='right',
        va='top'
    )

    axs[0].set_xlim([-2, 5 * tensor_top.shape[1]])
    axs[0].set_xlabel('Cytokine')
    axs[0].set_ylabel('Cytokine Expression')

    axs[1].set_xlim([-2, 5 * tensor_bottom.shape[1]])
    axs[1].set_xlabel('Cytokine')
    axs[1].set_ylabel('Cytokine Expression')

    legend_patches = [
        Patch(color=COLOR_CYCLE[0]),
        Patch(color=COLOR_CYCLE[1]),
        Patch(color=COLOR_CYCLE[2]),
        Patch(color=COLOR_CYCLE[3])
    ]
    axs[0].legend(
        legend_patches,
        ['Imputed Serum', 'Measured Serum', 'Imputed Plasma', 'Measured Plasma']
    )

    axs[2].bar(
        np.arange(0, 4 * correlations.shape[0], 4),
        correlations.loc[:, 'Imputed Serum'],
        width=1,
        color=COLOR_CYCLE[0]
    )
    axs[2].bar(
        np.arange(1, 4 * correlations.shape[0], 4),
        correlations.loc[:, 'Imputed Plasma'],
        width=1,
        color=COLOR_CYCLE[1]
    )
    axs[2].bar(
        np.arange(2, 4 * correlations.shape[0], 4),
        correlations.loc[:, 'Measured'],
        width=1,
        color=COLOR_CYCLE[2]
    )

    axs[2].set_xticks(
        np.arange(1, 4 * correlations.shape[0], 4)
    )
    axs[2].set_xticklabels(
        correlations.index,
        rotation=45,
        ha='right',
        va='top'
    )

    axs[2].set_xlim([-2, 4 * correlations.shape[0]])
    axs[2].set_xlabel('Cytokine')
    axs[2].set_ylabel('Pearson Correlation')

    legend_patches = [
        Patch(color=COLOR_CYCLE[0]),
        Patch(color=COLOR_CYCLE[1]),
        Patch(color=COLOR_CYCLE[2])
    ]
    axs[2].legend(
        legend_patches,
        ['Imputed Serum', 'Imputed Plasma', 'Measured Serum v. Plasma']
    )

    return fig


def makeFigure():
    tensor, imputed = impute_cv()
    correlations = get_correlations(tensor, imputed)
    fig = plot_results(tensor, imputed, correlations)

    return fig
