"""
Creates Figure 2 -- CMTF Plotting
"""
from itertools import product
import numpy as np
import pandas as pd

from .common import getSetup
from ..dataImport import form_tensor, get_factors
from ..predict import run_model
from tensorpack import calcR2X, perform_CMTF


def get_r2x_results():
    """
    Calculates CMTF R2X with regards to the number of CMTF components and
    RNA/cytokine scaling.

    Parameters:
        None

    Returns:
        r2x_v_components (pandas.Series): R2X vs. number of CMTF components
        r2x_v_scaling (pandas.Series): R2X vs. RNA/cytokine scaling
    """
    # R2X v. Components
    tensor, matrix, _ = form_tensor()

    shuffled_matrix = matrix.copy()
    shuffled_tensor = tensor.copy()
    np.random.shuffle(shuffled_matrix)
    np.random.shuffle(shuffled_tensor)

    components = 12
    r2x_v_components = pd.DataFrame(
        columns=['Baseline', 'Shuffled Tensor', 'Shuffled Matrix'],
        index=np.arange(1, components + 1),
        dtype=float
    )
    for n_components in r2x_v_components.index:
        print(f'Starting decomposition with {n_components} components.')
        t_fac = perform_CMTF(
            tensor,
            matrix,
            r=n_components
        )
        shuffled_t_fac = perform_CMTF(
            shuffled_tensor,
            matrix,
            r=n_components
        )
        shuffled_m_fac = perform_CMTF(
            tensor,
            shuffled_matrix,
            r=n_components
        )

        r2x_v_components.loc[n_components, 'Baseline'] = t_fac.R2X
        r2x_v_components.loc[n_components, 'Shuffled Tensor'] = \
            shuffled_t_fac.R2X
        r2x_v_components.loc[n_components, 'Shuffled Matrix'] = \
            shuffled_m_fac.R2X

    # R2X v. Scaling
    scaling_v = np.logspace(-10, 10, base=2, num=25)
    shuffle_types = ['Baseline', 'Tensor', 'Matrix']
    r2x_types = ['Total', 'Tensor', 'Matrix']
    r2x_v_scaling = pd.DataFrame(
        index=scaling_v,
        columns=pd.MultiIndex.from_product([shuffle_types, r2x_types])
    )
    for scaling in scaling_v:
        tensor, matrix, _ = form_tensor(scaling)
        shuffled_matrix = matrix.copy()
        shuffled_tensor = tensor.copy()
        np.random.shuffle(shuffled_matrix)
        np.random.shuffle(shuffled_tensor)

        t_fac = perform_CMTF(
            tensor,
            matrix,
            r=8
        )
        shuffled_t_fac = perform_CMTF(
            shuffled_tensor,
            matrix,
            r=8
        )
        shuffled_m_fac = perform_CMTF(
            tensor,
            shuffled_matrix,
            r=8
        )

        r2x_v_scaling.loc[scaling, 'Baseline'] = [
            t_fac.R2X,
            calcR2X(t_fac, tIn=tensor),
            calcR2X(t_fac, mIn=matrix)
        ]
        r2x_v_scaling.loc[scaling, 'Tensor'] = [
            shuffled_t_fac.R2X,
            calcR2X(shuffled_t_fac, tIn=shuffled_tensor),
            calcR2X(shuffled_t_fac, mIn=matrix)
        ]
        r2x_v_scaling.loc[scaling, 'Matrix'] = [
            shuffled_m_fac.R2X,
            calcR2X(shuffled_m_fac, tIn=tensor),
            calcR2X(shuffled_m_fac, mIn=shuffled_matrix)
        ]

    return r2x_v_components, r2x_v_scaling


def plot_results(r2x_v_components, r2x_v_scaling):
    """
    Plots prediction model performance.

    Parameters:
        r2x_v_components (pandas.Series): R2X vs. number of CMTF components
        r2x_v_scaling (pandas.DataFrame): R2X vs. RNA/cytokine scaling

    Returns:
        fig (matplotlib.Figure): figure depicting CMTF parameterization plots
    """
    fig_size = (5, 5)
    layout = {
        'ncols': 2,
        'nrows': 2
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )

    # R2X v. Components

    for column in r2x_v_components.columns:
        axs[0].plot(
            r2x_v_components.index,
            r2x_v_components.loc[:, column],
            label=column
        )

    axs[0].legend()
    axs[0].set_ylabel('R2X')
    axs[0].set_xlabel('Number of Components')
    axs[0].set_ylim(0, 1)
    axs[0].set_xticks(r2x_v_components.index)

    # R2X v. Scaling

    r2x_v_scaling.loc[:, 'Baseline'].plot(ax=axs[1])
    axs[1].legend(
        ['Total', 'Cytokine', 'RNA']
    )
    axs[1].set_xscale('log')
    axs[1].set_ylabel('R2X')
    axs[1].set_xlabel('Variance Scaling\n(Cytokine/RNA)')
    axs[1].set_ylim(0, 1)
    axs[1].tick_params(axis='x', pad=-3)
    axs[1].set_title('Baseline')

    r2x_v_scaling.loc[:, 'Tensor'].plot(ax=axs[2])
    axs[2].legend(
        ['Total', 'Cytokine', 'RNA']
    )
    axs[2].set_xscale('log')
    axs[2].set_ylabel('R2X')
    axs[2].set_xlabel('Variance Scaling\n(Cytokine/RNA)')
    axs[2].set_ylim(0, 1)
    axs[2].tick_params(axis='x', pad=-3)
    axs[2].set_title('Shuffled Tensor')

    r2x_v_scaling.loc[:, 'Matrix'].plot(ax=axs[3])
    axs[3].legend(
        ['Total', 'Cytokine', 'RNA']
    )
    axs[3].set_xscale('log')
    axs[3].set_ylabel('R2X')
    axs[3].set_xlabel('Variance Scaling\n(Cytokine/RNA)')
    axs[3].set_ylim(0, 1)
    axs[3].tick_params(axis='x', pad=-3)
    axs[3].set_title('Shuffled Matrix')

    return fig


def makeFigure():
    r2x_v_components, r2x_v_scaling = get_r2x_results()
    fig = plot_results(r2x_v_components, r2x_v_scaling)

    return fig
