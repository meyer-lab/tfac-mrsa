"""
Runs hyperparameter optimization for a Logistic Regression model that
uses CMTF components to classify MRSA persistance. Generates a figure
depicting model accuracy against scaling and component count.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figureCommon import getSetup, OPTIMAL_SCALING
from ..dataImport import form_tensor
from ..tensor import calcR2X, perform_CMTF
from ..predict import evaluate_scaling, evaluate_components


def plot_results(by_scaling, by_components):
    """
    Plots accuracy of model with regards to variance scaling and CMTF
    component parameters.

    Parameters:
        by_scaling (pandas.Series): Model accuracy with regards to
            variance scaling
        by_components (pandas.Series): Model accuracy with regards to
            number of CMTF components

    Returns:
        fig (matplotlib.pyplot.Figure): Figure containing plots of
            scaling and CMTF component analyses
    """
    # Sets up plotting space
    fig_size = (8, 4)
    layout = (1, 3)
    axs, fig = getSetup(
        fig_size,
        layout
    )

    # Model Performance v. Scaling

    axs[0].semilogx(by_scaling.index, by_scaling, base=2)
    axs[0].set_ylabel('Best Accuracy over Repeated\n10-fold Cross Validation', fontsize=12)
    axs[0].set_xlabel('Variance Scaling (RNA/Cytokine)', fontsize=12)
    axs[0].set_ylim([0.5, 0.8])
    axs[0].set_xticks(np.logspace(-7, 7, base=2, num=8))
    axs[0].text(0.02, 0.9, 'A', fontsize=16, fontweight='bold', transform=plt.gcf().transFigure)

    # Best Scaling v. CMTF Components

    axs[1].plot(by_components.index, by_components)
    axs[1].set_ylabel('Best Accuracy over Repeated\n10-fold Cross Validation', fontsize=12)
    axs[1].set_xlabel('CMTF Components', fontsize=12)
    axs[1].set_xticks(by_components.index)
    axs[1].set_ylim([0.5, 0.8])
    axs[1].text(0.52, 0.9, 'B', fontsize=16, fontweight='bold', transform=plt.gcf().transFigure)

    # R2X vs. Scaling

    R2X = pd.DataFrame(
        index=np.logspace(-7, 7, base=2, num=29).tolist(), columns=["Total", "Tensor", "Matrix"], data=np.zeros((29, 3)),
        dtype=float
    )

    for scaling in R2X.index:
        tensor, matrix, _ = form_tensor(scaling)
        tFac = perform_CMTF(tOrig=tensor, mOrig=matrix)
        R2X.loc[scaling, "Total"] = calcR2X(tFac, tensor, matrix)
        R2X.loc[scaling, "Tensor"] = calcR2X(tFac, tIn=tensor)
        R2X.loc[scaling, "Matrix"] = calcR2X(tFac, mIn=matrix)

    R2X.plot(ax=axs[2])
    axs[2].set_xscale("log")
    axs[2].set_ylabel("R2X")
    axs[2].set_xlabel("Tensor scaled")
    axs[2].set_ylim(0, 1.0)

    return fig


def makeFigure():
    """
    Generates Figure 4.

    Parameters:
        None

    Returns:
        fig (matplotlib.pyplot.Figure): Figure containing plots of
            scaling and CMTF component analyses
    """
    by_scaling = evaluate_scaling()
    by_components = evaluate_components(OPTIMAL_SCALING)
    return plot_results(by_scaling, by_components)
