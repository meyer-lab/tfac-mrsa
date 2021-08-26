import numpy as np
import pandas as pd

from tfac.dataImport import form_tensor
from tfac.figures.figureCommon import getSetup, OPTIMAL_SCALING
from tfac.predict import run_model
from tfac.tensor import perform_CMTF


def makeFigure():
    weights = get_model_coefficients()
    fig = plot_results(weights)

    return fig


def get_model_coefficients():
    """
    Fits Logistic Regression model to CMTF components, then returns the
    coefficient associated with each component.

    Parameters:
        None

    Returns:
        weights (numpy.array): Logistic Regression coefficients associated
            with CMTF components
    """
    tensor, matrix, patient_data = form_tensor(OPTIMAL_SCALING)
    labels = patient_data.loc[:, 'status']
    components = perform_CMTF(tensor, matrix)
    components = components[1][0]

    data = pd.DataFrame(
        components,
        index=patient_data.index,
        columns=list(range(1, components.shape[1] + 1))
    )
    labels = labels.loc[labels != 'Unknown']
    data = data.loc[labels.index, :]

    _, model = run_model(data, labels)
    model.fit(data, labels)

    return model.coef_[0]


def plot_results(weights):
    """
    Plots model coefficients for each component in CMTF factorization.

    Parameters:
        weights (numpy.array): model weights associated with each component

    Returns:
        fig (matplotlib.Figure): bar plot depicting model coefficients for
            each CMTF component
    """
    fig_size = (4, 4)
    layout = (1, 1)
    axs, fig = getSetup(
        fig_size,
        layout
    )

    axs[0].bar(
        range(1, len(weights) + 1),
        weights
    )

    axs[0].set_xlabel('Component', fontsize=12)
    axs[0].set_ylabel('Model Coefficient', fontsize=12)
    axs[0].set_xticks(np.arange(1, len(weights) + 1))
    axs[0].set_xticklabels(np.arange(1, len(weights) + 1))

    return fig
