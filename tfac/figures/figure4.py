"""
Creates Figure 4 -- Validation Model and Predictions
"""
from os.path import abspath, dirname

from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from .figureCommon import getSetup, OPTIMAL_SCALING
from ..dataImport import import_validation_patient_metadata, form_tensor
from tensorpac import perform_CMTF

PATH_HERE = dirname(dirname(abspath(__file__)))


def bootstrap_weights():
    """
    Predicts samples with unknown outcomes.

    Parameters:
        None

    Returns:
        weights (pandas.Series): bootstrapped coefficient weights
    """
    tensor, matrix, patient_data = form_tensor(OPTIMAL_SCALING)
    components = perform_CMTF(tensor, matrix)
    components = components[1][0]
    data = pd.DataFrame(
        components,
        index=patient_data.index,
        columns=list(range(1, components.shape[1] + 1))
    )

    weights = None
    predictions = pd.DataFrame(
        index=patient_data.index
    )
    predictions = predictions.loc[patient_data['status'] == 'Unknown']

    labels = patient_data.loc[data.index, 'status']

    #     _predictions, coef = predict_validation(
    #         data, labels, return_coef=True
    #     )
    #     predictions.loc[_predictions.index, source] = _predictions
    #     weights = coef
    # else:
    #     _predictions = predict_validation(data, labels)
    #     predictions.loc[_predictions.index, source] = _predictions

    validation_meta = import_validation_patient_metadata()
    predictions.loc[:, 'Actual'] = validation_meta.loc[:, 'status']

    return predictions, weights


def plot_results(validation_predictions, weights):
    """
    Plots validation predictions and model coefficients for each component in 
    CMTF factorization.

    Parameters:
        validation_predictions (pandas.DataFrame): predictions for validation
            samples
        weights (numpy.array): model weights associated with each component

    Returns:
        fig (matplotlib.Figure): bar plot depicting model coefficients for
            each CMTF component
    """
    fig_size = (8, 4)
    layout = (2, 1)
    axs, fig = getSetup(
        fig_size,
        layout
    )

    # Validation Predictions
    
    validation_predictions = validation_predictions.fillna(-1).astype(int)
    sns.heatmap(
        validation_predictions.T,
        ax=axs[0],
        cbar=False,
        cmap=['dimgrey', '#ffd2d2', '#9caeff'],
        vmin=-1,
        linewidths=1
    )

    axs[0].set_xticks(
        np.arange(0.6, validation_predictions.shape[0], 1)
    )
    axs[0].set_xticklabels(
        validation_predictions.index,
        fontsize=10,
        ha='center',
        rotation=90
    )
    axs[0].set_yticklabels(
        validation_predictions.columns,
        fontsize=10,
        rotation=0
    )
    axs[0].set_xlabel('Patient', fontsize=12)

    legend_elements = [Patch(facecolor='dimgrey', edgecolor='dimgrey', label='Data Not Available'),
                       Patch(facecolor='#ffd2d2', edgecolor='#ffd2d2', label='Persistor'),
                       Patch(facecolor='#9caeff', edgecolor='#9caeff', label='Resolver')]
    axs[0].legend(handles=legend_elements, loc=[1.02, 0.4])

    # Feature Weight Plotting

    axs[1].bar(
        range(1, len(weights) + 1),
        weights
    )

    axs[1].set_xlabel('Component', fontsize=12)
    axs[1].set_ylabel('Model Coefficient', fontsize=12)
    axs[1].set_yticks([0, -2, -4])
    axs[1].set_yticklabels([0, -2, -4], fontsize=10)
    axs[1].set_xticks(np.arange(1, len(weights) + 1))
    axs[1].set_xticklabels(np.arange(1, len(weights) + 1), fontsize=10)

    return fig


def makeFigure():
    weights = bootstrap_weights()
    fig = plot_results(weights)

    return fig
