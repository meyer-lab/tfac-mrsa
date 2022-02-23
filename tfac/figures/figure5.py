"""
Creates Figure 5 -- Reduced Model
"""
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
from os.path import abspath, dirname
import pandas as pd
from sklearn.metrics import roc_curve

from .common import getSetup
from ..dataImport import import_validation_patient_metadata, get_factors
from ..predict import predict_known

COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
PATH_HERE = dirname(dirname(abspath(__file__)))


def run_cv(components, patient_data):
    """
    Predicts samples with known outcomes via cross-validation.

    Parameters:
        components (numpy.array): CMTF components
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        predictions (pandas.Series): predictions for each data source
    """
    labels = patient_data.loc[components.index, 'status'].astype(int)

    predictions = pd.DataFrame(
        index=patient_data.index
    )
    probabilities = predictions.copy()

    predictions.loc[:, 'Full'] = predict_known(components, labels)
    probabilities.loc[:, 'Full'] = predict_known(
        components,
        labels,
        method='predict_proba'
    )

    predictions.loc[:, '5_7'] = predict_known(
        components.loc[:, [5, 7]],
        labels
    )
    probabilities.loc[:, '5_7'] = predict_known(
        components.loc[:, [5, 7]],
        labels,
        method='predict_proba'
    )

    predictions.loc[:, 'Actual'] = patient_data.loc[:, 'status']

    return predictions, probabilities


def get_accuracy(predicted, actual):
    """
    Returns the accuracy for the provided predictions.

    Parameters:
        predicted (pandas.Series): predicted values for samples
        actual (pandas.Series): actual values for samples

    Returns:
        float: accuracy of predicted values
    """
    predicted = predicted.astype(float)
    actual = actual.astype(float)

    correct = [1 if predicted.loc[i] == actual.loc[i] else 0 for i in
               predicted.index]

    return np.mean(correct)


def get_accuracies(samples):
    """
    Calculates prediction accuracy for samples with known outcomes.

    Parameters:
        samples (pandas.DataFrame): predictions for different models

    Returns:
        accuracies (pandas.Series): model accuracy w/r to each model type
    """
    actual = samples.loc[:, 'Actual']
    samples = samples.drop('Actual', axis=1)

    d_types = samples.columns
    accuracies = pd.Series(
        index=d_types,
        dtype=float
    )

    for d_type in d_types:
        col = samples.loc[:, d_type]
        accuracies.loc[d_type] = get_accuracy(col, actual)

    return accuracies


def plot_results(train_samples, train_probabilities, components, patient_data):
    """
    Plots prediction model performance.

    Parameters:
        train_samples (pandas.DataFrame): predictions for training samples
        train_probabilities (pandas.DataFrame): predicted probability of
            persistence for training samples
        components (pandas.DataFrame): CMTF components
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        fig (matplotlib.Figure): figure depicting predictions for all samples
    """
    fig_size = (6, 2)
    layout = {
        'ncols': 3,
        'nrows': 1,
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )

    # Cross-validation Accuracies

    accuracies = get_accuracies(train_samples)
    axs[0].bar(
        [0, 1],
        accuracies,
        color=COLOR_CYCLE[:2],
        width=0.8
    )

    axs[0].set_xlim(-0.5, 1.5)
    axs[0].set_ylim(0, 1)
    axs[0].set_xticks(
        np.arange(0, 2, 1)
    )
    axs[0].set_xticklabels(
        ['All\nComponents', 'Components\n5 & 7'],
        fontsize=9,
        ma='center',
        rotation=0,
        va='top'
    )
    axs[0].set_ylabel('Prediction Accuracy')

    # AUC-ROC Curves

    fpr_full, svc_tpr, _ = roc_curve(
        train_samples.loc[:, 'Actual'],
        train_probabilities.loc[:, 'Full']
    )
    fpr_5_7, tpr_5_7, _ = roc_curve(
        train_samples.loc[:, 'Actual'],
        train_probabilities.loc[:, '5_7']
    )

    axs[1].plot(fpr_5_7, tpr_5_7, color=COLOR_CYCLE[0])
    axs[1].plot(fpr_full, svc_tpr, color=COLOR_CYCLE[1])

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(['All Components', 'Components 5 & 7'])
    axs[1].plot([0, 1], [0, 1], color='k', linestyle='--')

    # 5 v 7 scatter

    color = patient_data.loc[:, 'status'].astype(int)
    color = color.replace(0, COLOR_CYCLE[3])
    color = color.replace(1, COLOR_CYCLE[4])

    style = patient_data.loc[:, 'gender'].astype(int)
    style = style.replace(0, 's')
    style = style.replace(1, 'o')

    for marker in ['s', 'o']:
        index = style.loc[style == marker].index
        axs[2].scatter(
            components.loc[index, 5],
            components.loc[index, 7],
            c=color.loc[index],
            edgecolors='k',
            marker=marker
        )

    axs[2].set_xticks(np.arange(-1, 1.1, 0.5))
    axs[2].set_xlim([-1.1, 1.1])
    axs[2].set_yticks(np.arange(-1, 1.1, 0.5))
    axs[2].set_ylim([-1.1, 1.1])

    axs[2].set_xlabel('Component 5')
    axs[2].set_ylabel('Component 7')

    legend_markers = [
        Line2D([0], [0], lw=4, color=COLOR_CYCLE[3], label='Resolver'),
        Line2D([0], [0], lw=4, color=COLOR_CYCLE[4], label='Persistor'),
        Line2D([0], [0], marker='s', color='k', label='Male'),
        Line2D([0], [0], marker='o', color='k', label='Female')
    ]
    axs[2].legend(handles=legend_markers)

    return fig


def makeFigure():
    t_fac, patient_data = get_factors()
    val_data = import_validation_patient_metadata()
    patient_data.loc[val_data.index, 'status'] = val_data.loc[:, 'status']

    components = t_fac[1][0]
    components = pd.DataFrame(
        components,
        index=patient_data.index,
        columns=list(range(1, components.shape[1] + 1))
    )

    train_samples, train_probabilities = run_cv(components, patient_data)
    train_samples = train_samples.astype(int)

    fig = plot_results(
        train_samples,
        train_probabilities,
        components,
        patient_data
    )

    return fig
