"""
Creates Figure 5 -- Reduced Model
"""
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from os.path import abspath, dirname
import pandas as pd
from sklearn.metrics import roc_curve

from .common import getSetup
from ..dataImport import import_validation_patient_metadata, get_factors
from ..predict import get_accuracy, predict_known

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

    predictions.loc[:, 'Full'], _ = predict_known(components, labels)
    probabilities.loc[:, 'Full'], _ = predict_known(
        components,
        labels,
        method='predict_proba'
    )

    best_reduced = (0, (None, None), None)
    persistence_components = [4, 5, 6, 7]
    for i in np.arange(len(persistence_components)):
        for j in np.arange(i + 1, len(persistence_components)):
            comp_1 = persistence_components[i]
            comp_2 = persistence_components[j]

            predictions[(comp_1, comp_2)], _ = predict_known(
                components.loc[:, [comp_1, comp_2]],
                labels
            )
            probabilities[(comp_1, comp_2)], model = \
                predict_known(
                    components.loc[:, [comp_1, comp_2]],
                    labels,
                    method='predict_proba'
                )

            reduced_accuracy = get_accuracy(
                predictions[(comp_1, comp_2)],
                patient_data.loc[:, 'status']
            )
            if reduced_accuracy > best_reduced[0]:
                best_reduced = (reduced_accuracy, (comp_1, comp_2), model)

    predictions.loc[:, 'Actual'] = patient_data.loc[:, 'status']

    return predictions, probabilities, best_reduced


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
        col = samples[d_type]
        accuracies[d_type] = get_accuracy(col, actual)

    return accuracies


def plot_results(train_samples, train_probabilities, model, components,
                 patient_data):
    """
    Plots prediction model performance.

    Parameters:
        train_samples (pandas.DataFrame): predictions for training samples
        train_probabilities (pandas.DataFrame): predicted probability of
            persistence for training samples
        model (tuple): Best LR model using 2 components
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
        np.arange(len(accuracies)),
        accuracies,
        color=COLOR_CYCLE[:4],
        width=0.8
    )

    axs[0].set_xlim(-1, len(accuracies))
    axs[0].set_ylim(0, 1)
    axs[0].set_xticks(
        np.arange(len(accuracies))
    )

    labels = ['All']
    for comps in accuracies.index[1:]:
        labels.append(f'{comps[0]} & {comps[1]}')

    axs[0].set_xticklabels(
        labels,
        rotation=45,
        ha='right',
        va='top'
    )

    axs[0].set_xlabel('Components')
    axs[0].set_ylabel('Prediction Accuracy')

    # AUC-ROC Curves

    for i, reduced in enumerate(train_probabilities.columns):
        fpr, tpr, _ = roc_curve(
            train_samples.loc[:, 'Actual'],
            train_probabilities[reduced]
        )

        axs[1].plot(fpr, tpr, color=COLOR_CYCLE[i])

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(labels)
    axs[1].plot([0, 1], [0, 1], color='k', linestyle='--')

    # Best Model Scatter

    color = patient_data.loc[:, 'status'].astype(int)
    color = color.replace(0, COLOR_CYCLE[3])
    color = color.replace(1, COLOR_CYCLE[4])

    style = patient_data.loc[:, 'gender'].astype(int)
    style = style.replace(0, 's')
    style = style.replace(1, 'o')

    xx, yy = np.meshgrid(
        np.linspace(-1.1, 1.1, 10),
        np.linspace(-1.1, 1.1, 10)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    prob_map = model[2].predict_proba(grid)[:, 1].reshape(xx.shape)
    axs[2].contour(
        xx,
        yy,
        prob_map,
        levels=[0.25, 0.5, 0.75],
        colors=[COLOR_CYCLE[3], 'grey', COLOR_CYCLE[4]],
        linestyles='--'
    )

    for marker in ['s', 'o']:
        index = style.loc[style == marker].index
        axs[2].scatter(
            components.loc[index, model[1][0]],
            components.loc[index, model[1][1]],
            c=color.loc[index],
            s=10,
            edgecolors='k',
            marker=marker,
            zorder=2
        )

    axs[2].set_xticks(np.arange(-1, 1.1, 0.5))
    axs[2].set_xlim([-1.1, 1.1])
    axs[2].set_yticks(np.arange(-1, 1.1, 0.5))
    axs[2].set_ylim([-1.1, 1.1])

    axs[2].set_xlabel(f'Component {model[1][0]}')
    axs[2].set_ylabel(f'Component {model[1][1]}')

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
        columns=list(np.arange(1, components.shape[1] + 1))
    )

    train_samples, train_probabilities, model = \
        run_cv(components, patient_data)
    train_samples = train_samples.astype(int)

    fig = plot_results(
        train_samples,
        train_probabilities,
        model,
        components,
        patient_data
    )

    return fig
