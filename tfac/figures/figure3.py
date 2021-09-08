from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from os.path import abspath, dirname, join
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from tfac.figures.figureCommon import getSetup, OPTIMAL_SCALING
from tfac.dataImport import form_tensor, import_cytokines
from tfac.predict import predict_known, predict_unknown
from tfac.tensor import perform_CMTF

PATH_HERE = dirname(dirname(abspath(__file__)))


def get_data_types():
    """
    Creates data for classification.

    Parameters:
        None

    Returns:
        data_types (list[tuple]): data sources and their names
        patient_data (pandas.DataFrame): patient metadata
    """
    plasma_cyto, serum_cyto = import_cytokines()
    tensor, matrix, patient_data = form_tensor(OPTIMAL_SCALING)
    patient_data = patient_data.loc[:, ['status', 'type']]

    components = perform_CMTF(tensor, matrix)
    components = components[1][0]

    data_types = [
        ('Plasma Cytokines', plasma_cyto.T),
        ('Plasma IL-10', plasma_cyto.loc['IL-10', :]),
        ('Serum Cytokines', serum_cyto.T),
        ('Serum IL-10', serum_cyto.loc['IL-10', :]),
        ('CMTF', pd.DataFrame(
            components,
            index=patient_data.index,
            columns=list(range(1, components.shape[1] + 1))
        )
        )
    ]

    return data_types, patient_data


def run_unknown(data_types, patient_data):
    """
    Predicts samples with unknown outcomes.

    Parameters:
        data_types (list[tuple]): data sources to predict
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        predictions (pandas.Series): predictions for each data source
    """
    predictions = pd.DataFrame(
        index=patient_data.index
    )
    predictions = predictions.loc[patient_data['status'] == 'Unknown']

    for data_type in data_types:
        source = data_type[0]
        data = data_type[1]
        labels = patient_data.loc[data.index, 'status']

        _predictions = predict_unknown(data, labels)
        predictions.loc[_predictions.index, source] = _predictions

    return predictions


def run_cv(data_types, patient_data):
    """
    Predicts samples with known outcomes via cross-validation.

    Parameters:
        data_types (list[tuple]): data sources to predict
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        predictions (pandas.Series): predictions for each data source
    """
    predictions = pd.DataFrame(
        index=patient_data.index
    )
    probabilities = predictions.copy()

    for data_type in data_types:
        source = data_type[0]
        data = data_type[1]
        labels = patient_data.loc[data.index, 'status']

        _predictions = predict_known(data, labels)
        _probabilities = predict_known(data, labels, method='predict_proba')

        probabilities.loc[_probabilities.index, source] = _probabilities
        predictions.loc[_predictions.index, source] = _predictions

    predictions.loc[:, 'Actual'] = patient_data.loc[:, 'status']

    return predictions, probabilities


def get_accuracies(cv_results):
    """
    Calculates prediction accuracy for training samples with known outcomes.

    Parameters:
        cv_results (pandas.DataFrame): predictions for training samples

    Returns:
        accuracies (pandas.Series): model accuracy w/r to each model type
    """
    d_types = cv_results.columns[:-1]
    accuracies = pd.Series(
        index=d_types
    )

    for d_type in d_types:
        col = cv_results.loc[:, d_type].dropna().astype(int)
        actual = cv_results.loc[col.index, 'Actual'].astype(int)
        correct = [1 if col.loc[i] == actual.loc[i] else 0 for i in col.index]
        accuracies.loc[d_type] = np.mean(correct)

    return accuracies


def plot_results(cv_results, cv_probabilities, val_results):
    """
    Plots predictions as heatmaps.

    Parameters:
        cv_results (pandas.DataFrame): predictions for samples with known
            outcomes
        cv_probabilities (pandas.DataFrame): predicted probability of
            persistence for samples with known outcomes
        val_results (pandas.DataFrame): predictions for samples with unknown
            outcomes

    Returns:
        fig (matplotlib.Figure): figure depicting predictions for all samples
    """
    fig_size = (10, 8)
    layout = (3, 2)
    axs, fig = getSetup(
        fig_size,
        layout
    )

    # Cross-validation Accuracies

    cv_results = cv_results.loc[cv_results['Actual'] != 'Unknown']
    accuracies = get_accuracies(cv_results)
    axs[0].bar(
        range(len(accuracies)),
        accuracies
    )

    ticks = [label.replace(' ', '\n') for label in accuracies.index]
    axs[0].set_xticks(range(len(accuracies)))
    axs[0].set_xticklabels(ticks, fontsize=10)
    axs[0].set_ylabel('Mean Accuracy over\n10-fold Cross-Validation', fontsize=12)
    axs[0].text(
        0.06,
        0.96,
        'A',
        fontsize=16,
        fontweight='bold',
        transform=plt.gcf().transFigure
    )

    # AUC-ROC Curves

    for d_type in cv_probabilities.columns:
        col = cv_probabilities.loc[:, d_type].dropna()
        actual = cv_results.loc[col.index, 'Actual'].astype(int)
        fpr, tpr, _ = roc_curve(actual, col)
        axs[1].plot(fpr, tpr)

    axs[1].legend(cv_probabilities.columns)
    axs[1].set_xlabel('False Positive Rate', fontsize=12)
    axs[1].set_ylabel('True Positive Rate', fontsize=12)
    axs[1].text(
        0.49,
        0.96,
        'B',
        fontsize=16,
        fontweight='bold',
        transform=plt.gcf().transFigure
    )

    # Validation set predictions

    axs[2].remove()
    axs[3].remove()
    gs = axs[2].get_gridspec()
    h_map = fig.add_subplot(gs[2:4])

    val_results = val_results.fillna(-1).astype(int)
    sns.heatmap(
        val_results.T,
        ax=h_map,
        cbar=False,
        cmap=['dimgrey', '#ffd2d2', '#9caeff'],
        vmin=-1,
        linewidths=1
    )

    h_map.set_xticks(
        np.arange(0.5, val_results.shape[0], 1)
    )
    h_map.set_xticklabels(
        val_results.index,
        fontsize=10,
        rotation=90
    )
    h_map.set_yticks(
        np.arange(0.5, val_results.shape[1], 1)
    )
    h_map.set_yticklabels(
        val_results.columns,
        fontsize=10,
        rotation=0
    )
    h_map.set_xlabel('Patient', fontsize=12)
    legend_elements = [Patch(facecolor='dimgrey', edgecolor='dimgrey', label='Data Not Available'),
                       Patch(facecolor='#ffd2d2', edgecolor='#ffd2d2', label='Persistor'),
                       Patch(facecolor='#9caeff', edgecolor='#9caeff', label='Resolver')]
    h_map.legend(handles=legend_elements, loc=[1.02, 0.4])
    h_map.text(
        0.06,
        0.62,
        'C',
        fontsize=16,
        fontweight='bold',
        transform=plt.gcf().transFigure
    )

    return fig


def export_results(train_samples, train_probabilities, validation_samples):
    """
    Reformats prediction DataFrames and saves as .txt.

    Parameters:
        train_samples (pandas.DataFrame): predictions for training samples
        train_probabilities (pandas.DataFrame): probabilities of persistence for
            training samples
        validation_samples (pandas.DataFrame): predictions for validation samples

    Returns:
        None
    """
    train_samples = train_samples.astype(str)
    train_probabilities = train_probabilities.astype(str)
    validation_samples = validation_samples.astype(str)

    validation_samples = validation_samples.replace('0', 'ARMB')
    validation_samples = validation_samples.replace('1', 'APMB')
    train_samples = train_samples.replace('0', 'ARMB')
    train_samples = train_samples.replace('1', 'APMB')

    validation_samples.to_csv(
        join(
            PATH_HERE,
            '..',
            'output',
            'validation_predictions.txt'
        )
    )
    train_samples.to_csv(
        join(
            PATH_HERE,
            '..',
            'output',
            'train_predictions.txt'
        )
    )
    train_probabilities.to_csv(
        join(
            PATH_HERE,
            '..',
            'output',
            'train_probabilities.txt'
        )
    )


def makeFigure():
    data_types, patient_data = get_data_types()
    train_samples, train_probabilities = run_cv(data_types, patient_data)
    validation_samples = run_unknown(data_types, patient_data)

    export_results(train_samples, train_probabilities, validation_samples)
    fig = plot_results(train_samples, train_probabilities, validation_samples)

    return fig
