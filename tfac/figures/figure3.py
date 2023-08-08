"""
Creates Figure 3 -- Model Performance
"""
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
from os.path import abspath, dirname, join
import pandas as pd
from sklearn.metrics import r2_score, roc_curve

from .common import getSetup
from ..dataImport import import_validation_patient_metadata, get_factors, import_cytokines, import_rna
from ..predict import get_accuracy, predict_known, predict_validation, predict_regression

COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][2:]
PATH_HERE = dirname(dirname(abspath(__file__)))


def run_validation(data_types, patient_data):
    """
    Predicts validation samples.

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
    probabilities = predictions.copy()

    validation_meta = import_validation_patient_metadata()

    for data_type in data_types:
        source = data_type[0]
        data = data_type[1]

        data = data.reindex(index=patient_data.index)
        data = data.dropna(axis=0)
        labels = patient_data.loc[data.index, 'status']

        _predictions, _probabilities = predict_validation(data, labels)
        predictions.loc[_predictions.index, source] = _predictions
        probabilities.loc[_probabilities.index, source] = _probabilities

    predictions.loc[:, 'Actual'] = validation_meta.loc[:, 'status']
    return predictions, probabilities


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
    sex_predictions = pd.DataFrame(
        index=patient_data.index
    )
    race_predictions = pd.DataFrame(
        index=patient_data.index
    )
    probabilities = predictions.copy()

    prediction_types = [
        predictions,
        sex_predictions,
        race_predictions
    ]
    columns = [
        'status',
        'gender',
        'race'
    ]
    for column, df in zip(columns, prediction_types):
        for data_type in data_types:
            source = data_type[0]
            data = data_type[1]

            data = data.reindex(index=patient_data.index)
            data = data.dropna(axis=0)
            labels = patient_data.loc[data.index, column]

            _predictions, _ = predict_known(data, labels)
            if column == 'status':
                _probabilities, _ = predict_known(
                    data,
                    labels,
                    method='predict_proba'
                )
                probabilities.loc[_probabilities.index, source] = _probabilities

            df.loc[_predictions.index, source] = _predictions

        df.loc[:, 'Actual'] = patient_data.loc[:, column]

    return predictions, probabilities, sex_predictions, race_predictions


def run_age_regression(data, patient_data):
    """
    Predicts patient age from provided data.

    Parameters:
        data (pandas.DataFrame): data to predict age
        patient_data (pandas.DataFrame): patient metadata, including age
    """
    labels = patient_data.loc[:, 'age']
    age_predictions, _ = predict_regression(data, labels)
    age_predictions.name = 'CMTF'

    age_predictions = pd.DataFrame(age_predictions)
    age_predictions.loc[:, 'Actual'] = labels.loc[age_predictions.index]

    return age_predictions


def get_accuracies(samples):
    """
    Calculates prediction accuracy for samples with known outcomes.

    Parameters:
        samples (pandas.DataFrame): predictions for different models

    Returns:
        accuracies (pandas.Series): model accuracy w/r to each model type
    """
    cmtf = samples.loc[:, 'CMTF']
    actual = samples.loc[:, 'Actual']
    samples = samples.drop(['CMTF', 'Actual'], axis=1)

    d_types = samples.columns
    accuracies = pd.Series(
        index=d_types,
        dtype=float
    )
    cmtf_accuracies = pd.Series(
        index=d_types,
        dtype=float
    )

    for d_type in d_types:
        col = samples.loc[:, d_type]
        col = col.dropna()
        labels = actual.loc[col.index]
        cmtf_col = cmtf.loc[col.index]

        accuracies.loc[d_type] = get_accuracy(col, actual)
        cmtf_accuracies.loc[d_type] = get_accuracy(cmtf_col, labels)

    return accuracies, cmtf_accuracies


def plot_results(train_samples, train_probabilities, validation_samples,
                 validation_probabilities, sex_predictions,
                 race_predictions, age_predictions):
    """
    Plots prediction model performance.

    Parameters:
        train_samples (pandas.DataFrame): predictions for training samples
        train_probabilities (pandas.DataFrame): predicted probability of
            persistence for training samples
        validation_samples (pandas.DataFrame): predictions for validation
            cohort samples
        validation_probabilities (pandas.DataFrame): predicted probability
            of persistence for validation samples
        sex_predictions (pandas.DataFrame): sex predictions for samples with
            known outcomes and sex
        race_predictions (pandas.DataFrame): race predictions for samples
            with known outcomes and predictions
        age_predictions (pandas.DataFrame): age predictions for samples with
            known outcomes and age

    Returns:
        fig (matplotlib.Figure): figure depicting predictions for all samples
    """
    fig_size = (5, 5)
    layout = {
        'ncols': 2,
        'nrows': 3,
        'height_ratios': [1, 1, 0.1]
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )

    # Cross-validation Accuracies

    train_samples = train_samples.loc[train_samples['Actual'] != 'Unknown']
    accuracies, train_cmtf = get_accuracies(train_samples)
    axs[0].bar(
        np.arange(0, 3 * len(accuracies), 3),
        accuracies,
        width=1
    )
    axs[0].bar(
        np.arange(1, 3 * len(train_cmtf), 3),
        train_cmtf,
        width=1
    )

    axs[0].set_xlim(-1, 3 * len(accuracies) - 1)
    axs[0].set_ylim(0, 1)
    axs[0].legend(['Raw Data', 'CMTF'])
    ticks = [label.replace(' ', '\n') for label in accuracies.index]
    axs[0].set_xticks(
        np.arange(0.5, 3 * len(accuracies), 3)
    )
    axs[0].set_xticklabels(
        ticks,
        fontsize=9,
        ma='right',
        rotation=90,
        va='top'
    )
    axs[0].set_ylabel('Prediction Accuracy')
    axs[0].text(
        -0.35,
        0.9,
        'A',
        fontsize=14,
        fontweight='bold',
        transform=axs[0].transAxes
    )

    # AUC-ROC Curves

    cmtf_probabilities = train_probabilities.loc[:, 'CMTF']
    train_probabilities = train_probabilities.drop(
        ['CMTF', 'Plasma IL-10', 'Serum IL-10'],
        axis=1
    )
    legend_lines = []
    legend_names = []
    for d_type, color in zip(train_probabilities.columns,
                             COLOR_CYCLE[:train_probabilities.shape[1]]):
        col = train_probabilities.loc[:, d_type].dropna()
        cmtf_col = cmtf_probabilities.loc[col.index]

        actual = train_samples.loc[col.index, 'Actual'].astype(int)
        fpr, tpr, _ = roc_curve(actual, col)
        cmtf_fpr, cmtf_tpr, _ = roc_curve(actual, cmtf_col)

        axs[1].plot(cmtf_fpr, cmtf_tpr, color=color)
        axs[1].plot(fpr, tpr, color=color, linestyle='--')

        legend_lines.append(Line2D([0], [0], color=color, linewidth=0.5))
        legend_names.append(d_type)

    legend_lines.append(Line2D([0], [0], color='k', linestyle='-', linewidth=0.5))
    legend_lines.append(Line2D([0], [0], color='k', linestyle='--', linewidth=0.5))
    legend_names.extend(['CMTF', 'Raw Data'])

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(legend_lines, legend_names)
    axs[1].plot([0, 1], [0, 1], color='k', linestyle='--')
    axs[1].text(
        -0.35,
        0.9,
        'B',
        fontsize=14,
        fontweight='bold',
        transform=axs[1].transAxes
    )

    # Validation Accuracies

    val_accuracies, val_cmtf = get_accuracies(validation_samples)
    axs[2].bar(
        np.arange(0, 3 * len(val_accuracies), 3),
        val_accuracies,
        width=1
    )
    axs[2].bar(
        np.arange(1, 3 * len(val_cmtf), 3),
        val_cmtf,
        width=1
    )

    axs[2].set_xlim(-1, 3 * len(accuracies) - 1)
    axs[2].set_ylim(0, 1)
    axs[2].legend(['Raw Data', 'CMTF'])
    axs[2].set_xticks(
        np.arange(0.5, 3 * len(val_accuracies), 3)
    )
    axs[2].set_xticklabels(
        ticks,
        fontsize=9,
        ma='right',
        rotation=90,
        va='top'
    )
    axs[2].set_ylabel('Prediction Accuracy')
    axs[2].text(
        -0.35,
        0.9,
        'C',
        fontsize=14,
        fontweight='bold',
        transform=axs[2].transAxes
    )

    # Validation AUC-ROC Curves

    cmtf_probabilities = validation_probabilities.loc[:, 'CMTF']
    validation_probabilities = validation_probabilities.drop(
        ['CMTF', 'Plasma IL-10', 'Serum IL-10'],
        axis=1
    )
    for d_type, color in zip(validation_probabilities.columns,
                             COLOR_CYCLE[:validation_probabilities.shape[1]]):
        col = validation_probabilities.loc[:, d_type].dropna()
        cmtf_col = cmtf_probabilities.loc[col.index]

        actual = validation_samples.loc[col.index, 'Actual'].astype(int)
        fpr, tpr, _ = roc_curve(actual, col)
        cmtf_fpr, cmtf_tpr, _ = roc_curve(actual, cmtf_col)

        axs[3].plot(fpr, tpr, color=color, linestyle='--')
        axs[3].plot(cmtf_fpr, cmtf_tpr, color=color)

    axs[3].set_xlim(0, 1)
    axs[3].set_ylim(0, 1)
    axs[3].legend(legend_lines, legend_names)
    axs[3].set_xlabel('False Positive Rate')
    axs[3].set_ylabel('True Positive Rate')
    axs[3].plot([0, 1], [0, 1], color='k', linestyle='--')
    axs[3].text(
        -0.35,
        0.9,
        'D',
        fontsize=14,
        fontweight='bold',
        transform=axs[3].transAxes
    )

    # Metadata predictions

    axs[4].remove()
    axs[5].remove()
    gs = axs[4].get_gridspec()
    meta_ax = fig.add_subplot(gs[4:6])

    sex_accuracy = get_accuracy(
        sex_predictions.loc[:, 'CMTF'],
        sex_predictions.loc[:, 'Actual']
    )
    race_accuracy = get_accuracy(
        race_predictions.loc[:, 'CMTF'],
        race_predictions.loc[:, 'Actual']
    )
    age_accuracy = r2_score(
        age_predictions.loc[:, 'Actual'],
        age_predictions.loc[:, 'CMTF']
    )

    meta_performance = pd.DataFrame(
        index=['Sex', 'Race', 'Age'],
        columns=['Accuracy', 'Metric']
    )
    meta_performance.loc[:, 'Accuracy'] = [
        sex_accuracy,
        race_accuracy,
        round(age_accuracy, 4)
    ]
    meta_performance.loc[:, 'Metric'] = \
        ['Balanced Accuracy', 'Balanced Accuracy', 'R-Squared']

    table = meta_ax.table(
        cellText=meta_performance.to_numpy(),
        loc='center',
        rowLabels=meta_performance.index,
        colLabels=meta_performance.columns,
        cellLoc='center',
        rowLoc='center',
        colWidths=[0.25, 0.25],
        colColours=['#DAEBFE'] * meta_performance.shape[1],
        rowColours=['#DAEBFE'] * meta_performance.shape[0]
    )
    table.auto_set_font_size(False)
    table.scale(1.25, 1.5)
    meta_ax.axis('off')

    meta_ax.text(
        0.1,
        2.5,
        'E',
        fontsize=14,
        fontweight='bold',
        transform=meta_ax.transAxes
    )

    return fig


def export_results(train_samples, train_probabilities, validation_samples,
                   validation_probabilities, sex_predictions, race_predictions,
                   age_predictions):
    """
    Reformats prediction DataFrames and saves as .txt.

    Parameters:
        train_samples (pandas.DataFrame): predictions for training samples
        train_probabilities (pandas.DataFrame): probabilities of persistence for
            training samples
        validation_samples (pandas.DataFrame): predictions for validation
            samples
        validation_probabilities (pandas.DataFrame): probabilities of
            persistence for validation samples
        sex_predictions (pandas.DataFrame): sex predictions for samples with
            known outcomes and sex
        race_predictions (pandas.DataFrame): race predictions for samples
            with known outcomes and race
        age_predictions (pandas.DataFrame): age predictions for samples with
            known outcomes and age

    Returns:
        None
    """
    train_samples = train_samples.astype(str)
    train_probabilities = train_probabilities.astype(str)

    train_samples = train_samples.replace('0', 'ARMB')
    train_samples = train_samples.replace('1', 'APMB')

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
    validation_samples.to_csv(
        join(
            PATH_HERE,
            '..',
            'output',
            'validation_predictions.txt'
        )
    )
    validation_probabilities.to_csv(
        join(
            PATH_HERE,
            '..',
            'output',
            'validation_probabilities.txt'
        )
    )
    sex_predictions.to_csv(
        join(
            PATH_HERE,
            '..',
            'output',
            'sex_predictions.txt'
        )
    )
    race_predictions.to_csv(
        join(
            PATH_HERE,
            '..',
            'output',
            'race_predictions.txt'
        )
    )
    age_predictions.to_csv(
        join(
            PATH_HERE,
            '..',
            'output',
            'age_predictions.txt'
        )
    )


def makeFigure():
    plasma_cyto, serum_cyto = import_cytokines()
    rna = import_rna()
    components, _, patient_data = get_factors()
    components = components[1][0]

    data_types = [
        ('Plasma Cytokines', plasma_cyto.T),
        ('Plasma IL-10', plasma_cyto.loc['IL-10', :]),
        ('Serum Cytokines', serum_cyto.T),
        ('Serum IL-10', serum_cyto.loc['IL-10', :]),
        ('RNA Modules', rna),
        ('CMTF', pd.DataFrame(
            components,
            index=patient_data.index,
            columns=list(range(1, components.shape[1] + 1))
        )
        )
    ]

    validation_samples, validation_probabilities = \
        run_validation(data_types, patient_data)
    train_samples, train_probabilities, sex_predictions, race_predictions = \
        run_cv(data_types, patient_data)
    age_predictions = run_age_regression(data_types[-1][-1], patient_data)

    # export_results(train_samples, train_probabilities, validation_samples,
    #                validation_probabilities, sex_predictions, race_predictions,
    #                age_predictions)

    age_predictions = age_predictions.loc[:, ['CMTF', 'Actual']].dropna(axis=1)
    sex_predictions = sex_predictions.loc[:, ['CMTF', 'Actual']].dropna(axis=1)
    race_predictions = \
        race_predictions.loc[:, ['CMTF', 'Actual']].dropna(axis=1)

    fig = plot_results(train_samples, train_probabilities, validation_samples,
                       validation_probabilities, sex_predictions,
                       race_predictions, age_predictions)

    return fig
