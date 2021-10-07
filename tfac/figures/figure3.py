"""
Creates Figure 3 -- Model Performance
"""
import numpy as np
from os.path import abspath, dirname, join
import pandas as pd
from sklearn.metrics import r2_score, roc_curve

from tfac.dataImport import import_validation_patient_metadata
from tfac.figures.figureCommon import getSetup, get_data_types
from tfac.predict import predict_known, predict_validation, predict_regression

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
        labels = patient_data.loc[data.index, 'status']

        _predictions = predict_validation(data, labels)
        _probabilities = predict_validation(data, labels, predict_proba=True)
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
            labels = patient_data.loc[data.index, column]

            _predictions = predict_known(data, labels)
            if column == 'status':
                _probabilities = predict_known(
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
    age_predictions = predict_regression(data, labels)
    age_predictions.name = 'CMTF'

    age_predictions = pd.DataFrame(age_predictions)
    age_predictions.loc[:, 'Actual'] = labels.loc[age_predictions.index]

    return age_predictions


def get_accuracies(train_samples):
    """
    Calculates prediction accuracy for training samples with known outcomes.

    Parameters:
        train_samples (pandas.DataFrame): predictions for training samples

    Returns:
        accuracies (pandas.Series): model accuracy w/r to each model type
    """
    d_types = train_samples.columns[:-1]
    accuracies = pd.Series(
        index=d_types
    )

    for d_type in d_types:
        col = train_samples.loc[:, d_type].dropna().astype(int)
        actual = train_samples.loc[col.index, 'Actual'].astype(int)
        correct = [1 if col.loc[i] == actual.loc[i] else 0 for i in col.index]
        accuracies.loc[d_type] = np.mean(correct)

    return accuracies


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
    accuracies = get_accuracies(train_samples)
    axs[0].bar(
        range(len(accuracies)),
        accuracies
    )

    axs[0].set_ylim(0, 1)
    ticks = [label.replace(' ', '\n') for label in accuracies.index]
    axs[0].set_xticks(range(len(accuracies)))
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

    for d_type in train_probabilities.columns:
        col = train_probabilities.loc[:, d_type].dropna()
        actual = train_samples.loc[col.index, 'Actual'].astype(int)
        fpr, tpr, _ = roc_curve(actual, col)
        axs[1].plot(fpr, tpr)

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].legend(train_probabilities.columns)
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
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

    val_accuracies = get_accuracies(validation_samples)
    axs[2].bar(
        range(len(val_accuracies)),
        val_accuracies
    )

    axs[2].set_ylim(0, 1)
    ticks = [label.replace(' ', '\n') for label in val_accuracies.index]
    axs[2].set_xticks(range(len(val_accuracies)))
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

    for d_type in validation_probabilities.columns:
        col = validation_probabilities.loc[:, d_type].dropna()
        actual = validation_samples.loc[col.index, 'Actual'].astype(int)
        fpr, tpr, _ = roc_curve(actual, col)
        axs[3].plot(fpr, tpr)

    axs[3].set_xlim(0, 1)
    axs[3].set_ylim(0, 1)
    axs[3].legend(validation_probabilities.columns)
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

    sex_accuracy = get_accuracies(sex_predictions.loc[:, ['CMTF', 'Actual']])
    race_accuracy = get_accuracies(race_predictions.loc[:, ['CMTF', 'Actual']])
    age_accuracy = r2_score(
        age_predictions.loc[:, 'Actual'],
        age_predictions.loc[:, 'CMTF']
    )

    meta_performance = pd.DataFrame(
        index=['Sex', 'Race', 'Age'],
        columns=['Accuracy', 'Metric']
    )
    meta_performance.loc[:, 'Accuracy'] = [
        sex_accuracy.loc['CMTF'],
        race_accuracy.loc['CMTF'],
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
        rowColours=['#DAEBFE'] * meta_performance.shape[0],
        fontsize=10
    )
    table.auto_set_font_size(False)
    table.scale(1, 1.5)
    meta_ax.axis('off')

    meta_ax.text(
        0.15,
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
    data_types, patient_data = get_data_types()
    validation_samples, validation_probabilities = \
        run_validation(data_types, patient_data)
    train_samples, train_probabilities, sex_predictions, race_predictions = \
        run_cv(data_types, patient_data)
    age_predictions = run_age_regression(data_types[-1][-1], patient_data)

    export_results(train_samples, train_probabilities, validation_samples,
                   validation_probabilities, sex_predictions, race_predictions,
                   age_predictions)

    fig = plot_results(train_samples, train_probabilities, validation_samples,
                       validation_probabilities, sex_predictions,
                       race_predictions, age_predictions)

    return fig
