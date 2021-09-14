"""
Creates Figure 3 -- Model Performance Plotting
"""
import numpy as np
from os.path import abspath, dirname, join
import pandas as pd
from sklearn.metrics import roc_curve

from tfac.figures.figureCommon import getSetup, get_data_types
from tfac.predict import predict_known, predict_unknown, predict_regression

PATH_HERE = dirname(dirname(abspath(__file__)))


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
                _probabilities = predict_known(data, labels, method='predict_proba')
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


def plot_results(cv_results, cv_probabilities, sex_predictions,
                 race_predictions, age_predictions):
    """
    Plots prediction model performance.

    Parameters:
        cv_results (pandas.DataFrame): predictions for samples with known
            outcomes
        cv_probabilities (pandas.DataFrame): predicted probability of
            persistence for samples with known outcomes
        sex_predictions (pandas.DataFrame): sex predictions for samples with
            known outcomes and sex
        race_predictions (pandas.DataFrame): race predictions for samples
            with known outcomes and predictions
        age_predictions (pandas.DataFrame): age predictions for samples with
            known outcomes and age

    Returns:
        fig (matplotlib.Figure): figure depicting predictions for all samples
    """
    fig_size = (8, 8)
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

    axs[0].set_ylim(0, 1)
    ticks = [label.replace(' ', '\n') for label in accuracies.index]
    axs[0].set_xticks(range(len(accuracies)))
    axs[0].set_xticklabels(ticks, fontsize=10)
    axs[0].set_title('Persistence Prediction Accuracy', fontsize=14)
    axs[0].set_ylabel('Mean Accuracy over\n10-fold Cross-Validation', fontsize=12)
    axs[0].text(
        -2,
        1,
        'A',
        fontsize=16,
        fontweight='bold',
    )

    # AUC-ROC Curves

    for d_type in cv_probabilities.columns:
        col = cv_probabilities.loc[:, d_type].dropna()
        actual = cv_results.loc[col.index, 'Actual'].astype(int)
        fpr, tpr, _ = roc_curve(actual, col)
        axs[1].plot(fpr, tpr)

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].legend(cv_probabilities.columns)
    axs[1].set_title('ROC Curves', fontsize=14)
    axs[1].set_xlabel('False Positive Rate', fontsize=12)
    axs[1].set_ylabel('True Positive Rate', fontsize=12)
    axs[1].plot([0, 1], [0, 1], color='k', linestyle='--')
    axs[1].text(
        -0.25,
        1,
        'B',
        fontsize=16,
        fontweight='bold',
    )

    # Sex Predictions

    sex_accuracies = get_accuracies(sex_predictions)
    axs[2].bar(
        range(len(sex_accuracies)),
        sex_accuracies
    )

    sex_ticks = [label.replace(' ', '\n') for label in sex_accuracies.index]
    axs[2].set_ylim(0, 1)
    axs[2].set_xticks(range(len(sex_accuracies)))
    axs[2].set_xticklabels(sex_ticks, fontsize=10)
    axs[2].set_title('Sex Prediction Accuracy', fontsize=14)
    axs[2].set_ylabel('Mean Accuracy over\n10-fold Cross-Validation', fontsize=12)
    axs[2].text(
        -2,
        1,
        'C',
        fontsize=16,
        fontweight='bold',
    )

    # Race Predictions

    race_accuracies = get_accuracies(race_predictions)
    axs[3].bar(
        range(len(race_accuracies)),
        race_accuracies
    )

    race_ticks = [label.replace(' ', '\n') for label in race_accuracies.index]
    axs[3].set_ylim(0, 1)
    axs[3].set_xticks(range(len(race_accuracies)))
    axs[3].set_xticklabels(race_ticks, fontsize=10)
    axs[3].set_title('Race Prediction Accuracy', fontsize=14)
    axs[3].set_ylabel('Mean Accuracy over\n10-fold Cross-Validation', fontsize=12)
    axs[3].text(
        -2,
        1,
        'D',
        fontsize=16,
        fontweight='bold',
    )

    # Age Predictions

    axs[4].remove()
    axs[5].remove()
    gs = axs[4].get_gridspec()
    age_ax = fig.add_subplot(gs[4:6])

    age_ax.scatter(
        age_predictions.loc[:, 'Actual'],
        age_predictions.loc[:, 'CMTF']
    )
    age_ax.plot(
        [0, 100],
        [0, 100],
        color='k',
        linestyle='--'
    )

    age_ax.set_xlim(0, 100)
    age_ax.set_ylim(0, 100)
    age_ax.set_title('CMTF Age Predictions', fontsize=14)
    age_ax.set_xlabel('Actual Patient Age', fontsize=12)
    age_ax.set_ylabel('Predicted Patient Age', fontsize=12)
    age_ax.text(
        -11,
        100,
        'E',
        fontsize=16,
        fontweight='bold',
    )

    return fig


def export_results(train_samples, train_probabilities, sex_predictions,
                   race_predictions, age_predictions):
    """
    Reformats prediction DataFrames and saves as .txt.

    Parameters:
        train_samples (pandas.DataFrame): predictions for training samples
        train_probabilities (pandas.DataFrame): probabilities of persistence for
            training samples
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
    train_samples, train_probabilities, sex_predictions, race_predictions = \
        run_cv(data_types, patient_data)
    age_predictions = run_age_regression(data_types[-1][-1], patient_data)

    export_results(train_samples, train_probabilities,
                   sex_predictions, race_predictions, age_predictions)

    fig = plot_results(train_samples, train_probabilities,
                       sex_predictions, race_predictions, age_predictions)

    return fig
