"""
Creates Figure 5 -- SVM Model
"""
import matplotlib
import numpy as np
from os.path import abspath, dirname
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.svm import SVC

from .common import getSetup
from ..dataImport import import_validation_patient_metadata, get_factors
from ..predict import predict_known, predict_validation

COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
SUBSET_CYTO = [3, 4, 6]
PATH_HERE = dirname(dirname(abspath(__file__)))


def run_validation(components, patient_data):
    """
    Predicts validation samples.

    Parameters:
        components (numpy.array): CMTF components
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        predictions (pandas.DataFrame): predictions for each model
    """
    predictions = pd.DataFrame(
        index=patient_data.index,
        columns=['LR', 'SVM']
    )
    predictions = predictions.loc[patient_data['status'] == 'Unknown']
    probabilities = predictions.copy()
    validation_meta = import_validation_patient_metadata()

    labels = patient_data.loc[components.index, 'status']

    predictions.loc[:, 'LR'] = predict_validation(components, labels)
    probabilities.loc[:, 'LR'] = predict_validation(
        components,
        labels,
        predict_proba=True
    )

    svm_pred, svm_prob = predict_subsets(components, labels, validation_meta)
    predictions.loc[:, 'SVM'] = svm_pred
    probabilities.loc[:, 'SVM'] = svm_prob

    predictions.loc[:, 'Actual'] = validation_meta.loc[:, 'status']
    return predictions, probabilities


def predict_subsets(data, labels, validation_meta):
    """
    Predicts persistence for validation cohort using SVM subset.

    Parameters:
        data (pandas.DataFrame): CMTF components
        labels (pandas.Series): patient persistence labels
        validation_meta (pandas.DataFrame): validation cohort metadata

    Returns:
        predicted (numpy.array): SVM persistence predictions for validation
            cohort
        probabilities (numpy.array): SVM persistence probabilities for
            validation cohort
    """
    train_data = data.drop(validation_meta.index)
    train_labels = labels.drop(validation_meta.index)
    val_data = data.loc[validation_meta.index]

    model = SVC(probability=True)
    model.fit(train_data.loc[:, SUBSET_CYTO], train_labels)
    predicted = model.predict(val_data.loc[:, SUBSET_CYTO])
    probabilities = model.predict_proba(val_data.loc[:, SUBSET_CYTO])
    probabilities = probabilities[:, 1]

    return predicted, probabilities


def run_cv(components, patient_data):
    """
    Predicts samples with known outcomes via cross-validation.

    Parameters:
        components (numpy.array): CMTF components
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        predictions (pandas.Series): predictions for each data source
    """
    patient_data = patient_data.loc[patient_data.loc[:, 'status'] != 'Unknown']
    components = components.loc[patient_data.index, :]
    labels = patient_data.loc[components.index, 'status']

    predictions = pd.DataFrame(
        index=patient_data.index,
        columns=['LR', 'SVM']
    )
    probabilities = predictions.copy()

    predictions.loc[:, 'LR'] = predict_known(components, labels)
    probabilities.loc[:, 'LR'] = predict_known(
        components,
        labels,
        method='predict_proba'
    )

    svm_pred, svm_prob = subset_cv(components, labels)
    predictions.loc[:, 'SVM'] = svm_pred
    probabilities.loc[:, 'SVM'] = svm_prob

    predictions.loc[:, 'Actual'] = patient_data.loc[:, 'status']

    return predictions, probabilities


def subset_cv(data, labels):
    """
    Predicts persistence over training data using SVM subset.

    Parameters:
        data (pandas.DataFrame): CMTF components
        labels (pandas.Series): patient persistence labels

    Returns:
        predicted (numpy.array): SVM persistence predictions for validation
            cohort
        probabilities (numpy.array): SVM persistence probabilities for
            validation cohort
    """
    model = SVC(probability=True)
    skf = StratifiedKFold(
        n_splits=10,
    )

    predicted = cross_val_predict(
        model,
        data.loc[:, SUBSET_CYTO],
        labels,
        cv=skf
    )
    probabilities = cross_val_predict(
        model,
        data.loc[:, SUBSET_CYTO],
        labels,
        cv=skf,
        method='predict_proba'
    )
    probabilities = probabilities[:, 1]

    return predicted, probabilities


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


def plot_results(train_samples, train_probabilities, validation_samples,
                 validation_probabilities):
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

    Returns:
        fig (matplotlib.Figure): figure depicting predictions for all samples
    """
    fig_size = (5, 5)
    layout = {
        'ncols': 2,
        'nrows': 2,
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
        width=1
    )

    axs[0].set_xlim(-1, 2)
    axs[0].set_ylim(0, 1)
    axs[0].set_xticks(
        np.arange(0, 2, 1)
    )
    axs[0].set_xticklabels(
        accuracies.index,
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

    svc_fpr, svc_tpr, _ = roc_curve(
        train_samples.loc[:, 'Actual'],
        train_probabilities.loc[:, 'SVM']
    )
    lr_fpr, lr_tpr, _ = roc_curve(
        train_samples.loc[:, 'Actual'],
        train_probabilities.loc[:, 'LR']
    )

    axs[1].plot(lr_fpr, lr_tpr, color=COLOR_CYCLE[2])
    axs[1].plot(svc_fpr, svc_tpr, color=COLOR_CYCLE[3])

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(['LR', 'SVM'])
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
        [0, 1],
        val_accuracies,
        color=COLOR_CYCLE[:2],
        width=1
    )

    axs[2].set_xlim(-1, 2)
    axs[2].set_ylim(0, 1)
    axs[2].set_xticks(
        np.arange(0, 2, 1)
    )
    axs[2].set_xticklabels(
        val_accuracies.index,
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

    svc_fpr, svc_tpr, _ = roc_curve(
        validation_samples.loc[:, 'Actual'],
        validation_probabilities.loc[:, 'SVM']
    )
    lr_fpr, lr_tpr, _ = roc_curve(
        validation_samples.loc[:, 'Actual'],
        validation_probabilities.loc[:, 'LR']
    )

    axs[3].plot(lr_fpr, lr_tpr, color=COLOR_CYCLE[2])
    axs[3].plot(svc_fpr, svc_tpr, color=COLOR_CYCLE[3])

    axs[3].set_xlim(0, 1)
    axs[3].set_ylim(0, 1)
    axs[3].set_xlabel('False Positive Rate')
    axs[3].set_ylabel('True Positive Rate')
    axs[3].legend(['LR', 'SVM'])
    axs[3].plot([0, 1], [0, 1], color='k', linestyle='--')
    axs[3].text(
        -0.35,
        0.9,
        'D',
        fontsize=14,
        fontweight='bold',
        transform=axs[3].transAxes
    )

    return fig


def makeFigure():
    t_fac, patient_data = get_factors()
    components = t_fac[1][0]
    components = pd.DataFrame(
        components,
        index=patient_data.index,
        columns=list(range(1, components.shape[1] + 1))
    )

    validation_samples, validation_probabilities = \
        run_validation(components, patient_data)
    train_samples, train_probabilities = run_cv(components, patient_data)

    train_samples = train_samples.astype(int)
    validation_samples = validation_samples.astype(int)

    fig = plot_results(train_samples, train_probabilities, validation_samples,
                       validation_probabilities)

    return fig
