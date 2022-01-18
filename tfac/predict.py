import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score

from .dataImport import form_tensor, import_validation_patient_metadata

warnings.filterwarnings('ignore', category=UserWarning)


def predict_validation(data, labels, predict_proba=False, return_coef=False):
    """
    Trains a LogisticRegression model using samples with known outcomes,
    then predicts samples with unknown outcomes.

    Parameters:
        data (pandas.DataFrame): data to classify
        labels (pandas.Series): labels for samples in data
        predict_proba (bool, default: False): predict probability of positive
            case
        return_coef (bool, default:False): return model coefficients

    Returns:
        predictions (pandas.Series): predictions for samples with unknown
            outcomes
    """
    validation_data = import_validation_patient_metadata()
    validation_samples = set(validation_data.index) & set(labels.index)

    train_labels = labels.drop(validation_samples)
    test_labels = labels.loc[validation_samples]

    if isinstance(data, pd.Series):
        train_data = data.loc[train_labels.index]
        test_data = data.loc[test_labels.index]
    else:
        train_data = data.loc[train_labels.index, :]
        test_data = data.loc[test_labels.index, :]

    _, model = run_model(train_data, train_labels)

    if isinstance(data, pd.Series):
        train_data = train_data.values.reshape(-1, 1)
        test_data = test_data.values.reshape(-1, 1)

    model.fit(train_data, train_labels)

    if predict_proba:
        predicted = model.predict_proba(test_data)
        predicted = predicted[:, -1]
    else:
        predicted = model.predict(test_data)

    predictions = pd.Series(predicted)
    predictions.index = test_labels.index

    if return_coef:
        return predictions, model.coef_[0]
    else:
        return predictions


def predict_known(data, labels, method='predict'):
    """
    Predicts outcomes for all samples in data via cross-validation.

    Parameters:
        data (pandas.DataFrame): data to classify
        labels (pandas.Series): labels for samples in data
        method (str, default: 'predict'): prediction method to use; accepts any
            of ‘predict’, ‘predict_proba’, ‘predict_log_proba’, or ‘decision_function’

    Returns:
        predictions (pandas.Series): predictions for samples
    """
    labels = labels.loc[labels != 'Unknown']

    if isinstance(data, pd.Series):
        data = data.loc[labels.index]
    else:
        data = data.loc[labels.index, :]

    _, model = run_model(data, labels)
    skf = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    if isinstance(data, pd.Series):
        data = data.values.reshape(-1, 1)

    predictions = cross_val_predict(
        model,
        data,
        labels,
        cv=skf,
        method=method,
        n_jobs=-1
    )

    if len(predictions.shape) > 1:
        predictions = predictions[:, -1]

    predictions = pd.Series(
        predictions,
        index=labels.index
    )

    return predictions


def predict_regression(data, labels, return_coef=False):
    """
    Predicts value for all samples in data via cross-validation.

    Parameters:
        data (pandas.DataFrame): data to classify
        labels (pandas.Series): labels for samples in data
        return_coef (bool, default: False): return model coefficients

    Returns:
        predictions (pandas.Series): predictions for samples
    """
    model = LinearRegression()
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    if isinstance(data, pd.Series):
        data = data.values.reshape(-1, 1)

    predictions = cross_val_predict(
        model,
        data,
        labels,
        cv=skf,
        n_jobs=-1
    )

    if len(predictions.shape) > 1:
        predictions = predictions[:, -1]

    predictions = pd.Series(
        predictions,
        index=labels.index
    )

    if return_coef:
        model.fit(data, labels)
        return predictions, model.coef_
    else:
        return predictions


def run_model(data, labels, return_coef=False):
    """
    Runs provided LogisticRegression model with the provided data
    and labels.

    Parameters:
        data (pandas.DataFrame): DataFrame of CMTF components
        labels (pandas.Series): Labels for provided data
        return_coef (bool, default: False): return model coefficients

    Returns:
        score (float): Accuracy for best-performing model (considers
            l1-ratio and C)
        model (sklearn.LogisticRegression)
    """
    skf = RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=15
    )

    if isinstance(labels, pd.Series):
        labels = labels.reset_index(drop=True)
    else:
        labels = pd.Series(labels)

    labels = labels[labels != 'Unknown']

    if isinstance(data, pd.Series):
        data = data.iloc[labels.index]
        data = data.values.reshape(-1, 1)
    elif isinstance(data, pd.DataFrame):
        data = data.iloc[labels.index, :]
    else:
        data = data[labels.index, :]

    model = LogisticRegression(
        penalty="none",
        max_iter=100000,
        multi_class="ovr"
    )
    model.fit(data, labels)

    score = cross_val_score(model, data, labels, cv=skf, scoring="balanced_accuracy", n_jobs=-1)

    if return_coef:
        return score, model, np.squeeze(model.coef_[0, :])
    else:
        return score, model


def evaluate_accuracy(data):
    """
    Evaluates the model's accuracy for a given subject factors matrix.

    Parameters:
        Subject factors matrix.

    Returns:
        Model accuracy
    """
    _, _, patient_data = form_tensor()
    labels = patient_data.loc[:, 'status']

    score, _ = run_model(data, labels)
    return score
