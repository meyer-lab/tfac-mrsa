import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

from .dataImport import form_tensor, import_validation_patient_metadata
from tensorpack import perform_CMTF

warnings.filterwarnings('ignore', category=UserWarning)


def predict_validation(data, labels, predict_proba=False, return_coef=False):
    """
    Trains a LogisticRegressionCV model using samples with known outcomes,
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
    Runs provided LogisticRegressionCV model with the provided data
    and labels.

    Parameters:
        data (pandas.DataFrame): DataFrame of CMTF components
        labels (pandas.Series): Labels for provided data
        return_coef (bool, default: False): return model coefficients

    Returns:
        score (float): Accuracy for best-performing model (considers
            l1-ratio and C)
        model (sklearn.LogisticRegressionCV)
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

    model = LogisticRegressionCV(
        l1_ratios=[0.8],
        solver="saga",
        penalty="elasticnet",
        n_jobs=-1,
        cv=skf,
        max_iter=100000,
        scoring='balanced_accuracy',
        multi_class='ovr'
    )
    model.fit(data, labels)

    coef = None
    if return_coef:
        coef = model.coef_[0]

    scores = np.mean(list(model.scores_.values())[0], axis=0)

    model = LogisticRegression(
        C=model.C_[0],
        l1_ratio=model.l1_ratio_[0],
        solver="saga",
        penalty="elasticnet",
        n_jobs=-1,
        max_iter=100000,
    )

    if return_coef:
        return np.max(scores), model, coef
    else:
        return np.max(scores), model


def evaluate_scaling():
    """
    Evaluates the model's accuracy over a range of variance scaling
    values.

    Parameters:
        None

    Returns:
        by_scaling (pandas.Series): Model accuracy over a range of
            variance scaling values
    """
    by_scaling = pd.Series(
        index=np.logspace(-7, 7, base=2, num=29).tolist(),
        dtype=float
    )

    for scaling, _ in by_scaling.items():
        tensor, matrix, patient_data = form_tensor(scaling)
        labels = patient_data.loc[:, 'status']

        data = perform_CMTF(tensor, matrix)
        data = data[1][0]

        score, _ = run_model(data, labels)
        by_scaling.loc[scaling] = score

    return by_scaling


def evaluate_components():
    """
    Evaluates the model's accuracy over a range of CMTF component
    counts.

    Returns:
        by_scaling (pandas.Series): Model accuracy over a range of
            CMTF component counts
    """
    by_components = pd.Series(
        index=np.arange(1, 13).tolist(),
        dtype=float
    )

    tensor, matrix, patient_data = form_tensor()
    labels = patient_data.loc[:, 'status']
    for n_components, _ in by_components.items():
        data = perform_CMTF(tensor, matrix, n_components)
        data = data[1][0]

        score, _ = run_model(data, labels)
        by_components.loc[n_components] = score

    return by_components
