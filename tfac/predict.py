import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .dataImport import form_tensor, import_patient_metadata
from .tensor import perform_CMTF


def predict_unknown(data, labels):
    """
    """
    c, l1_ratio = fit_lr_params(data, labels)
    model = LogisticRegression(
        C=c,
        l1_ratio=l1_ratio,
        max_iter=100000,
        solver='saga',
        penalty='elasticnet',
    )

    train_labels = labels.loc[labels != 'Unknown']
    test_labels = labels.loc[labels == 'Unknown']

    if isinstance(data, pd.Series):
        train_data = data.loc[train_labels.index]
        test_data = data.loc[test_labels.index]
        train_data = train_data.values.reshape(-1, 1)
        test_data = test_data.values.reshape(-1, 1)
    else:
        train_data = data.loc[train_labels.index, :]
        test_data = data.loc[test_labels.index, :]

    model.fit(train_data, train_labels)
    predicted = model.predict(test_data)
    predictions = pd.Series(predicted)
    predictions.index = test_labels.index

    return predictions


def predict_known(data, labels):
    """
    """
    c, l1_ratio = fit_lr_params(data, labels)
    model = LogisticRegression(
        C=c,
        l1_ratio=l1_ratio,
        max_iter=100000,
        solver='saga',
        penalty='elasticnet',
    )

    predictions = pd.Series(
        index=labels.index,
    )

    skf = RepeatedStratifiedKFold(
        n_splits=10,
        random_state=42
    )

    labels = labels.loc[labels != 'Unknown']
    if isinstance(data, pd.Series):
        data = data.loc[labels.index]
    else:
        data = data.loc[labels.index, :]

    for train_index, test_index in skf.split(data, labels):
        train_data, train_out = data.iloc[train_index], labels.iloc[train_index]
        test_data, test_out = data.iloc[test_index], labels.iloc[test_index]
        train_out = train_out.values.ravel()

        if isinstance(train_data, pd.Series):
            train_data = train_data.values.reshape(-1, 1)
            test_data = test_data.values.reshape(-1, 1)

        model.fit(train_data, train_out)
        predicted = model.predict(test_data)
        predictions.loc[test_out.index] = predicted

    return predictions


def fit_lr_params(data, labels):
    cv_model, _ = run_model(data, labels)
    return cv_model.C_[0], cv_model.l1_ratio_[0]


def run_model(data, labels):
    """
    Runs provided LogisticRegressionCV model with the provided data
    and labels.

    Parameters:
        data (pandas.DataFrame): DataFrame of CMTF components
        labels (pandas.Series): Labels for provided data

    Returns:
        score (float): Accuracy for best-performing model (considers
            l1-ratio and C)
    """
    skf = RepeatedStratifiedKFold(
        n_splits=10,
        random_state=42
    )

    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)

    known_out = labels != 'Unknown'
    labels = labels.loc[known_out]

    if isinstance(data, pd.Series):
        data = data.loc[known_out]
        data = data.values.reshape(-1, 1)
    else:
        data = data.loc[known_out, :]

    model = LogisticRegressionCV(
        l1_ratios=[0.0, 0.5, 0.8, 1.0],
        solver="saga",
        penalty="elasticnet",
        n_jobs=-1,
        cv=skf,
        max_iter=100000,
        scoring='balanced_accuracy_score'
    )
    model.fit(data, labels)

    scores = np.mean(list(model.scores_.values())[0], axis=0)
    return model, np.max(scores)


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

        score = run_model(data, labels)
        by_scaling.loc[scaling] = score

    return by_scaling


def evaluate_components(var_scaling):
    """
    Evaluates the model's accuracy over a range of CMTF component
    counts.

    Parameters:
        var_scaling (float): Variance scaling (RNA/cytokine)

    Returns:
        by_scaling (pandas.Series): Model accuracy over a range of
            CMTF component counts
    """
    by_components = pd.Series(
        index=np.arange(1, 13).tolist(),
        dtype=float
    )

    tensor, matrix, patient_data = form_tensor(var_scaling)
    labels = patient_data.loc[:, 'status']
    for n_components, _ in by_components.items():
        data = perform_CMTF(tensor, matrix, n_components)
        data = data[1][0]

        score = run_model(data, labels)
        by_components.loc[n_components] = score

    return by_components


def run_scaling_analyses(var_scaling):
    """
    Evaluates model accuracy with regards to variance scaling and
    CMTF component count.

    Parameters:
        var_scaling (float): Variance scaling (Cytokine/RNA)

    Returns:
        by_scaling (pandas.Series): Model accuracy with regards to
            variance scaling
        by_components (pandas.Series): Model accuracy with regards to
            number of CMTF components
    """
    by_scaling = evaluate_scaling()
    by_components = evaluate_components(var_scaling)

    return by_scaling, by_components
