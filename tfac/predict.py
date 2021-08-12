import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold

from .dataImport import form_tensor
from .tensor import perform_CMTF


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

    if isinstance(labels, pd.Series):
        labels = labels.reset_index(drop=True)
    else:
        labels = pd.Series(labels)

    labels = labels[labels != 'Unknown']
    data = data[labels.index, :]

    model = LogisticRegressionCV(l1_ratios=[0.0, 0.5, 0.8, 1.0], solver="saga", penalty="elasticnet", n_jobs=-1, cv=skf, max_iter=100000)
    model.fit(data, labels)

    scores = np.mean(list(model.scores_.values())[0], axis=0)
    return np.max(scores)


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
