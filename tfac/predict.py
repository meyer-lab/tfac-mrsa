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

    # TODO: Remove validation samples here since you're passing in the labels. Clearly Unknown should be left out of training.

    model = LogisticRegressionCV(l1_ratios=[0.0, 0.5, 0.8, 1.0], solver="saga", penalty="elasticnet", n_jobs=10, cv=skf, max_iter=100000)
    model.fit(data, labels)

    scores = np.mean(model.scores_[1], axis=0)
    return np.max(scores)


def evaluate_scaling():
    """
    Evaluates the model's accuracy over a range of variance scaling
    values.

    Parameters:
        n_components (int): Number of components to use in CMTF

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
        model (sklearn.linear_model.LogisticRegressionCV):
            Logistic Regression model
        var_scaling (float): Variance scaling (RNA/cytokine)

    Returns:
        by_scaling (pandas.Series): Model accuracy over a range of
            CMTF component counts
    """
    by_components = pd.Series(
        index=np.arange(1, 11).tolist(),
        dtype=float
    )

    tensor, matrix, patient_data = form_tensor(var_scaling)
    labels = patient_data.loc[:, 'status']
    for n_components, _ in by_components.items():
        data = perform_CMTF(tensor, matrix, n_components)
        data = data[1][0]
        data = data[labels.index, :]

        score = run_model(data, labels)
        by_components.loc[n_components] = score

    return by_components


def run_scaling_analyses(cs, l1_ratios, var_scaling):
    """
    Evaluates model accuracy with regards to variance scaling and
    CMTF component count.

    Parameters:
        cs (int): Number of C (regularization coefficients) to test;
            logarithmically-spaced between 1E-4 and 1E4
        l1_ratios (int): Number of l1-ratios to test;
            linearly-spaced between 0 and 1 (inclusive)

    Returns:
        by_scaling (pandas.Series): Model accuracy with regards to
            variance scaling
        by_components (pandas.Series): Model accuracy with regards to
            number of CMTF components
    """
    cs = np.logspace(-4, 4, cs)
    l1_ratios = np.linspace(0, 1, l1_ratios)

    by_scaling = evaluate_scaling()
    by_components = evaluate_components(var_scaling)

    return by_scaling, by_components
