import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold

from .dataImport import get_scaled_tensors
from .tensor import perform_CMTF


def run_model(model, data, labels):
    """
    Runs provided LogisticRegressionCV model with the provided data
    and labels.

    Parameters:
        model (sklearn.linear_model.LogisticRegressionCV): 
            Logistic Regression model
        data (pandas.DataFrame): DataFrame of CMTF components
        labels (pandas.Series): Labels for provided data

    Returns:
        score (float): Accuracy for best-performing model (considers
            l1-ratio and C)
    """
    model.fit(data, labels)

    scores = model.scores_[1]
    scores = np.mean(scores, axis=0)
    scores = np.max(scores, axis=1)
    score = max(scores)

    return score


def evaluate_scaling(model, n_components):
    """
    Evaluates the model's accuracy over a range of variance scaling
    values.

    Parameters:
        model (sklearn.linear_model.LogisticRegressionCV): 
            Logistic Regression model
        n_components (int): Number of components to use in CMTF
        
    Returns:
        by_scaling (pandas.Series): Model accuracy over a range of 
            variance scaling values
    """
    scaling_array = np.logspace(-7, 7, base=2, num=29)
    by_scaling = pd.Series(
        index=scaling_array,
        dtype=float
    )

    for scaling in scaling_array:
        tensor, matrix, labels = get_scaled_tensors(scaling)
        data = perform_CMTF(tensor, matrix, n_components)
        data = data[1][0]
        data = data[labels.index, :]

        score = run_model(model, data, labels)
        by_scaling.loc[scaling] = score

    return by_scaling


def evaluate_components(model, var_scaling):
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
    cmtf_components = np.arange(1, 11)
    by_components = pd.Series(
        index=cmtf_components,
        dtype=float
    )

    tensor, matrix, labels = get_scaled_tensors(var_scaling)
    for n_components in cmtf_components:
        data = perform_CMTF(tensor, matrix, n_components)
        data = data[1][0]
        data = data[labels.index, :]

        score = run_model(model, data, labels)
        by_components.loc[n_components] = score

    return by_components


def run_scaling_analyses(cs, l1_ratios, splits, var_scaling, n_components):
    """
    Evaluates model accuracy with regards to variance scaling and
    CMTF component count.

    Parameters:
        cs (int): Number of C (regularization coefficients) to test;
            logarithmically-spaced between 1E-4 and 1E4
        l1_ratios (int): Number of l1-ratios to test;
            linearly-spaced between 0 and 1 (inclusive)
        splits (int): Number of cross-validation splits to use when
            evaluating model accuracy

    Returns:
        by_scaling (pandas.Series): Model accuracy with regards to
            variance scaling
        by_components (pandas.Series): Model accuracy with regards to
            number of CMTF components
    """
    cs = np.logspace(-4, 4, cs)
    l1_ratios = np.linspace(0, 1, l1_ratios)

    skf = RepeatedStratifiedKFold(
        n_splits=splits,
        random_state=42
    )
    model = LogisticRegressionCV(
        Cs=cs, 
        cv=skf,
        max_iter=100000, 
        l1_ratios=l1_ratios,
        solver='saga',
        penalty='elasticnet',
        n_jobs=10
    )

    by_scaling = evaluate_scaling(model, n_components)
    by_components = evaluate_components(model, var_scaling)

    return by_scaling, by_components
