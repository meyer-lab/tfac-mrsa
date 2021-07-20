import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
Runs hyperparameter optimization for a Logistic Regression model that 
uses CMTF components to classify MRSA persistance. Generates a figure
depicting model accuracy against scaling and component count.
"""
import argparse
import os
import sys

sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm

from dataImport import form_missing_tensor
from figureCommon import getSetup
from tensor import perform_CMTF

OPTIMAL_COMPONENTS = 9
OPTIMAL_SCALING = 2 ** 5


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


def get_scaled_tensors(scaling):
    """
    Creates scaled CMTF tensor and matrix.

    Parameters:
        scaling (float): Ratio of variance between RNA and cytokine
            data
    
    Returns:
        tensor (np.array): CMTF data tensor
        matrix (np.array): CMTF data matrix
        labels (pandas.Series): Patient outcome labels
    """
    slices, _, _, pat_info = form_missing_tensor(scaling)
    pat_info = pat_info.T.reset_index()

    tensor = np.stack(
            (slices[0], slices[1])
        ).T
    matrix = slices[2].T
    labels = pat_info.loc[:, 'status']
    labels = labels.loc[labels != 'Unknown'].astype(int)
    
    return tensor, matrix, labels


def evaluate_scaling(model):
    """
    Evaluates the model's accuracy over a range of variance scaling
    values.

    Parameters:
        model (sklearn.linear_model.LogisticRegressionCV): 
            Logistic Regression model
        
    Returns:
        by_scaling (pandas.Series): Model accuracy over a range of 
            variance scaling values
    """
    scaling_array = np.logspace(-7, 7, base=2, num=29)
    by_scaling = pd.Series(
        index=scaling_array,
        dtype=float
    )

    for scaling in tqdm(scaling_array):
        tensor, matrix, labels = get_scaled_tensors(scaling)
        data = perform_CMTF(tensor, matrix, OPTIMAL_COMPONENTS)
        data = data[1][0]
        data = data[labels.index, :]

        score = run_model(model, data, labels)
        by_scaling.loc[scaling] = score

    return by_scaling


def evaluate_components(model):
    """
    Evaluates the model's accuracy over a range of CMTF component
    counts.

    Parameters:
        model (sklearn.linear_model.LogisticRegressionCV): 
            Logistic Regression model
        
    Returns:
        by_scaling (pandas.Series): Model accuracy over a range of 
            CMTF component counts
    """
    cmtf_components = np.arange(1, 11)
    by_components = pd.Series(
        index=cmtf_components,
        dtype=float
    )

    tensor, matrix, labels = get_scaled_tensors(OPTIMAL_SCALING)
    for n_components in tqdm(cmtf_components):
        data = perform_CMTF(tensor, matrix, n_components)
        data = data[1][0]
        data = data[labels.index, :]

        score = run_model(model, data, labels)
        by_components.loc[n_components] = score

    return by_components


def run_scaling_analyses(cs, l1_ratios, splits):
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
        penalty='elasticnet'
    )

    by_scaling = evaluate_scaling(model)
    by_components = evaluate_components(model)

    return by_scaling, by_components


def plot_results(by_scaling, by_components):
    """
    Plots accuracy of model with regards to variance scaling and CMTF
    component parameters.

    Parameters:
        by_scaling (pandas.Series): Model accuracy with regards to
            variance scaling
        by_components (pandas.Series): Model accuracy with regards to
            number of CMTF components

    Returns:
        fig (matplotlib.pyplot.Figure): Figure containing plots of 
            scaling and CMTF component analyses
    """
    # Sets up plotting space
    fig_size = (8, 4)
    layout = (1, 2)
    axs, fig = getSetup(
        fig_size,
        layout
    )

    # Model Performance v. Scaling

    axs[0].semilogx(by_scaling.index, by_scaling, base=2)
    axs[0].set_ylabel('Best Accuracy over Repeated\n10-fold Cross Validation', fontsize=12)
    axs[0].set_xlabel('Variance Scaling (RNA/Cytokine)', fontsize=12)
    axs[0].set_ylim([0.5, 0.8])
    axs[0].set_xticks(np.logspace(-7, 7, base=2, num=8))
    axs[0].text(0.02, 0.9, 'A', fontsize=16, fontweight='bold', transform=plt.gcf().transFigure)

    # Best Scaling v. CMTF Components

    axs[1].plot(by_components.index, by_components)
    axs[1].set_ylabel('Best Accuracy over Repeated\n10-fold Cross Validation', fontsize=12)
    axs[1].set_xlabel('CMTF Components', fontsize=12)
    axs[1].set_xticks(by_components.index)
    axs[1].set_ylim([0.5, 0.8])
    axs[1].text(0.52, 0.9, 'B', fontsize=16, fontweight='bold', transform=plt.gcf().transFigure)

    return fig


def makeFigure():
    """
    Generates Figure 4.

    Parameters:
        None

    Returns:
        fig (matplotlib.pyplot.Figure): Figure containing plots of 
            scaling and CMTF component analyses    
    """
    cs = 100
    l1_ratios = 11
    splits = 10

    by_scaling, by_components = run_scaling_analyses(
        cs,
        l1_ratios,
        splits
    )
    fig = plot_results(by_scaling, by_components)

    return fig


def main(parser):
    cs = parser.cs
    l1_ratios = parser.l1_ratios
    splits = parser.splits

    by_scaling, by_components = run_scaling_analyses(
        cs, 
        l1_ratios,
        splits
    )

    fig = plot_results(by_scaling, by_components)
    fig.savefig('figure_4.png')


def _read_args():
    """
    Reads command line arguments--we use these to specify pickle files for
    classification

    Parameters:
        None

    Returns:
        argparse.ArgumentParser with command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Specify files for classification hyper-parameter optimization'
    )
    parser.add_argument(
        '-c',
        '--Cs',
        dest='cs',
        default=100,
        type=int,
        help='Maximum C values (regularization coefficient) to test;'\
            'logarithmically spaced between 1E-4 and 1E4 (default: 100)'
    )
    parser.add_argument(
        '-l',
        '--l1_ratios',
        dest='l1_ratios',
        default=11,
        type=int,
        help='L1-ratios to test; linearly spaced between 0 and 1' \
            '(default: 11)'    
    )
    parser.add_argument(
        '-s',
        '--splits',
        dest='splits',
        default=10,
        type=int,
        help='CV folds to use (default: 10)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    parser = _read_args()
    main(parser)
