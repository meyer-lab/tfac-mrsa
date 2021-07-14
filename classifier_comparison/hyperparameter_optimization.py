"""
Runs hyperparameter optimization for each of the classifiers specified in the
CLASSIFIERS dict below. This code can be called via the command line as
follows:

python hyperparamter_optimization.py -d [path/to/data.pkl] -l [path/to/label.pkl] -m [number of evals]
"""
from dataImport import form_missing_tensor
from classifier_feat_comparison import run_sequential, run_exhaustive
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from hyperopt.pyll import scope
from hyperopt import fmin, hp, tpe, Trials
import argparse
from functools import partial
import pickle
import os
import sys
import warnings

sys.path.append('../tfac')


warnings.filterwarnings('ignore', category=ConvergenceWarning)

CLASSIFIERS = {
    'Logistic Regression': {
        'model': LogisticRegression,
        'file': 'lr_hp_trials_large.pkl',
        'space': {
            'C': hp.uniform('C', 1E-4, 1E4),
            'n_feats': scope.int(hp.quniform('n_feats', 1, 39, 1))
        },
    },
    'SVC_rbf': {
        'model': SVC,
        'file': 'SVC_rbf_hp_trials_large.pkl',
        'space': {
            'C': hp.uniform('C', 1E-4, 1E4),
            'gamma': hp.lognormal('gamma', 0.5, 0.25),
            'kernel': 'rbf',
            'n_feats': scope.int(hp.quniform('n_feats', 1, 39, 1)),
            'probability': True
        },
    },
    'Gaussian Naive Bayes': {
        'model': GaussianNB,
        'file': 'gnb_hp_trials_large.pkl',
        'space': {
            'var_smoothing': hp.uniform('var_smoothing', 1E-4, 1E4),
            'n_feats': scope.int(hp.quniform('n_feats', 1, 39, 1))
        },
    },
}


def obj_func(space, clf_name, data, labels, exhaustive):
    """
    Defines objective function for Hyperopt to minimize.

    Parameters:
        space (dict): dict mapping hyperparameter names to search spaces
        clf_name (str): name of classifier
        data (pandas.DataFrame): data for classification
        labels (pandas.Series): labels for data samples
        exhaustive (bool): whether to run exhaustive feature selection

    Returns:
        Average loss across folds, given as (1 - average accuracy)
    """
    n_feats = space.pop('n_feats')

    for key in space.keys():
        if isinstance(space[key], float):
            space[key] = max(1E-8, space[key])

        if isinstance(space[key], int):
            space[key] = max(0, space[key])

    clf = CLASSIFIERS[clf_name]['model'](**space)

    if exhaustive:
        auc_score, _ = run_exhaustive(clf, data, labels, n_splits=30)
    else:
        auc_score, _ = run_sequential(clf, data, labels, n_feats, n_splits=30)

    return 1 - auc_score


def main(parser):
    # Reads parser arguments
    exhaustive = parser.exhaustive
    max_evals = parser.max_evals

    # Imports data
    slices, _, _, pat_info = form_missing_tensor()
    pat_info = pat_info.T.reset_index()

    tensor = np.stack(
            (slices[0], slices[1])
        ).T
    matrix = slices[2].T
    matrix = matrix[pat_info.index, :]
    tensor = tensor[pat_info.index, :, :]
    pat_info = pat_info.reset_index(drop=True)

    if parser.drop:
        cohorts = pat_info.loc[:, 'cohort']
        cohorts = cohorts.loc[cohorts != 2]

        tensor = np.stack(
            (slices[0][:, cohorts.index], slices[1][:, cohorts.index])
        ).T
        matrix = slices[2][:, cohorts.index].T
        pat_info = pat_info.loc[cohorts.index]

    pat_info = pat_info.reset_index(drop=True)
    labels = pat_info.loc[:, 'status']

    # Runs for each classifier
    for clf_name in CLASSIFIERS.keys():
        file_name = os.path.join(os.getcwd(), CLASSIFIERS[clf_name]['file'])
        if exhaustive:
            names = file_name.split('.')
            file_name = names[0] + '_exhaustive.' + names[1]

        # Check if trials object already exists--if not, makes a new one
        try:
            trials = pickle.load(open(file_name, 'rb'))
        except FileNotFoundError:
            trials = Trials()

        # Declares objective function to minimize--partial used to pass constant arguments
        # to objective function
        fmin_objective = partial(obj_func, clf_name=clf_name, data=data, labels=labels, exhaustive=exhaustive)
        opt_params = fmin(fn=fmin_objective, algo=tpe.suggest, space=CLASSIFIERS[clf_name]['space'],
                          max_evals=max_evals, trials=trials)

        # Dumps trial object to file
        with open(file_name, 'wb') as handle:
            pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        '-d',
        '--drop',
        dest='drop',
        action='store_true',
        help='Pass to remove cohort 2 from analyses',
    )
    parser.add_argument(
        '-e',
        '--exhaustive',
        dest='exhaustive',
        action='store_true',
        help='Enables exhaustive feature search',
    )
    parser.add_argument(
        '-m',
        '--max_evals',
        dest='max_evals',
        type=int,
        help='Maximum hyperopt evals per classifier',
    )

    return parser.parse_args()


if __name__ == '__main__':
    parser = _read_args()
    main(parser)
