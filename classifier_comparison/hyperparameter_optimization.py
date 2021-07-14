"""
Runs hyperparameter optimization for each of the classifiers specified in the
CLASSIFIERS dict below. This code can be called via the command line as
follows:

python hyperparamter_optimization.py -d [path/to/data.pkl] -l [path/to/label.pkl] -m [number of evals]
"""
import argparse
from functools import partial
import pickle
import os
import sys
import warnings

from hyperopt.base import STATUS_OK

sys.path.append(os.path.join(sys.path[0], '..', 'tfac'))

from hyperopt import fmin, hp, tpe, Trials
from hyperopt.pyll import scope
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from classifier_feat_comparison import run_cv, run_sequential, run_exhaustive
from dataImport import form_missing_tensor
from tensor import perform_CMTF

warnings.filterwarnings('ignore', category=ConvergenceWarning)

NOT_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279]


CLASSIFIERS = {
    'Logistic Regression': {
        'model': LogisticRegression,
        'file': 'lr_hp_trials_no2_unknown.pkl',
        'space': {
            'C': hp.uniform('C', 1E-4, 1E4),
            'tfac_components': scope.int(hp.quniform('tfac_components', 2, 10, 1)),
        },
    },
    'SVC_rbf': {
        'model': SVC,
        'file': 'SVC_rbf_hp_trials_no2_unknown.pkl',
        'space': {
            'C': hp.uniform('C', 1E-4, 1E4),
            'gamma': hp.lognormal('gamma', 0.5, 0.25),
            'kernel': 'rbf',
            'tfac_components': scope.int(hp.quniform('tfac_components', 2, 10, 1))
        },
    },
    'Gaussian Naive Bayes': {
        'model': GaussianNB,
        'file': 'gnb_hp_trials_no2_unknown.pkl',
        'space': {
            'var_smoothing': hp.uniform('var_smoothing', 1E-12, 1E-6),
            'tfac_components': scope.int(hp.quniform('tfac_components', 2, 10, 1))
        },
    },
    # 'Random Forest': {
    #     'model': RandomForestClassifier,
    #     'file': 'rf_hp_trials_no2.pkl',
    #     'space': {
    #         'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
    #         'tfac_components': scope.int(hp.quniform('tfac_components', 1, 10, 1))
    #     },
    # },
    # 'XGB': {
    #     'model': XGBClassifier,
    #     'file': 'xgb_hp_trials_no2.pkl',
    #     'space': {
    #         'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
    #         # 'reg_alpha': hp.uniform('reg_alpha', 0, 0.5),
    #         # 'reg_lambda': hp.uniform('reg_lambda', 0, 0.5),
    #         'tfac_components': scope.int(hp.quniform('tfac_components', 1, 10, 1))
    #     },
    # },
}


def obj_func(space, clf_name, tensor, matrix, labels, selector):
    """
    Defines objective function for Hyperopt to minimize.

    Parameters:
        space (dict): dict mapping hyperparameter names to search spaces
        clf_name (str): name of classifier
        tensor (numpy.array): tfac tensor
        matrix (numpy.array): tfac matrix
        labels (pandas.Series): labels for data samples
        selector (str): feature selection method

    Returns:
        Average loss across folds, given as (1 - average accuracy)
    """
    tfac_components = space.pop('tfac_components')

    if 'n_components' in space and space['n_components'] > tfac_components:
        space['n_components'] = tfac_components

    # Hyperopt will sometimes suggest impossible parameters, so we return 1
    # to encourage it in the opposite direction
    for key in space.keys():
        if (not isinstance(space[key], str)) and space[key] < 1E-8:
            return 1

    # Initialize model
    clf = CLASSIFIERS[clf_name]['model'](**space)

    # Run factorization and remove unknowns
    factorized = perform_CMTF(tensor, matrix, tfac_components)
    data = pd.DataFrame(factorized[1][0], index=labels.index)

    labels = labels.loc[labels != 'Unknown'].astype(int)
    data = data.loc[labels.index, :]

    # Run cross-validation (and feature selection is specified)
    if selector is None:
        auc_score, sample_preds = run_cv(clf, data, labels, n_splits=30)
    elif selector == 'sequential':
        auc_score, feats, sample_preds = run_sequential(
            clf, 
            data, 
            labels, 
            min(3, tfac_components - 1), 
            n_splits=30
        )
    else:
        auc_score, feats, sample_preds = run_exhaustive(clf, data, labels, n_splits=30)

    if selector is None:
        return {'loss': 1 - auc_score, 'pred': sample_preds, 'status': STATUS_OK, 'model': clf}
    else:
        return {'loss': 1 - auc_score, 'feats': feats, 'pred': sample_preds, 'status': STATUS_OK, 'model': clf}


def main(parser):
    slices, _, _, pat_info = form_missing_tensor()

    # Extract tensors and patient information
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

    max_evals = parser.max_evals
    selector = parser.selector

    # Runs for each classifier
    for clf_name in CLASSIFIERS.keys():
        # Adds selector to file name if one is used
        file_name = os.path.join(os.getcwd(), CLASSIFIERS[clf_name]['file'])
        if selector is not None:
            names = file_name.split('.')
            file_name = names[0] + f'_{selector}.' + names[1]
        
        # Check if trials object already exists--if not, makes a new one
        try:
            trials = pickle.load(open(file_name, 'rb'))
        except FileNotFoundError:
            trials = Trials()

        # Declares objective function to minimize--partial used to pass 
        # constant arguments to objective function
        fmin_objective = partial(
            obj_func, 
            clf_name=clf_name, 
            tensor=tensor, 
            matrix=matrix, 
            labels=labels, 
            selector=selector
            )
        opt_params = fmin(
            fn=fmin_objective, 
            algo=tpe.suggest, 
            space=CLASSIFIERS[clf_name]['space'],
            max_evals=max_evals, 
            trials=trials
            )

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
        '-m',
        '--max_evals',
        dest='max_evals',
        default=100,
        type=int,
        help='Maximum hyperopt evals per classifier',
    )
    parser.add_argument(
        '-s',
        '--selector',
        default=None,
        dest='selector',
        type=str,
        help='Feature selection method to use (default: None); if an ' \
            'argument is passed, must be "exhaustive" or "sequential"'
    )

    parser = parser.parse_args()
    if parser.selector not in ['sequential', 'exhaustive', None]:
        raise argparse.ArgumentError('If provided, --selector argument must' \
            'be one of "exhaustive" or "sequential"')

    return parser


if __name__ == '__main__':
    parser = _read_args()
    main(parser)
