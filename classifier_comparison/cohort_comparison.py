import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
from tqdm import tqdm

sys.path.append(os.path.join(sys.path[0], '..', 'tfac'))

from dataImport import full_import, form_missing_tensor

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main(parser):
    if parser.rna:
        _, _, data, _ = full_import()
        data = data.T
    else:
        data, _, _, _ = full_import()
        data = pd.concat(data, axis=1)
        data = data.T
        data = data.loc[~data.index.duplicated(keep='first'), :]

    # Get patient outcomes
    _, _, _, outcomes = form_missing_tensor()
    outcomes = outcomes.T
    outcomes = outcomes.loc[outcomes['status'] != 'Unknown']
    
    # Remove samples with unknown outcomes
    samples = list(set(outcomes.index) & set(data.index))
    outcomes = outcomes.loc[samples, :]
    data = data.loc[samples, :]

    if not parser.rna:
        # Drops cytokine samples without Serum data
        outcomes = outcomes.loc[outcomes['type'].str.contains('0Serum')]
        data = data.loc[outcomes.index, :]
        outcomes = outcomes.drop('type', axis=1)

    # Tracks auc-roc scores
    clf_performance = pd.DataFrame(
        data=None,
        index=['Cohort_1', 'Cohort_2', 'Cohort_3'],
        columns=['Accuracy', 'C']
    )
    # Tracks feature weights
    feat_importances = pd.DataFrame(
        index=['Cohort_1', 'Cohort_2', 'Cohort_3'],
        columns=data.columns,
    )
    # Tracks performance of models applied across cohorts
    cross_performance = pd.DataFrame(
        data=None,
        index=['Cohort_1', 'Cohort_2', 'Cohort_3'],
        columns=['Cohort_1', 'Cohort_2', 'Cohort_3']
    )

    # Log-Reg Model
    clf = LogisticRegressionCV(Cs=parser.c, cv=parser.splits)

    # Trains model on each cohort
    for cohort in tqdm(range(1, 4)):
        if cohort == 2 and parser.rna:
            continue

        train_data, train_labels = get_data_labels(data, outcomes, cohort)
        clf.fit(train_data, train_labels)
        feats = clf.coef_[0]

        scores = [0] * parser.c
        for score in clf.scores_[1]:
            scores += score / parser.splits

        best = np.argmax(scores)
        cohort = f'Cohort_{cohort}'
        feat_importances.loc[cohort, :] = feats
        clf_performance.loc[cohort, :] = [scores[best], clf.Cs_[best]]

        # Applies trained model to every other cohort
        for comp in range(1, 4):
            if comp == 2 and parser.rna:
                continue 

            test_data, test_labels = get_data_labels(data, outcomes, comp)
            score = clf.score(test_data, test_labels)
            comp = f'Cohort_{comp}'
            
            # If test and train cohorts are the same, instead returns
            # cross-validation result
            if comp == cohort:
                cross_performance.loc[cohort, comp] = scores[best]
            else:
                cross_performance.loc[cohort, comp] = score

    cross_performance.to_csv('cross_cohort_performance.csv')
    feat_importances.to_csv('cohort_feat_weights.csv')
    clf_performance.to_csv('cohort_cv_accuracy.csv')


def get_data_labels(data, outcomes, cohort):
    """
    Trims data and labels to only include samples from specified cohort, then
    Z-scores data.

    Parameters:
        data (pandas.DataFrame): patient data
        outcomes (pandas.Series): patient outcomes
        cohort (int): cohort to select

    Returns:
        data (pandas.DataFrame): normalized training data for cohort
        labels (pandas.Series): patient outcomes for cohort
    """
    labels = outcomes.loc[outcomes['cohort'] == cohort]
    labels = labels.loc[:, 'status']
    labels = labels.astype(int)

    cohort_data = data.loc[labels.index, :]
    cohort_data[:] = scale(cohort_data, axis=0)
    cohort_data[:] = scale(cohort_data, axis=1)

    return cohort_data, labels


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
        '-r',
        '--rna',
        dest='rna',
        action='store_true',
        help='Pass to use RNA data; uses cytokine data otherwise',
    )
    parser.add_argument(
        '-s',
        '--splits', 
        dest='splits',
        type=int,
        default=10,
        help='Number of cross-validation folds to use',
    )
    parser.add_argument(
        '-c',
        dest='c',
        type=int, 
        default=100,
        help='Number of Cs (regularization constants) to test',
    )

    return parser.parse_args()


if __name__ == "__main__":
    parser = _read_args()
    main(parser)
