"""
Evaluates classifier performance for various feature set sizes. This code can 
be called via the command line:

python classifier_feat_comparison.py -c [cytokine.pkl] -g [genes.pkl] -p [parafac2.pkl] -l [labels.pkl]

Original pickle files were extracted from figure6.py.
"""
import argparse
import warnings

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

MAX_FEATS = 30
CLASSIFIERS = {
    "Gaussian Naive Bayes": {"model": GaussianNB, "params": {}},
    "Logistic Regression": {"model": LogisticRegression, "params": {"C": 1}},
    "SVC": {"model": SVC, "params": {}},
}


def get_scores(clf, train_data, train_out, test_data, test_out, selector=None, 
               normalize=True):
    """
    Selects most informative features, then trains and tests model and
    returns performance.

    Parameters:
        clf (sklearn classifier): sklearn classification class
        train_data (pandas.DataFrame): training data
        train_out (pandas.Series): training data classes
        test_data (pandas.DataFrame): testing data
        test_out (pandas.Series): testing data classes
        selector (sklearn feature selector): feature selector to pick most
            informative features (optional)
        normalize (boolean): feature normalization via z-score (default: True)  

    Returns:
        auc_score (float): auc-roc score for testing data
        chosen_feats (list): index of feats selected via selector
    """
    chosen_feats = None
    if selector:
        selector.fit(train_data, train_out)

        if hasattr(selector, 'best_idx_'):
            chosen_feats = list(selector.best_idx_)
        else:
            chosen_feats = selector.get_support()

        train_data = train_data.loc[:, chosen_feats]
        test_data = test_data.loc[:, chosen_feats]

    if normalize:
        train_data = scale(train_data)
        test_data = scale(test_data)

    clf.fit(train_data, train_out)
    proba = clf.predict_proba(test_data)
    proba = proba[:, 1]
    auc_score = roc_auc_score(test_out, proba)

    return auc_score, chosen_feats


def run_exhaustive(clf, data, outcomes, n_splits=30, normalize=True):
    """
    Define cross-validation folds, performs exhaustive feature
    selection on training data, and tests against validation fold.

    Parameters:
        clf (sklearn classifier): sklearn classification class
        data (pandas.DataFrame): data prior to split
        outcomes (pandas.Series): data classes
        n_feats (int): number of features to select
        n_splits (int): cross-validation splits to use
        normalize (boolean): feature normalization via z-score (default: True)

    Returns:
        auc_score (float): average auc-roc score across folds
        feat_freq (pandas.Series): average feature weights across
            cross-validation folds
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    feat_freq = pd.Series(index=data.columns, dtype=float, data=0)
    avg_auc = 0

    for train_index, test_index in kf.split(data, outcomes):
        train_data, train_out = data.iloc[train_index], outcomes.iloc[train_index]
        test_data, test_out = data.iloc[test_index], outcomes.iloc[test_index]

        train_out = train_out.values.ravel()
        test_out = test_out.values.ravel()

        efs = EFS(clf, min_features=2, max_features=2, scoring='roc_auc', print_progress=False)
        auc_score, chosen_feats = get_scores(
            clf, train_data, train_out, test_data, test_out, selector=efs, normalize=normalize
        )

        avg_auc += auc_score / n_splits
        feat_freq.loc[chosen_feats] += 1 / n_splits

    return avg_auc, feat_freq


def run_sequential(clf, data, outcomes, n_feats, n_splits=30, normalize=True):
    """
    Define cross-validation folds, performs sequential feature
    selection on training data, and tests against validation fold.

    Parameters:
        clf (sklearn classifier): sklearn classification class
        data (pandas.DataFrame): data prior to split
        outcomes (pandas.Series): data classes
        n_feats (int): number of features to select
        n_splits (int): cross-validation splits to use
        normalize (boolean): feature normalization via z-score (default: True)  

    Returns:
        auc_score (float): average auc-roc score across folds
        feat_freq (pandas.Series): average feature weights across
            cross-validation folds
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    feat_freq = pd.Series(index=data.columns, dtype=float, data=0)
    avg_auc = 0

    for train_index, test_index in kf.split(data, outcomes):
        train_data, train_out = data.iloc[train_index], outcomes.iloc[train_index]
        test_data, test_out = data.iloc[test_index], outcomes.iloc[test_index]

        train_out = train_out.values.ravel()
        test_out = test_out.values.ravel()

        sfs = SequentialFeatureSelector(clf, n_features_to_select=n_feats, direction='backward')
        auc_score, chosen_feats = get_scores(
            clf, train_data, train_out, test_data, test_out, selector=sfs, normalize=normalize
        )

        avg_auc += auc_score / n_splits
        feat_freq.loc[chosen_feats] += 1 / n_splits

    return avg_auc, feat_freq


def run_k_best(clf, data, outcomes, n_feats, n_splits=30, normalize=True):
    """
    Define cross-validation folds, performs k-best feature selection on 
    training data, and tests against validation fold.

    Parameters:
        clf (sklearn classifier): sklearn classification class
        data (pandas.DataFrame): data prior to split
        outcomes (pandas.Series): data classes
        n_feats (int): number of features to select
        n_splits (int): cross-validation splits to use
        normalize (boolean): feature normalization via z-score (default: True)  

    Returns:
        auc_score (float): average auc-roc score across folds
        feat_freq (pandas.Series): average feature weights across
            cross-validation folds
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    feat_freq = pd.Series(index=data.columns, dtype=float, data=0)
    avg_auc = 0

    for train_index, test_index in kf.split(data, outcomes):
        train_data, train_out = data.loc[train_index], outcomes.loc[train_index]
        test_data, test_out = data.loc[test_index], outcomes.loc[test_index]

        k_best = SelectKBest(k=n_feats)
        auc_score, chosen_feats = get_scores(
            clf, train_data, train_out, test_data, test_out, selector=k_best, normalize=normalize
        )

        avg_auc += auc_score / n_splits
        feat_freq.loc[chosen_feats] += 1 / n_splits

    return avg_auc, feat_freq


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
        description="Specify files for MRSA classification"
    )
    parser.add_argument(
        "-c",
        "--cytokines",
        dest="cytokines",
        type=str,
        help="Cytokine pickle file path",
    )
    parser.add_argument(
        "-g", "--genes", dest="genes", type=str, help="Gene pickle file path"
    )
    parser.add_argument(
        "-p", "--parafac", dest="parafac", type=str, help="PARAFAC2 pickle file path"
    )
    parser.add_argument(
        "-l", "--labels", dest="labels", type=str, help="Class labels pickle file path"
    )

    return parser.parse_args()


def main(parser):
    # Reads data
    cytokines = pd.read_pickle(parser.cytokines)
    genes = pd.read_pickle(parser.genes)
    combined = pd.read_pickle(parser.parafac)
    outcomes = pd.read_pickle(parser.labels).T.iloc[0, :]

    # Iterates over each classifier
    for clf_name in tqdm(CLASSIFIERS.keys()):
        # Tracks auc-roc scores
        clf_performance = pd.DataFrame(
            index=["Cytokines", "Genes", "Combined"],
            columns=[f"{i}_features" for i in range(1, MAX_FEATS)],
        )
        # Tracks feature weights
        feat_df = [
            pd.DataFrame(
                index=data.columns,
                columns=[f"{i}_features" for i in range(1, MAX_FEATS)],
            )
            for data in [cytokines, combined, genes]
        ]
        # Instances classifier
        clf = CLASSIFIERS[clf_name]["model"](**CLASSIFIERS[clf_name]["params"])

        # Runs cross-validation for feature set sizes in [1, 30]
        for n_feats in tqdm(range(1, MAX_FEATS + 1)):
            # Runs sequential for cytokines and PARAFAC2, k-best for genes
            cyto_auc, cyto_feats = run_sequential(
                clf, cytokines, outcomes, n_feats
            )
            comb_auc, comb_feats = run_sequential(
                clf, combined, outcomes, n_feats
            )
            gene_auc, gene_feats = run_k_best(
                clf, genes, outcomes, n_feats
            )

            # Records auc-roc scores
            clf_performance.loc[:, f"{n_feats}_features"] = [
                cyto_auc,
                comb_auc,
                gene_auc,
            ]

            # Records feature weights
            feat_df[0].loc[:, f"{n_feats}_features"] = cyto_feats
            feat_df[1].loc[:, f"{n_feats}_features"] = comb_feats
            feat_df[2].loc[:, f"{n_feats}_features"] = gene_feats

        # Saves auc-roc scores and feature weights for each dataset
        file_name = clf_name.replace(" ", "_")
        clf_performance.to_excel(f"{file_name}_performance_v_feats.xlsx")
        feat_df[0].to_excel(f"{file_name}_cyto_importance.xlsx")
        feat_df[1].to_excel(f"{file_name}_combined_importance.xlsx")
        feat_df[2].to_excel(f"{file_name}_gene_importance.xlsx")


if __name__ == "__main__":
    parser = _read_args()
    main(parser)
