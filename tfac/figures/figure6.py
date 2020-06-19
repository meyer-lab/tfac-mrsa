"""
This creates Figure 6 - ROC Curve.
"""
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import get_patient_info, produce_outcome_bools, find_CV_decisions
from ..tensor import MRSA_decomposition

_, outcomeID = get_patient_info()

true_y = produce_outcome_bools(outcomeID)
variance = 1
components = 38
tensor_slices, parafac2tensor = MRSA_decomposition(variance, components)
patient_matrix = parafac2tensor[1][2]

score_y = find_CV_decisions(patient_matrix, true_y)

fpr, tpr, thresholds = roc_curve(true_y, score_y)
auc = roc_auc_score(true_y, score_y)
print(auc)

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))

    df = pd.DataFrame()
    df['FPR'] = fpr
    df['TPR'] = tpr
    sns.lineplot(data=df, x='FPR', y='TPR', estimator=None, ax=ax[0])
    df = pd.DataFrame()
    df['FPR'] = [0, 1]
    df['TPR'] = [0, 1]
    sns.lineplot(data=df, x='FPR', y='TPR', estimator=None, ax=ax[0])

    # Add subplot labels
    subplotLabel(ax)

    return f
