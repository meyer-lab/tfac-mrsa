"""
This creates Figure 5.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.decomposition import parafac2
import tensorly as tl
from tensorly.parafac2_tensor import parafac2_to_slice
from tensorly.metrics.regression import variance as tl_var
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import form_MRSA_tensor, get_patient_info


tl.set_backend("numpy")


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects

    tensor_slices, cytokines, geneIDs = form_MRSA_tensor()
    components = 4
    parafac2tensor = parafac2(tensor_slices, components, random_state=1)
    cohortID, outcomeID = get_patient_info()

    patient_matrix = parafac2tensor[1][2]

    outcome_bools = []

    for outcome in outcomeID:
        if outcome == 'APMB':
            outcome_bools.append(0)
        else:
            outcome_bools.append(1)

    outcomes = np.asarray(outcome_bools)

    #clf = LogisticRegression(random_state=1).fit(patient_matrix, outcomes)
    #c = clf.score(patient_matrix, outcomes)
    kf = KFold(n_splits=61)
    c = []
    for train, test in kf.split(patient_matrix):
        clf = LogisticRegression(random_state=1).fit(patient_matrix[train], outcomes[train])
        c.append(clf.score(patient_matrix[test], outcomes[test]))
    print(sum(c) / len(c))

    patient_df = pd.DataFrame(patient_matrix)
    patient_df['Outcome'] = outcomeID
    columns = []
    for component in range(1, components + 1):
        columns.append('Component ' + str(component))
    columns.append('Outcome')
    patient_df.columns = columns

    patient_df = pd.melt(patient_df, id_vars=['Outcome'], var_name='Component')

    ax, f = getSetup((8, 8), (1, 1))
    sns.stripplot(data=patient_df, x='Component', y='value', hue='Outcome')

    return f


def R2Xparafac2(tensor_slices, decomposition):
    """Calculate the R2X of parafac2 decomposition"""
    R2X = [0, 0]
    for idx, tensor_slice in enumerate(tensor_slices):
        reconstruction = parafac2_to_slice(decomposition, idx, validate=False)
        R2X[idx] = 1.0 - tl_var(reconstruction - tensor_slice) / tl_var(tensor_slice)
    return R2X
