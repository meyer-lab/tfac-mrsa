import numpy as np
import pandas as pd
import pickle
from os.path import join, dirname
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from .MRSA_dataHelpers import form_MRSA_tensor, get_C1_patient_info, produce_outcome_bools, find_SVC_proba
from .tensor import MRSA_decomposition, R2Xparafac2


path_here = dirname(dirname(__file__))


def pickle_all():
    '''Create and pickle the best predicting decomposition, R2X, GSEA, and cell type deconvolution'''
    tensor_slices, _, _, _ = form_MRSA_tensor('serum')
    components = tensor_slices[0].shape[0]
    AllR2X = []
    parafac2tensors = []
    #Run factorization at each component number up to limit (38 due to 38 cytokines)
    for component in range(1, components + 1):
        parafac2tensor = MRSA_decomposition(tensor_slices, component, random_state=None)
        R2X = np.round(R2Xparafac2(tensor_slices, parafac2tensor), 6)
        parafac2tensors.append(parafac2tensor)
        AllR2X.append(R2X)
    best = Full_SVC(parafac2tensors, components)
    best_decomp = parafac2tensors[best[1] - 1]
    best_comps = best[2]
    gsea = Full_GSEA(best_decomp)


def import_deconv():
    '''Imports and returns cell deconvolution data.'''
    return pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter=",", index_col='sample').sort_index().drop(['gender', 'cell_type'], axis=1)


def Full_SVC(parafac2tensors, components):
    '''Perform cross validated SVC for each decomposition to determine optimal decomposition and component pair.
  
    Parameters: 
    parafac2tensors (list): list of 38 parafac2tensor objects
  
    Returns: 
    best_decomp (parafac2tensor): Decomposition with best status prediction
    comps (tuple): tuple of component pair that provides optimal prediction within SVC'''
    #import information on patient ID, P vs R status
    cohort_ID, status_ID, type_ID = get_C1_patient_info()
    outcomes = produce_outcome_bools(status_ID)
    #For each decomposition, perform loo CV SVC
    pairs = []
    for comp in range(2, components + 1):
        patient_matrix = parafac2tensors[comp - 1][1][2]
        patient_matrix = patient_matrix[:61, :]
        loo = LeaveOneOut()
        bests = []
        choices = []
        for train, test in loo.split(patient_matrix):
            values_comps = []
            #for each component pair, perform SVC and score it
            for i in range(0, comp - 1):
                for j in range(i + 1, comp):
                    double = np.vstack((patient_matrix[train, i], patient_matrix[train, j])).T
                    decisions = find_SVC_proba(double, outcomes[train])        
                    auc = roc_auc_score(outcomes[train], decisions)
                    values_comps.append([i, j, auc])
            #Make it look nice and fit a model using the best pair, then predict
            df_comp = pd.DataFrame(values_comps)
            df_comp.columns = ["First", "Second", "AUC"]
            df_comp = df_comp.sort_values(by=["AUC"], ascending=False)
            best1 = df_comp.iloc[0, 0]
            best2 = df_comp.iloc[0, 1]
            clf = SVC().fit(np.vstack((patient_matrix[train, best1], patient_matrix[train, best2])).T, outcomes[train])
            choices.append(clf.decision_function(np.vstack((patient_matrix[test, best1], patient_matrix[test, best2])).T))
            bests.append(df_comp.iloc[:1, :])
        best_auc = roc_auc_score(outcomes, choices)
        pairs.append([best_auc, comp, bests])
    return sorted(pairs, reverse=True)[0]


def best_comps(bests):
    common = {}
    for best in bests:
        if (best.values[0][0], best.values[0][1]) in common:
            common[(best.values[0][0], best.values[0][1])] += 1
        else:
            common[(best.values[0][0], best.values[0][1])] = 1
    return max(common, key=lambda key: common[key])
