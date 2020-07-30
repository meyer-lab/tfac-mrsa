import pickle
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.parafac2_tensor import apply_parafac2_projections
from sklearn.metrics import roc_auc_score
from tfac.MRSA_dataHelpers import produce_outcome_bools, get_patient_info, form_MRSA_tensor, find_regularization, find_CV_proba
from tfac.tensor import R2Xparafac2

tl.set_backend("numpy")

def make_pickles():
    #Figure 1
    _, statusID = get_patient_info()
    outcomes = produce_outcome_bools(statusID)
    components = 38
    tensor_slices, _, _ = form_MRSA_tensor(1, 1)
    parafac2tensors = pickle.load(open("cyto_exp.p", "rb"))
    AllR2X = []
    for comp in range(components):
        parafac2tensor = parafac2tensors[comp]
        R2X = np.round(R2Xparafac2(tensor_slices, parafac2tensor), 6)
        AllR2X.append(R2X)
    pickle.dump(AllR2X, open("R2X_SVC.p", "wb"))
    
    #Fig 3
    components = 38
    patient_matrices = pickle.load(open("cyto_exp.p", "rb"))
    patient_matrices2 = pickle.load(open("cyto_exp_2.p", "rb"))
    patient_matrices3 = pickle.load(open("cyto_exp_3.p", "rb"))
    all_mats = [patient_matrices, patient_matrices2, patient_matrices3]
    values_comps = []
    for components in range(1, components + 1):
        for patient_matrices in all_mats:
            patient_matrix = patient_matrices[components - 1][1][2]
            reg = find_regularization(patient_matrix, outcomes)
            proba = find_CV_proba(patient_matrix, outcomes, random_state=None, C=reg)        
            auc = roc_auc_score(outcomes, proba)
            values_comps.append([components, auc])
    df_comp = pd.DataFrame(values_comps)
    df_comp.columns = ['Components', 'AUC']
    pickle.dump(df_comp, open("LogReg.p", "wb"))

    #Fig 4
    components = 38
    _, cytos, _ = form_MRSA_tensor(1, 1)
    patient_mats_applied = apply_parafac2_projections(patient_matrices[components - 1])
    pickle.dump(patient_mats_applied, open("Factors.p", "wb"))

    #Fig 5
    pickle.dump(patient_matrices[components - 1], open("Parafac2tensor.p", "wb"))