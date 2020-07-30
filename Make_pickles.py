import pickle
import numpy as np
import tensorly as tl
from tensorly.parafac2_tensor import apply_parafac2_projections
from tfac.MRSA_dataHelpers import get_patient_info, form_MRSA_tensor
from tfac.tensor import R2Xparafac2

tl.set_backend("numpy")

def make_pickles():
    #Figure 1
    _, statusID = get_patient_info()
    components = 38
    tensor_slices, _, _ = form_MRSA_tensor(1, 1)
    parafac2tensors = pickle.load(open("cyto_exp.p", "rb"))
    AllR2X = []
    for comp in range(components):
        parafac2tensor = parafac2tensors[comp]
        R2X = np.round(R2Xparafac2(tensor_slices, parafac2tensor), 6)
        AllR2X.append(R2X)
    pickle.dump(AllR2X, open("R2X_SVC.p", "wb"))

    #Fig 4
    components = 38
    patient_matrices = pickle.load(open("cyto_exp.p", "rb"))
    _, cytos, _ = form_MRSA_tensor(1, 1)
    patient_mats_applied = apply_parafac2_projections(patient_matrices[components - 1])
    pickle.dump(patient_mats_applied, open("Factors.p", "wb"))

    #Fig 5
    pickle.dump(patient_matrices[components - 1], open("Parafac2tensor.p", "wb"))