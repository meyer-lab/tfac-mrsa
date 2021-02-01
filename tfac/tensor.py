"""
Tensor decomposition methods
"""
import numpy as np
from tensorly.decomposition import parafac2
from tensorly.metrics.regression import variance as tl_var
from tensorly.parafac2_tensor import parafac2_to_slice


def R2Xparafac2(tensor_slices, decomposition):
    """Calculate the R2X of parafac2 decomposition"""
    R2XX = np.zeros(len(tensor_slices))
    for idx, tensor_slice in enumerate(tensor_slices):
        reconstruction = parafac2_to_slice(decomposition, idx, validate=False)
        R2XX[idx] = 1.0 - tl_var(reconstruction - tensor_slice) / tl_var(tensor_slice)
    return R2XX

#### Decomposition Methods ###################################################################


def MRSA_decomposition(tensor_slices, components, random_state=None):
    """Perform tensor formation and decomposition for particular variance and component number
    ---------------------------------------------
    Returns
        parafac2tensor object
        tensor_slices list
    """
    parafac2tensor = parafac2(tensor_slices, components, random_state=random_state, verbose=False, n_iter_parafac=10, n_iter_max=500)
    return parafac2tensor
