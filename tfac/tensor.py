"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac2
from tensorly.metrics.regression import variance as tl_var
from tensorly.parafac2_tensor import parafac2_to_slice


tl.set_backend("numpy")  # Set the backend


def z_score_values(A, cell_dim):
    """ Function that takes in the values tensor and z-scores it. """
    assert cell_dim < tl.ndim(A)
    convAxes = tuple([i for i in range(tl.ndim(A)) if i != cell_dim])
    convIDX = [None] * tl.ndim(A)
    convIDX[cell_dim] = slice(None)

    sigma = tl.tensor(np.std(tl.to_numpy(A), axis=convAxes))
    return A / sigma[tuple(convIDX)]

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
    parafac2tensor = parafac2(tensor_slices, components, random_state=random_state, verbose=False)
    return parafac2tensor
