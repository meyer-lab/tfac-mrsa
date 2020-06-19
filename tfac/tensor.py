"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.metrics.regression import variance as tl_var
from tensorly.decomposition import parafac2
from tensorly.parafac2_tensor import parafac2_to_slice
from tensorly.tenalg import mode_dot
from .MRSA_dataHelpers import form_MRSA_tensor


tl.set_backend("numpy")  # Set the backend


def z_score_values(A, cell_dim):
    """ Function that takes in the values tensor and z-scores it. """
    assert cell_dim < tl.ndim(A)
    convAxes = tuple([i for i in range(tl.ndim(A)) if i != cell_dim])
    convIDX = [None] * tl.ndim(A)
    convIDX[cell_dim] = slice(None)

    sigma = tl.tensor(np.std(tl.to_numpy(A), axis=convAxes))
    return A / sigma[tuple(convIDX)]


def R2X(reconstructed, original):
    """ Calculates R2X of two tensors. """
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)


def R2Xparafac2(tensor_slices, decomposition):
    """Calculate the R2X of parafac2 decomposition"""
    R2X = [0, 0]
    for idx, tensor_slice in enumerate(tensor_slices):
        reconstruction = parafac2_to_slice(decomposition, idx, validate=False)
        R2X[idx] = 1.0 - tl_var(reconstruction - tensor_slice) / tl_var(tensor_slice)
    return R2X


def reorient_factors(factors):
    """ Reorient factors based on the sign of the mean so that only the last factor can have negative means. """
    for index in range(len(factors) - 1):
        meann = np.sign(np.mean(factors[index], axis=0))
        assert meann.size == factors[0].shape[1]

        factors[index] *= meann
        factors[index + 1] *= meann

    return factors


#### Decomposition Methods ###################################################################


def partial_tucker_decomp(tensor, mode_list, rank):
    """Perform Partial Tucker decomposition.
    -----------------------------------------------
    Input:
        tensor: 3D data tensor
        mode_list: which mode(s) to apply tucker decomposition to
        rank: rank of decomposition
    Returns
        output[0]: core tensor
        output[1]: list of factor matrices
    """
    return partial_tucker(tensor, mode_list, rank, tol=1.0e-12)

def MRSA_decomposition(variance, components):
    '''Perform tensor formation and decomposition for particular variance and component number
    ---------------------------------------------
    Returns 
        parafac2tensor object
        tensor_slices list
    '''
    tensor_slices, _, _ = form_MRSA_tensor(variance)
    parafac2tensor = None
    best_error = np.inf
    for _ in range(1):
        decomposition, errors = parafac2(tensor_slices, components, return_errors=True, tol=1e-7, n_iter_max=1000, random_state=1)
        if best_error > errors[-1]:
            best_error = errors[-1]
            parafac2tensor = decomposition
    return tensor_slices, parafac2tensor


#### For R2X Plots ###########################################################################


def find_R2X_partialtucker(tucker_output, orig):
    """Compute R2X for the tucker decomposition."""
    return R2X(mode_dot(tucker_output[0], tucker_output[1][0], 2), orig)
