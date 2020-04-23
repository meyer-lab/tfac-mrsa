"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, non_negative_tucker, parafac, tucker
from tensorly.metrics.regression import variance as tl_var

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

#### Decomposition Methods ###################################################################


def cp_decomp(tensor, r, nneg=False):
    """Perform PARAFAC decomposition.
    -----------------------------------------------
    Input
        tensor: 3D data tensor
        r: rank of decomposition
    Returns
        output[0]: component weights
        output[1]: list of factor matrices
    """
    if nneg:
        output = non_negative_parafac(tensor, r, tol=1.0e-10, n_iter_max=6000)
    else:
        output = parafac(tensor, r, tol=1.0e-10, n_iter_max=6000)
    return output


def tucker_decomp(tensor, rank_list, nneg=False):
    """Perform Tucker decomposition.
    -----------------------------------------------
    Input:
        tensor: 3D data tensor
        r: rank of decomposition (list of ranks)
    Returns
        output[0]: core tensor
        output[1]: list of factor matrices
    """
    if nneg:
        output = non_negative_tucker(tensor, rank_list, tol=1.0e-10, n_iter_max=2000)
    else:
        output = tucker(tensor, rank_list, tol=1.0e-10, n_iter_max=2000)
    return output

#### For R2X Plots ###########################################################################


def find_R2X_parafac(cp_output, orig):
    """Compute R2X for the tucker decomposition."""
    return R2X(tl.kruskal_to_tensor(cp_output), orig)


def find_R2X_tucker(tucker_output, orig):
    """Compute R2X for the tucker decomposition."""
    return R2X(tl.tucker_to_tensor(tucker_output), orig)
