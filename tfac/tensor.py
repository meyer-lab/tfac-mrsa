"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_tucker, parafac, tucker, partial_tucker
from tensorly.metrics.regression import variance as tl_var
from tensorly.tenalg import mode_dot

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


def reorient_factors(factors):
    """ Reorient factors based on the sign of the mean so that only the last factor can have negative means. """
    for index in range(len(factors) - 1):
        meann = np.sign(np.mean(factors[index], axis=0))
        assert meann.size == factors[0].shape[1]

        factors[index] *= meann
        factors[index + 1] *= meann

    return factors


#### Decomposition Methods ###################################################################


def cp_decomp(tensor, r):
    """Perform PARAFAC decomposition.
    -----------------------------------------------
    Input
        tensor: 3D data tensor
        r: rank of decomposition
    Returns
        factors: list of factor matrices
    """
    weights, factors = parafac(tensor, r, tol=1.0e-10, n_iter_max=6000, orthogonalise=True, normalize_factors=True, random_state=1)
    factors[2] *= weights[np.newaxis, :]  # Put weighting in designated factor
    weights /= weights
    return weights, factors


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
        output = non_negative_tucker(tensor, rank_list, tol=1.0e-10, n_iter_max=2000, random_state=1)
    else:
        output = tucker(tensor, rank_list, tol=1.0e-10, n_iter_max=2000, random_state=1)
    return output


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
    return partial_tucker(tensor, mode_list, rank, tol=1.0e-10, n_iter_max=2000, random_state=1)

#### For R2X Plots ###########################################################################


def find_R2X_parafac(cp_output, orig):
    """Compute R2X for the tucker decomposition."""
    return R2X(tl.kruskal_to_tensor(cp_output), orig)


def find_R2X_tucker(tucker_output, orig):
    """Compute R2X for the tucker decomposition."""
    return R2X(tl.tucker_to_tensor(tucker_output), orig)


def find_R2X_partialtucker(tucker_output, orig):
    """Compute R2X for the tucker decomposition."""
    return R2X(mode_dot(tucker_output[0], tucker_output[1][0], 2), orig)
