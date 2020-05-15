"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker
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
        output[0]: reconstruction variance explained
        output[1]: list of factor matrices
    """
    weights, factors = parafac(tensor, r, tol=1.0e-15, n_iter_max=2000, orthogonalise=10, random_state=1, normalize_factors=True)
    factors[2] *= weights[np.newaxis, :]  # Put weighting in designated factor
    weights /= weights

    factors = reorient_factors(factors)

    recon2X = R2X(tl.kruskal_to_tensor((weights, factors)), tensor)
    return recon2X, factors


def tucker_decomp(tensor, rank_list):
    """Perform Tucker decomposition.
    -----------------------------------------------
    Input:
        tensor: 3D data tensor
        r: rank of decomposition (list of ranks)
    Returns
        output[0]: reconstruction variance explained
        output[1]: list of factor matrices
    """
    tucked = tucker(tensor, rank_list, tol=1.0e-15)
    recon2X = R2X(tl.tucker_to_tensor(tucked), tensor)

    return recon2X, tucked
