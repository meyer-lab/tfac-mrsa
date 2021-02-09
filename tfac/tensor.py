"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac2
from tensorly.parafac2_tensor import parafac2_to_slice


def R2Xparafac2(tensor_slices, decomposition):
    """Calculate the R2X of parafac2 decomposition"""
    R2XX = np.zeros(len(tensor_slices))
    for idx, tensor_slice in enumerate(tensor_slices):
        reconstruction = parafac2_to_slice(decomposition, idx)
        R2XX[idx] = 1.0 - np.var(reconstruction - tensor_slice) / np.var(tensor_slice)
    return R2XX


def calcR2X(tensorIn, matrixIn, tensorFac, matrixFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    mErr = np.nanvar(tl.cp_to_tensor(matrixFac) - matrixIn)
    return 1.0 - (tErr + mErr) / (np.nanvar(tensorIn) + np.nanvar(matrixIn))


#### Decomposition Methods ###################################################################


def MRSA_decomposition(tensor_slices, components, **kwargs):
    """Perform tensor formation and decomposition for particular variance and component number
    ---------------------------------------------
    Returns
        parafac2tensor object
        tensor_slices list
    """
    return parafac2(tensor_slices, components, **kwargs)


def reorient_factors(tensorFac, matrixFac):
    """ This function ensures that factors are negative on at most one direction. """
    for jj in range(1, len(tensorFac)):
        # Calculate the sign of the current factor in each component
        means = np.sign(np.mean(tensorFac[jj], axis=0))

        # Update both the current and last factor
        tensorFac[0] *= means[np.newaxis, :]
        matrixFac[0] *= means[np.newaxis, :]
        matrixFac[1] *= means[np.newaxis, :]
        tensorFac[jj] *= means[np.newaxis, :]
    return tensorFac, matrixFac


def censored_lstsq(A, B):
    """Solves least squares problem subject to missing data.
    Note: uses a for loop over the columns of B, leading to a
    slower but more numerically stable algorithm
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """
    X = np.empty((A.shape[1], B.shape[1]))
    for i in range(B.shape[1]):
        m = np.isfinite(B[:, i])  # drop rows where mask is zero
        X[:, i] = np.linalg.lstsq(A[m], B[m, i], rcond=None)[0]
    return X.T
