"""
Tensor decomposition methods
"""

import numpy as np
from scipy.sparse.linalg import svds
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.decomposition._cp import initialize_cp


tl.set_backend('numpy')


def buildMat(tFac):
    """ Build the glycan matrix from the factors. """
    return tFac.factors[0] @ tFac.mFactor.T


def calcR2X(tFac, tIn=None, mIn=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None) or (mIn is not None)

    vTop = 0.0
    vBottom = 0.0

    if tIn is not None:
        tMask = np.isfinite(tIn)
        vTop += np.sum(np.square(tl.cp_to_tensor(tFac) * tMask - np.nan_to_num(tIn)))
        vBottom += np.sum(np.square(np.nan_to_num(tIn)))
    if mIn is not None:
        mMask = np.isfinite(mIn)
        vTop += np.sum(np.square(buildMat(tFac) * mMask - np.nan_to_num(mIn)))
        vBottom += np.sum(np.square(np.nan_to_num(mIn)))

    return 1.0 - vTop / vBottom


def reorient_factors(tFac):
    """ This function ensures that factors are negative on at most one direction. """
    # Flip the types to be positive
    tMeans = np.sign(np.mean(tFac.factors[2], axis=0))
    tFac.factors[1] *= tMeans[np.newaxis, :]
    tFac.factors[2] *= tMeans[np.newaxis, :]

    # Flip the cytokines to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    tFac.factors[0] *= rMeans[np.newaxis, :]
    tFac.factors[1] *= rMeans[np.newaxis, :]
    tFac.mFactor *= rMeans[np.newaxis, :]
    return tFac


def cp_normalize(tFac):
    """ Normalize the factors using the inf norm. """
    for i, factor in enumerate(tFac.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        tFac.weights *= scales
        if i == 0:
            tFac.mFactor *= scales

        tFac.factors[i] /= scales

    return tFac


def censored_lstsq(A: np.ndarray, B: np.ndarray, uniqueInfo) -> np.ndarray:
    """Solves least squares problem subject to missing data.
    Note: uses a for loop over the missing patterns of B, leading to a
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
    # Missingness patterns
    unique, uIDX = uniqueInfo

    for i in range(unique.shape[1]):
        uI = uIDX == i
        uu = np.squeeze(unique[:, i])

        Bx = B[uu, :]
        X[:, uI] = np.linalg.lstsq(A[uu, :], Bx[:, uI], rcond=None)[0]
    return X.T


def initialize_cp(tensor: np.ndarray, matrix: np.ndarray, rank: int):
    r"""Initialize factors used in `parafac`.
    Parameters
    ----------
    tensor : ndarray
    rank : int
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    factors = [np.ones((tensor.shape[i], rank)) for i in range(tensor.ndim)]

    # SVD init mode 1
    unfold = tl.unfold(tensor, 1)

    # Remove completely missing columns
    unfold = unfold[:, np.all(np.isfinite(unfold), axis=0)]
    U, S, _ = np.linalg.svd(unfold)
    U = U @ np.diag(S)
    factors[1] = U[:, :rank]

    cp_init = tl.cp_tensor.CPTensor((None, factors))

    # Solve for the mFactor
    cp_init.mFactor, S, _ = svds(matrix[np.all(np.isfinite(matrix), axis=1), :].T, k=rank)
    cp_init.mFactor = cp_init.mFactor @ np.diag(S)

    return cp_init


def perform_CMTF(tOrig, mOrig, r=2):
    """ Perform CMTF decomposition. """
    tFac = initialize_cp(tOrig, mOrig, r)

    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]

    uniqueInfoM = np.unique(np.isfinite(mOrig), axis=1, return_inverse=True)
    unfolded[0] = np.hstack((unfolded[0], mOrig))

    R2X_last = -np.inf
    tFac.R2X = calcR2X(tFac, tOrig, mOrig)

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    for ii in range(2000):
        # PARAFAC on all modes
        for m in range(len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            if m == 0:
                kr = np.vstack((kr, tFac.mFactor))

            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

        # Solve for the glycan matrix fit
        tFac.mFactor = censored_lstsq(tFac.factors[0], mOrig, uniqueInfoM)

        if ii % 2 == 0:
            R2X_last = tFac.R2X
            tFac.R2X = calcR2X(tFac, tOrig, mOrig)
            assert tFac.R2X > 0.0

        if tFac.R2X - R2X_last < 1e-4:
            break

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)

    print(tFac.R2X)

    return tFac
