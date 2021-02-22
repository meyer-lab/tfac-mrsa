"""
Tensor decomposition methods
"""
import numpy as np
from scipy.linalg import khatri_rao
from scipy.stats import gmean
import tensorly as tl
from tensorly.random import random_cp


tl.set_backend("numpy")


def calcR2X(tensorIn, matrixIn, tensorFac, matrixFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    mErr = np.nanvar(tl.cp_to_tensor(matrixFac) - matrixIn)
    return 1.0 - (tErr + mErr) / (np.nanvar(tensorIn) + np.nanvar(matrixIn))


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


def censored_lstsq(A, B, uniqueInfo):
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
    unique, uIDX = uniqueInfo

    for i in range(unique.shape[1]):
        uI = (uIDX == i)
        uu = np.squeeze(unique[:, i])

        Bx = B[uu, :]
        X[:, uI] = np.linalg.lstsq(A[uu, :], Bx[:, uI], rcond=None)[0]
    return X.T


def perform_TMTF(tOrig, mOrig, r=10):
    """ Perform TMTF decomposition. """
    tFac = random_cp(np.shape(tOrig), r, random_state=1, normalise_factors=False)
    tFac.factors[2] = np.ones_like(tFac.factors[2])

    # Everything from the original mFac will be overwritten
    mFac = random_cp(np.shape(mOrig), r, random_state=1, normalise_factors=False)

    # Pre-unfold
    selPat = np.all(np.isfinite(mOrig), axis=1)
    unfolded = [tl.unfold(tOrig, i) for i in range(3)]
    unfolded[0] = np.hstack((unfolded[0], mOrig))

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    R2X = -1000.0
    for ii in range(400):
        if ii < 20:
            # Don't let the subjects drift to different scalings
            varr = np.array([np.var(tFac.factors[0][uniqueInfo[0][1] == ii, :], axis=0) for ii in range(uniqueInfo[0][0].shape[1])])
            varr /= gmean(varr, axis=0)[np.newaxis, :]

            for i in range(uniqueInfo[0][0].shape[1]):
                tFac.factors[0][uniqueInfo[0][1] == i, :] /= varr[i][np.newaxis, :]

            tFac.factors[0] = np.linalg.qr(tFac.factors[0])[0]
            mFac.factors[0] = tFac.factors[0]

        # PARAFAC on other antigen modes
        for m in [1, 2]:
            kr = khatri_rao(tFac.factors[0], tFac.factors[3 - m])
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

        # Solve for the glycan matrix fit
        mFac.factors[1] = np.linalg.lstsq(mFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

        # Solve for the subject matrix
        kr = khatri_rao(tFac.factors[1], tFac.factors[2])
        kr2 = np.vstack((kr, mFac.factors[1]))

        tFac.factors[0] = censored_lstsq(kr2, unfolded[0].T, uniqueInfo[0])
        mFac.factors[0] = tFac.factors[0]

        if ii % 2 == 0:
            R2X_last = R2X
            R2X = calcR2X(tOrig, mOrig, tFac, mFac)
            assert np.isfinite(R2X)

        if (ii > 40) and (R2X - R2X_last < 1e-4):
            break

    tFac.normalize()
    mFac.normalize()

    # Reorient the later tensor factors
    tFac.factors, mFac.factors = reorient_factors(tFac.factors, mFac.factors)

    return tFac, mFac, R2X
