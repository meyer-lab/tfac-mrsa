"""
Coupled Matrix Tensor Factorization
"""

import os
from copy import deepcopy
import numpy as np
import tensorly as tl
from tensorly.tenalg.svd import randomized_svd
from tensorly.tenalg.core_tenalg import khatri_rao
from statsmodels.multivariate.pca import PCA
from tqdm import tqdm
from tensorpack.cmtf import (
    cp_normalize,
    reorient_factors,
    sort_factors,
    mlstsq,
    calcR2X,
)

tl.set_backend("numpy")


class PCArand(PCA):
    def _compute_eig(self):
        """
        Override slower SVD methods
        """
        _, s, v = randomized_svd(self.transformed_data, self._ncomp)

        self.eigenvals = s ** 2.0
        self.eigenvecs = v.T


def perform_CMTF(tOrig, mOrig, r=8, tol=1e-6, maxiter=300, progress=None, linesearch: bool=True):
    """Perform CMTF decomposition."""
    assert tOrig.dtype == float
    assert mOrig.dtype == float
    factors = [np.ones((tOrig.shape[i], r)) for i in range(tOrig.ndim)]

    # Check if verbose was not set
    if progress is None:
        # Check if this is an automated build
        progress = "CI" not in os.environ

    acc_pow: float = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail: int = 0  # How many times acceleration have failed
    max_fail: int = 4  # Increase acc_pow with one after max_fail failure

    # SVD init mode 0
    unfold = np.hstack((tl.unfold(tOrig, 0), mOrig))
    pca = PCArand(unfold, ncomp=r, missing='fill-em')
    factors[0] = pca.factors
    tFac = tl.cp_tensor.CPTensor((None, factors))

    # Pre-unfold
    unfolded = np.hstack((tl.unfold(tOrig, 0), mOrig))
    missingM = np.all(np.isfinite(mOrig), axis=1)
    assert np.sum(missingM) >= 1, "mOrig must contain at least one complete row"
    R2X = -np.inf

    # Precalculate the missingness patterns
    uniqueInfo = np.unique(np.isfinite(unfolded.T), axis=1, return_inverse=True)

    tq = tqdm(range(maxiter), disable=(not progress))
    for iter in tq:
        tFac_old = deepcopy(tFac)

        for m in [1, 2]:
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = mlstsq(kr, tl.unfold(tOrig, m).T).T

        # Solve for the mRNA factors
        tFac.mFactor = np.linalg.lstsq(
            tFac.factors[0][missingM, :], mOrig[missingM, :], rcond=None
        )[0].T

        # Solve for subjects factors
        kr = khatri_rao(tFac.factors, skip_matrix=0)
        kr = np.vstack((kr, tFac.mFactor))
        tFac.factors[0] = mlstsq(kr, unfolded.T, uniqueInfo).T

        R2X_last = R2X
        R2X = calcR2X(tFac, tOrig, mOrig)

        # Initiate line search
        if linesearch and iter % 2 == 0 and iter > 3:
            jump = iter ** (1.0 / acc_pow)

            # Estimate error with line search
            tFac_ls = deepcopy(tFac)

            tFac_ls.factors = [
                tFac_old.factors[ii] + (f - tFac_old.factors[ii]) * jump
                for ii, f in enumerate(tFac.factors)
            ]
            tFac_ls.mFactor = tFac_old.mFactor + (tFac.mFactor - tFac_old.mFactor)

            R2X_ls = calcR2X(tFac_ls, tOrig, mOrig)

            if R2X_ls > R2X:
                acc_fail = 0
                R2X = R2X_ls
                tFac = tFac_ls

                if progress:
                    print(f"Accepted line search jump of {jump}.")
            else:
                acc_fail += 1

                if progress:
                    print(f"Line search failed for jump of {jump}.")

                if acc_fail == max_fail:
                    acc_pow += 1.0
                    acc_fail = 0

                    if progress:
                        print("Reducing acceleration.")


        tq.set_postfix(R2X=R2X, delta=R2X - R2X_last, refresh=False)
        assert R2X > 0.0

        if R2X - R2X_last < tol:
            break

    assert not np.all(tFac.mFactor == 0.0)
    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)
    tFac = sort_factors(tFac)
    tFac.R2X = R2X

    return tFac, pca
