"""
Tensor decomposition methods
"""
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, grad
from jax.config import config
from scipy.optimize import minimize
import tensorly as tl
from tensorly.decomposition._cp import initialize_cp


tl.set_backend('numpy')
config.update("jax_enable_x64", True)


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
    # Flip the subjects to be positive
    subjMeans = np.sign(np.mean(tFac.factors[0], axis=0))
    tFac.factors[0] *= subjMeans[np.newaxis, :]
    tFac.factors[1] *= subjMeans[np.newaxis, :]
    tFac.mFactor *= subjMeans[np.newaxis, :]

    # Flip the receptors to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    tFac.factors[1] *= rMeans[np.newaxis, :]
    tFac.factors[2] *= rMeans[np.newaxis, :]
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


def cp_to_vec(tFac):
    return np.concatenate([tFac.factors[i].flatten() for i in [0, 2]])


def buildTensors(pIn, tensor, matrix, r, cost=False):
    """ Use parameter vector to build kruskal tensors. """
    assert tensor.shape[0] == matrix.shape[0]
    nN = np.cumsum(np.array([tensor.shape[0], tensor.shape[2]]) * r)
    A = jnp.reshape(pIn[:nN[0]], (tensor.shape[0], r))
    C = jnp.reshape(pIn[nN[0]:nN[1]], (tensor.shape[2], r))

    kr = tl.tenalg.khatri_rao([A, C])
    unfold = tl.unfold(tensor, 1)
    selPat = np.all(np.isfinite(unfold), axis=0)
    selPatM = np.all(np.isfinite(matrix), axis=1)

    if cost:
        cost = jnp.sum(jnp.linalg.lstsq(kr[selPat, :], unfold[:, selPat].T, rcond=None)[1])
        cost += jnp.sum(jnp.linalg.lstsq(A[selPatM, :], matrix[selPatM, :], rcond=None)[1])
        return cost

    B = np.linalg.lstsq(kr[selPat, :], unfold[:, selPat].T, rcond=None)[0].T
    tFac = tl.cp_tensor.CPTensor((None, [A, B, C]))
    tFac.mFactor = np.linalg.lstsq(tFac.factors[0][selPatM, :], matrix[selPatM, :], rcond=None)[0].T
    return tFac


def cost(pIn, tOrig, mOrig, r):
    return buildTensors(pIn, tOrig, mOrig, r, cost=True)


def perform_CMTF(tOrig, mOrig, r=10):
    """ Perform CMTF decomposition by direct optimization. """
    tFac = initialize_cp(np.nan_to_num(tOrig), r)
    tFac.factors[2] = np.ones_like(tFac.factors[2])
    x0 = cp_to_vec(tFac)

    gF = value_and_grad(cost, 0)

    def gradF(x):
        return gF(x, tOrig, mOrig, r)

    def hvp(x, v):
        return grad(lambda x: jnp.vdot(gF(x, tOrig, mOrig, r)[1], v))(x)

    tl.set_backend('jax')
    res = minimize(gradF, x0, method="trust-constr", jac=True, hessp=hvp, options={"maxiter": 200})
    tl.set_backend('numpy')

    tFac = buildTensors(res.x, tOrig, mOrig, r)
    tFac.R2X = calcR2X(tFac, tOrig, mOrig)

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)
    return tFac
