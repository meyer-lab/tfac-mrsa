"""
Figure A2: R2X of CMTF vs PCA.
"""
import numpy as np
from matplotlib.ticker import ScalarFormatter
from statsmodels.multivariate.pca import PCA
from tensorpack import perform_CMTF, calcR2X, tensor_degFreedom

from .common import subplotLabel, getSetup
from ..dataImport import form_tensor
from ..impute import flatten_to_mat


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    fig_size = (9, 3)
    layout = {
        'ncols': 3,
        'nrows': 1
    }
    ax, f, _ = getSetup(
        fig_size,
        layout
    )

    comps = np.arange(1, 12)
    CMTFR2X = np.zeros(comps.shape)
    PCAR2X = np.zeros(comps.shape)
    sizeTfac = np.zeros(comps.shape)

    tOrig, mOrig, _ = form_tensor()
    tMat = flatten_to_mat(tOrig, mOrig)

    sizePCA = comps * np.sum(tMat.shape)

    for i, cc in enumerate(comps):
        outt = PCA(tMat, ncomp=cc, missing="fill-em", standardize=False, demean=False, normalize=False)
        recon = outt.scores @ outt.loadings.T
        PCAR2X[i] = calcR2X(recon, mIn=tMat)
        tFac = perform_CMTF(tOrig, mOrig, r=cc)
        CMTFR2X[i] = tFac.R2X
        sizeTfac[i] = tensor_degFreedom(tFac)

    ax[0].scatter(comps, CMTFR2X, s=10)
    ax[0].set_ylabel("CMTF R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0.5, np.amax(comps) + 0.5)

    ax[1].set_xscale("log", base=2)
    ax[1].plot(sizeTfac, 1.0 - CMTFR2X, ".", label="CMTF")
    ax[1].plot(sizePCA, 1.0 - PCAR2X, ".", label="PCA")
    ax[1].set_ylabel("Normalized Unexplained Variance")
    ax[1].set_xlabel("Size of Reduced Data")
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlim(2 ** 8, 2 ** 12)
    ax[1].xaxis.set_major_formatter(ScalarFormatter())
    ax[1].legend()

    # Add subplot labels
    subplotLabel(ax)

    return f
