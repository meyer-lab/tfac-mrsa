"""
This creates Figure 3.
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import find_R2X_partialtucker, partial_tucker_decomp
from ..Data_Mod import form_tensor


tensor, treatments, times = form_tensor()


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((14, 14), (1, 1))

    R2X_Figure_PartialTucker(ax[0], tensor)
    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_Figure_PartialTucker(ax, input_tensor):
    '''Create Partial Tucker R2X Figure'''
    R2X = np.zeros(13)
    for i in range(1, 13):
        output = partial_tucker_decomp(input_tensor, [2], i)
        R2X[i] = find_R2X_partialtucker(output, input_tensor)
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("Partial Tucker Decomposition")
    ax.set_yticks([0, .2, .4, .6, .8, 1])
