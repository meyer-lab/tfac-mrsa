"""
This creates Figure 2 - Tucker Decomposition Plots
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from .figure1 import treatmentPlot, timePlot, proteinPlot
from ..Data_Mod import form_tensor
from ..tensor import tucker_decomp, find_R2X_tucker

tensor, treatments, times = form_tensor()
results = tucker_decomp(tensor, (2, 2, 2))
factors = results[1]

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    R2X_figure(ax[0])
    treatmentPlot(ax[1], factors[0], 1, 2, treatments)
    timePlot(ax[2], factors[1])
    proteinPlot(ax[3], factors[2], 1, 2)

    # Add subplot labels
    subplotLabel(ax)
    return f

def R2X_figure(ax):
    '''Create Tucker R2X Figure'''
    R2X = np.zeros(14)
    nComps = range(1, len(R2X))
    for i in nComps:
        output = tucker_decomp(tensor, (i, i, i))
        R2X[i] = find_R2X_tucker(output, tensor)
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("CP Decomposition")
    ax.set_yticks([0, .2, .4, .6, .8, 1])
