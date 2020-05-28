"""
This creates Figure 1 - Partial Tucker Decomposition Protein Plots
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import partial_tucker_decomp
from ..Data_Mod import form_tensor

components = 7
tensor, treatments, times = form_tensor()
results = partial_tucker_decomp(tensor, [2], components)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 3
    col = 7
    ax, f = getSetup((21, 7), (row, col))

    proteinScatterPlot(ax, results, components)

    # Add subplot labels
    subplotLabel(ax)

    return f


def proteinScatterPlot(ax, results, components):
    '''Plot compared proteins (tensor axis 2) in factorization component space'''
    counter = 0
    for i in range(components):
        for j in range(i + 1, components):
            sns.scatterplot(results[1][0][:, i], results[1][0][:, j], ax=ax[counter])
            ax[counter].set_xlabel('Component ' + str(i + 1))
            ax[counter].set_ylabel('Component ' + str(j + 1))
            ax[counter].set_title('Protein Factors')
            counter += 1
    for _ in range(counter, len(ax)):
        ax[counter].axis('off')
        counter += 1


def setPlotLimits(axis, factors, r1, r2):
    '''Set appropriate limits for the borders of each component plot'''
    x = np.absolute(factors[:, r1 - 1])
    y = np.absolute(factors[:, r2 - 1])
    xlim = 1.1 * np.max(x)
    ylim = 1.1 * np.max(y)
    axis.set_xlim((-xlim, xlim))
    axis.set_ylim((-ylim, ylim))
