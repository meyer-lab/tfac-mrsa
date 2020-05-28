"""
This creates Figure 1 - Partial Tucker Decomposition Protein Plots
"""
from .figureCommon import subplotLabel, getSetup



def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 3
    col = 3
    ax, f = getSetup((7, 7), (row, col))

    # Add subplot labels
    subplotLabel(ax)

    return f
