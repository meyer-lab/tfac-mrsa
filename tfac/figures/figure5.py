"""
This creates Figure 5.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7.5, 6), (3, 4))

    ax[0].axis('off')  # blank out first axis for cartoon

    # Add subplot labels
    subplotLabel(ax)

    return f