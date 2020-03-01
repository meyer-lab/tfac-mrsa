"""
This creates Figure 2.
"""
from .figureCommon import subplotLabel, getSetup
import seaborn as sns


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7.5, 6), (3, 4))

    ax[0].axis('off')  # blank out first axis for cartoon

    # Add subplot labels
    subplotLabel(ax)
    
    myFigure(ax[1])

    return f


def myFigure(axis):
    array1 = [1, 2, 3, 4, 5]
    array2 = [5, 4, 3, 2, 1]
    sns.scatterplot(array1, array2, ax = axis)