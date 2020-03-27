"""
This creates Figure 2.
"""
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..regression import KFoldCV

sns.set(style="white")
sns.set_context('notebook')


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    ax[0].axis('off')  # blank out first axis for cartoon

    # Add subplot labels
    subplotLabel(ax)
    return f


def predVsActual(ax, x, y, reg):
    '''Predicted vs Actual plotting function for regression'''
    _, predicted, actual = KFoldCV(x, y, reg)
    sns.scatterplot(actual, predicted, color='darkslategrey', ax=ax)
    sns.despine()
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual ' + reg)
