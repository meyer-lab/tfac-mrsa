"""
This creates Figure 1.
"""
import numpy as np
import matplotlib as plt
import tensorly as tl
from .figureCommon import subplotLabel, getSetup
from ..tensor import calc_R2X_parafac

### Data Tensor Variable Gets Declared Here
tensor = tl.random.random_kruskal((50, 50, 50), rank=20, full=True)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7.5, 6), (3, 4))

    ax[0].axis('off')  # blank out first axis for cartoon
    R2X_figure(ax[1], tensor)

    # Add subplot labels
    subplotLabel(ax)

    return f

def R2X_figure(ax, tens):
    '''Create R2X Figure'''
    x_axis = np.arange(9)
    R2X = np.zeros(9)
    for i in range (1, 9):
        R2X[i] = calc_R2X_parafac(tens, i)
    ax.scatter(x_axis, R2X)
    ax.set_xlabel('Decomposition Rank')
    ax.set_ylabel('R2X')
    ax.set_title('PARAFAC')
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xticks(x_axis)
