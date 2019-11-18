"""
This creates Figure 1.
"""
import numpy as np
import tensorly as tl
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import calc_R2X_parafac, perform_parafac

### Data Tensor Variable Gets Declared Here
tensor = tl.random.random_kruskal((50, 50, 50), rank=20, full=True)
global_factors = perform_parafac(tensor, 4)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))

    ax[0].axis('off')  # blank out first axis for cartoon
    ax[1].axis('off')

    ## R2X Plot
    R2X_figure(ax[2], tensor)

    ## Cell Line Factor Plots
    cellLinePlot(ax[3], global_factors[0], 1, 2)
    cellLinePlot(ax[6], global_factors[0], 3, 4)

    ## Gene Factor Plots
    genePlot(ax[4], global_factors[1], 1, 2)
    genePlot(ax[7], global_factors[1], 3, 4)

    ## Genetic Characteristic Plots
    characPlot(ax[5], global_factors[2], 1, 2)
    characPlot(ax[8], global_factors[2], 3, 4)

    # Add subplot labels
    subplotLabel(ax)

    return f

def R2X_figure(ax, tens):
    '''Create Parafac R2X Figure'''
    x_axis = np.arange(9)
    R2X = np.zeros(9)
    for i in range(1, 9):
        R2X[i] = calc_R2X_parafac(tens, i)
    ax.scatter(x_axis, R2X)
    ax.set_xlabel('Decomposition Rank')
    ax.set_ylabel('R2X')
    ax.set_title('PARAFAC')
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xticks(x_axis)

def cellLinePlot(ax, factors, r1, r2):
    '''Plot Cell Lines (tensor axis 0) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Cell Line Factors')
    setPlotLimits(ax, factors, r1, r2)

def genePlot(ax, factors, r1, r2):
    '''Plot genes (tensor axis 1) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Gene Factors')
    setPlotLimits(ax, factors, r1, r2)

def characPlot(ax, factors, r1, r2):
    '''Plot the measured genetic characteristics (tensor axis 2) in component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Genetic Characteristic Factors')
    setPlotLimits(ax, factors, r1, r2)

def setPlotLimits(axis, factors, r1, r2):
    '''Set appropriate limits for the borders of each component plot'''
    x = np.absolute(factors[:, r1 - 1])
    y = np.absolute(factors[:, r2 - 1])
    xlim = 1.1 * np.max(x)
    ylim = 1.1 * np.max(y)
    axis.set_xlim((-xlim, xlim))
    axis.set_ylim((-ylim, ylim))
    axis.axvline(color='black')
    axis.axhline(color='black')
