"""
This creates Figure 1.
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import calc_R2X_parafac
from ..dataHelpers import getCellLineComps, getGeneComps, getCharacteristicComps, cellLineNames

cell_comps = getCellLineComps()
gene_comps = getGeneComps()
characteristic_comps = getCharacteristicComps()


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))

    ax[0].axis('off')  # blank out first axis for cartoon
    ax[1].axis('off')
    
    R2X_figure(ax[2])
    cellLinePlot(ax[3], getCellLineComps(), 1, 2)
    cellLinePlot(ax[4], getCellLineComps(), 2, 4)
    cellLinePlot(ax[5], getCellLineComps(), 6, 7)
    cellLinePlot(ax[6], getCellLineComps(), 12, 2)
    cellLinePlot(ax[7], getCellLineComps(), 19, 6)
    cellLinePlot(ax[8], getCellLineComps(), 22, 6)


    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_figure(ax):
    '''Create Parafac R2X Figure'''
    ### THIS DATA COMES FROM MATLAB ###
    nComps = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25]
    R2X = [0, .681, .744, .787, .805, .819, .861, .887, .904, .916]
    ax = sns.scatterplot(nComps, R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_title("CP Decomposition")


def cellLinePlot(ax, factors, r1, r2):
    '''Plot Cell Lines (tensor axis 0) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax, hue=cellLineNames(), legend=False)
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
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax, style=['Gene Expression', 'Copy Number', 'Methylation'])
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
