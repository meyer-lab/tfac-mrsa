"""
This creates Figure 1 - CP Decomposition Plots
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..Data_Mod import form_tensor
from ..tensor import cp_decomp, find_R2X_parafac


tensor, treatments, times = form_tensor()
results = cp_decomp(tensor, 2)
comps = results[1]


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    R2X_figure(ax[0])
    treatmentPlot(ax[1], comps[0], 1, 2, treatments)
    timePlot(ax[2], comps[1])
    proteinPlot(ax[3], comps[2], 1, 2)

    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_figure(ax):
    '''Create Parafac R2X Figure'''
    R2X = np.zeros(10)
    nComps = range(1, len(R2X))
    for i in nComps:
        output = cp_decomp(tensor, i)
        R2X[i] = find_R2X_parafac(output, tensor)
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("CP Decomposition")
    ax.set_yticks([0, .2, .4, .6, .8, 1])


def treatmentPlot(ax, factors, r1, r2, senthue):
    '''Plot treatments (tensor axis 0) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax, hue=senthue)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Treatment Factors')
    setPlotLimits(ax, factors, r1, r2)


def timePlot(ax, factors):
    '''Plot time points (tensor axis 1) in factorization component space'''
    for i in np.arange(factors.shape[1]):
        sns.lineplot(times, factors[:, i], ax=ax, label="Component " + str(i))
    ax.set_xlabel("Measurement Time")
    ax.set_ylabel('Component Value')
    ax.set_title('Time Components')


def proteinPlot(ax, factors, r1, r2):
    '''Plot proteins (tensor axis 2) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Protein Factors')
    setPlotLimits(ax, factors, r1, r2)


def setPlotLimits(axis, factors, r1, r2):
    '''Set appropriate limits for the borders of each component plot'''
    x = np.absolute(factors[:, r1 - 1])
    y = np.absolute(factors[:, r2 - 1])
    xlim = 1.1 * np.max(x)
    ylim = 1.1 * np.max(y)
    axis.set_xlim((-xlim, xlim))
    axis.set_ylim((-ylim, ylim))
