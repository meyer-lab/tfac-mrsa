"""
This creates Figure 2 - Partial Tucker Decomposition Treatment and Time Plots.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import partial_tucker_decomp, find_R2X_partialtucker
from ..Data_Mod import form_tensor

component = 5
tensor, treatment_list, times = form_tensor()
result = partial_tucker_decomp(tensor, [2], component)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 2
    col = 3
    ax, f = getSetup((12, 8), (row, col))

    R2X_Figure_PartialTucker(ax[0], tensor)
    treatmentvsTimePlot(result, component, treatment_list, ax[1::])

    # Add subplot labels
    subplotLabel(ax)

    return f


def treatmentvsTimePlot(results, components, treatments, ax):
    '''Plots the treatments over time by component for partial tucker decomposition of OHSU data'''
    frame_list = []
    for i in range(len(treatments)):
        df = pd.DataFrame(results[0][i])
        frame_list.append(df)

    for component in range(components):
        column_list = []
        for i in range(len(treatments)):
            column_list.append(pd.DataFrame(frame_list[i].iloc[:, component]))
        df = pd.concat(column_list, axis=1)
        df.columns = treatments
        df['Times'] = [0, 2, 4, 8, 24, 48]
        df = df.set_index('Times')
        b = sns.lineplot(data=df, ax=ax[component], dashes=None)
        b.set_title('Component ' + str(component + 1))
    for i in range(component + 1, len(ax)):
        ax[i].axis('off')


def R2X_Figure_PartialTucker(ax, input_tensor):
    '''Create Partial Tucker R2X Figure'''
    R2X = np.zeros(13)
    for i in range(1, 13):
        output = partial_tucker_decomp(input_tensor, [2], i)
        R2X[i] = find_R2X_partialtucker(output, input_tensor)
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("Partial Tucker Decomposition")
    ax.set_yticks([0, .2, .4, .6, .8, 1])
