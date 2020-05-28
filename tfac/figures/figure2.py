"""
This creates Figure 2 - Partial Tucker Decomposition Treatment and Time Plots.
"""
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCommon import subplotLabel, getSetup
from ..tensor import partial_tucker_decomp
from ..Data_Mod import form_tensor

components = 7
tensor, treatments, times = form_tensor()
results = partial_tucker_decomp(tensor, [2], components)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 2
    col = 4
    ax, f = getSetup((15, 8), (row, col))

    treatmentvsTimePlot(results, components, ax)

    # Add subplot labels
    subplotLabel(ax)
    return f


def treatmentvsTimePlot(results, components, ax):
    '''Plots the treatments over time by component for partial tucker decomposition of OHSU data'''
    frame_list = []
    for i in range(components):
        df = pd.DataFrame(results[0][i])
        frame_list.append(df)

    for component in range(components):
        column_list = []
        for i in range(components):
            column_list.append(pd.DataFrame(frame_list[i].iloc[:, component]))
        df = pd.concat(column_list, axis=1)
        df.columns = treatments
        df['Times'] = [0, 2, 4, 8, 24, 48]
        df = df.set_index('Times')
        b = sns.lineplot(data=df, ax=ax[component], dashes=None)
        b.set_title('Component ' + str(component + 1))
    for i in range(component + 1, len(ax)):
        ax[i].axis('off')
