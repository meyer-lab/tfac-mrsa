"""
This creates Figure 2 - Partial Tucker Decomposition Treatment and Time Plots.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import partial_tucker_decomp, find_R2X_partialtucker, flip_factors
from ..Data_Mod import form_tensor
from ..dataHelpers import importLINCSprotein


component = 5
tensor, treatment_list, times = form_tensor()
pre_flip_result = partial_tucker_decomp(tensor, [2], component)

result = flip_factors(pre_flip_result)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 2
    col = 6
    ax, f = getSetup((24, 11), (row, col))

    R2X_Figure_PartialTucker(ax[0], tensor)
    treatmentvsTimePlot(result, component, treatment_list, ax[1:6])
    proteinBoxPlot(ax[7], result[1][0][:, 0], 1)
    proteinBoxPlot(ax[8], result[1][0][:, 1], 2)
    proteinBoxPlot(ax[9], result[1][0][:, 2], 3)
    proteinBoxPlot(ax[10], result[1][0][:, 3], 4)
    proteinBoxPlot(ax[11], result[1][0][:, 4], 5)

    # Add subplot labels
    subplotLabel(ax)

    return f


def treatmentvsTimePlot(results, components, treatments, ax):
    '''Plots the treatments over time by component for partial tucker decomposition of OHSU data'''
    frame_list = []
    for i in range(len(treatments)):
        df = pd.DataFrame(results[0][i])
        frame_list.append(df)

    for comp in range(components):
        column_list = []
        for i in range(len(treatments)):
            column_list.append(pd.DataFrame(frame_list[i].iloc[:, comp]))
        df = pd.concat(column_list, axis=1)
        df.columns = treatments
        df['Times'] = [0, 2, 4, 8, 24, 48]
        df = df.set_index('Times')
        b = sns.lineplot(data=df, ax=ax[comp], dashes=None)
        b.set_title('Component ' + str(comp + 1))
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


def outliersForPlot(dframe):
    '''Determines outliers based on IQR range by component and returns dictionary by component that allows annotation'''
    df = dframe.copy(deep=True)
    proteins = importLINCSprotein()
    columns = proteins.columns[3:298]
    df["Proteins"] = columns
    Q1 = df.quantile(.25)
    Q3 = df.quantile(.75)
    IQR = Q3 - Q1
    prots = {}
    for i in range(df.columns.size - 1):
        prots[i] = []
        for _, col in df.iterrows():
            if (col[i] < (Q1[i] - 1.7 * IQR[i])) or (col[i] > (Q3[i] + 1.7 * IQR[i])):
                tup = [i, col[i], col['Proteins'][:-4], True, True]
                prots[i].append(tup)
        prots[i].sort(key=lambda x: x[1])
        for idx, tup in enumerate(prots[i]):
            if idx < len(prots[i]) - 4:
                if tup[1] > prots[i][idx + 2][1] - .012 and tup[3] == tup[4]:
                    random1 = np.random.choice([0, 1, 1])
                    prots[i][idx + (random1 * 2)][3] = False
                    tup[4] = False
                    prots[i][idx + 2][4] = False
                elif tup[1] > prots[i][idx + 2][1] - .012 and tup[3]:
                    prots[i][idx + 2][3] = False
                    prots[i][idx + 2][4] = False
                if tup[1] > prots[i][idx + 4][1] - .012 and tup[3]:
                    random2 = np.random.randint(0, 2)
                    #print(tup[2], random1, random2)
                    prots[i][idx + random2 * 4][3] = False
                    prots[i][idx + 4][4] = False
    return prots


def proteinBoxPlot(ax, resultsIn, componentIn):
    '''Plots protein component in partial tucker factorization space with annotation of some outliers'''
    df = pd.DataFrame(resultsIn)
    prots = outliersForPlot(df)
    sns.boxplot(data=df, ax=ax)
    ax.set_xlabel("Component " + str(componentIn))
    ax.set_ylabel('Component Value')
    ax.set_title('Protein Factors')
    for comp in prots:
        offset_side = 0
        for outlier in prots[comp]:
            if outlier[3]:
                if offset_side == 0:
                    ax.text(outlier[0] + .05, outlier[1] - .005, outlier[2], horizontalalignment='left', size='large', color='black', weight=100)
                    offset_side = 1
                elif offset_side == 1:
                    ax.text(outlier[0] - .05, outlier[1] - .005, outlier[2], horizontalalignment='right', size='large', color='black', weight=100)
                    offset_side = 0
            else:
                offset_side = 1 - offset_side
