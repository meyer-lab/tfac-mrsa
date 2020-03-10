"""
This creates Figure 2.
"""
import numpy as np
import seaborn as sns
from ..drugOrg import importDrugs, tempFilter
from .figureCommon import subplotLabel, getSetup
from ..regression import KFoldCV

sns.set(style="white")
sns.set_context('notebook')

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    ax[0].axis('off')  # blank out first axis for cartoon


    drugArr = importDrugs()
    x, y = tempFilter(drugArr[5])
    y = y[:,-1]

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    #predVsActual(ax[1], x, y, "SVR")
    #predVsActual(ax[2], x, y, "Ridge")
    #predVsActual(ax[3], x, y, "RF")
    # Add subplot labels
    subplotLabel(ax)
    return f

def barPlot(ax):
    average = np.array([-0.216, -0.124, 0.178, 0.28, 0.29, 0.305, 0.306, 0.309])
    std = np.array([0.175, 0.178, 0.0985, 0.0735, 0.0743, 0.0717, 0.0623, 0.0519])
    regMethods = ["Decision Tree", "XGBoost", "OLS", "Random Forest", "LASSO", "SVR", "Elastic Net", "Ridge"]
    x_pos = np.arange(len(average))
    ax.bar(x_pos, average, yerr=std, capsize=2, color='darkred')
    ax.set_ylabel('Average R2 Score')
    ax.set_xticklabels(regMethods)
    ax.set_title('Comparison of R2 Scores Across Different Models')


def predVsActual(ax, x, y, reg):
    _, predicted, actual = KFoldCV(x, y, reg)

    sns.scatterplot(actual, predicted, color = 'darkslategrey', ax=ax)
    sns.despine()
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual ' + reg)
