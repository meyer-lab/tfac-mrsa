"""
This creates Figure 3. - Logistic Regression Components vs AUC
"""
import pickle
import seaborn as sns
from .figureCommon import subplotLabel, getSetup


df_comp = pickle.load(open("LogReg.p", "rb"))


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 8), (1, 1))
    b = sns.pointplot(data=df_comp, x="Components", y="AUC", ax=ax[0], s=70, join=False)  # blue
    b.set_xlabel("Components", fontsize=20)
    b.set_ylabel("AUC", fontsize=20)
    b.tick_params(labelsize=14)
    ax[0].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
