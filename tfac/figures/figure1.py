"""
This creates Figure 1 - MRSA R2X
"""
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .figureCommon import subplotLabel, getSetup



_, AllR2X, _ = pickle.load(open("MRSA_pickle.p", "rb"))
df = pd.DataFrame(AllR2X)
components = 38
comps = []
for i in range(1, components + 1):
    comps.append(i)
df["Component"] = comps

df.columns = ["Cytokines", "GeneIDs", "Component"]
test = pd.melt(df, id_vars=["Component"])


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 1))
    b = sns.scatterplot(data=test, x="Component", y="value", hue="variable", style="variable", ax=ax[0], s=100)
    b.set_xlabel("Component", fontsize=20)
    b.set_ylabel("R2X", fontsize=20)
    b.tick_params(labelsize=15)
    plt.legend(prop={"size": 15})
    ax[0].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
