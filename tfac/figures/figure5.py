"""
This creates Figure 5 - MRSA R2X for parafac2.
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorly as tl
from .figureCommon import subplotLabel, getSetup


tl.set_backend("numpy")

R2X_list = pickle.load(open("R2X.p", "rb"))
components = 38
to_plot = []
comps = []
for i in range(1, components + 1):
    comps.append(i)
for df in R2X_list:
    df['Component'] = comps
    df.columns = ['Cytokines', 'GeneIDs', 'Component']
    test = pd.melt(df, id_vars=['Component'])
    test.columns = ["Component", "Measurement", "Value"]
    to_plot.append(test)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 20), (3, 1))
    for idx, variance in enumerate([.0007, .007, .07]):
        b = sns.scatterplot(data=to_plot[idx], x='Component', y='Value', hue='Measurement', style='Measurement', ax=ax[idx], s=100)
        b.set_xlabel("Component", fontsize=20)
        b.set_ylabel("R2X", fontsize=20)
        b.tick_params(labelsize=15)
        b.set_title("R2X with Cytokine Variance Scaling = " + str(variance), fontsize=20)
        b.legend(loc='lower right')
        plt.setp(ax[idx].get_legend().get_texts(), fontsize='15') # for legend text
        plt.setp(ax[idx].get_legend().get_title(), fontsize='15')
        ax[idx].set_ylim(0, 1)


    # Add subplot labels
    subplotLabel(ax)

    return f
