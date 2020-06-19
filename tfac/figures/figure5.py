"""
This creates Figure 5 - MRSA R2X for parafac2.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorly as tl
from .figureCommon import subplotLabel, getSetup
from ..tensor import R2Xparafac2, MRSA_decomposition


tl.set_backend("numpy")
components = 38
variance = 1


AllR2X = []
for i in range(1, components + 1):
    tensor_slices, parafac2tensor = MRSA_decomposition(variance, i)
    AllR2X.append(R2Xparafac2(tensor_slices, parafac2tensor))
df = pd.DataFrame(AllR2X)

comps = []
for i in range(1, components + 1):
    comps.append(i)
df['Component'] = comps

df.columns = ['Cytokines', 'GeneIDs', 'Component']
test = pd.melt(df, id_vars=['Component'])



def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 7), (1, 1))
    b = sns.scatterplot(data=test, x='Component', y='value', hue='variable', style='variable', ax=ax[0], s=100)
    b.set_xlabel("Component", fontsize=20)
    b.set_ylabel("R2X", fontsize=20)
    b.tick_params(labelsize=15)
    plt.legend(prop={'size': 15})
    ax[0].set_ylim(0, 1)
    
    # Add subplot labels
    subplotLabel(ax)

    return f

