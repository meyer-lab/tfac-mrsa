"""
This creates Figure 2.
"""
import pickle
import pandas as pd
import seaborn as sns
from tensorly.parafac2_tensor import apply_parafac2_projections
from ..MRSA_dataHelpers import form_MRSA_tensor
from ..explore_factors import label_points, ensembl_convert
from .figureCommon import subplotLabel, getSetup

patient_matrices, _, preranked_tissues = pickle.load(open("MRSA_pickle.p", "rb"))
patient_mats_applied = apply_parafac2_projections(patient_matrices)
_, _, geneids = form_MRSA_tensor(1, 1)
df = patient_mats_applied[1][1][1]
newtens = ensembl_convert(df, geneids, True)
newtenspair = pd.concat((newtens[8], newtens[32], newtens["Gene ID"]), axis=1)

genes8tlymph = newtenspair[newtenspair['Gene ID'].isin(preranked_tissues[0].loc['TLYMPHOCYTE'][6].split(";"))]
genes8plascell = newtenspair[newtenspair['Gene ID'].isin(preranked_tissues[0].loc['PLASMA CELL'][6].split(";"))]
genes8CD19B = newtenspair[newtenspair['Gene ID'].isin(preranked_tissues[0].loc['CD19+ B CELLS'][6].split(";"))]
genes32macro = newtenspair[newtenspair['Gene ID'].isin(preranked_tissues[1].loc['MACROPHAGE'][6].split(";"))]
genes32alvmac = newtenspair[newtenspair['Gene ID'].isin(preranked_tissues[1].loc['ALVEOLAR MACROPHAGE'][6].split(";"))]
genes32plasdend = newtenspair[newtenspair['Gene ID'].isin(preranked_tissues[1].loc['PLASMACYTOID DENDRITIC CELL'][6].split(";"))]

gene_df = pd.concat((newtens[8], newtens[32], newtens["Gene ID"]), axis=1)

types = [genes8tlymph, genes8plascell, genes8CD19B, genes32macro, genes32alvmac, genes32plasdend]
names = ["Comp A - T lymphocyte", " Comp A - Plasma cell", "Comp A - CD19+ B cell", "Comp B - Macrophage", "Comp B - Alveolar Macrophage", "Comp B - Plasmacytoid DC"]
topgenes = []
for cell in types:
    cell.columns = ['a', 'b', "Gene ID"]
    topgenes.append(pd.concat((cell.reindex(cell.b.abs().sort_values(ascending=False).index).iloc[:30], cell.reindex(cell.a.abs().sort_values(ascending=False).index).iloc[:30])))

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((45, 27), (2, 3))
    gene_df.columns = ["Component A", "Component B", "Gene ID"]
    for i in range(6):
        b = sns.scatterplot(data=gene_df, x='Component A', y='Component B', ax=ax[i], color='cornflowerblue', s=50)
        sns.scatterplot(data=types[i], x='a', y='b', ax=ax[i], color='lightcoral', s=50)
        b.set_xlabel("Component A", fontsize=25)
        b.set_ylabel("Component B", fontsize=25)
        b.tick_params(labelsize=20)
        b.set_title(names[i], fontsize=25)
        label_points(topgenes[i], "Gene ID", ax[i])

    # Add subplot labels
    subplotLabel(ax)

    return f
