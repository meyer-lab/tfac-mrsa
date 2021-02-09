"""
This creates Figure 2 - GSEA Plots
"""
import pickle
import pandas as pd
import seaborn as sns
from tensorly.parafac2_tensor import apply_parafac2_projections
from ..dataImport import form_MRSA_tensor
from ..explore_factors import label_points, ensembl_convert
from .figureCommon import subplotLabel, getSetup


def fig_2_setup():
    patient_matrices, _, preranked_tissues, _ = pickle.load(open("MRSA_pickle.p", "rb"))
    patient_mats_applied = apply_parafac2_projections(patient_matrices)
    _, _, geneids = form_MRSA_tensor(1, 1)
    df = patient_mats_applied[1][1][1]
    newtens = ensembl_convert(df, geneids, True)
    newtenspair = pd.concat((newtens[8], newtens[32], newtens["Gene ID"]), axis=1)

    genes8tlymph = newtenspair[newtenspair['Gene ID'].isin(preranked_tissues[0].loc['TLYMPHOCYTE'][6].split(";"))]
    genes32macro = newtenspair[newtenspair['Gene ID'].isin(preranked_tissues[1].loc['MACROPHAGE'][6].split(";"))]
    genes32mac_small = newtenspair[newtenspair['Gene ID'].isin(preranked_tissues[2].loc['Macrophage'][6].split(";"))]
    genes32mast = newtenspair[newtenspair['Gene ID'].isin(preranked_tissues[2].loc['Mast cell'][6].split(";"))]
    #gene_df = pd.concat((newtens[8], newtens[32], newtens["Gene ID"]), axis=1)

    types = [genes8tlymph, genes32macro, genes32mac_small, genes32mast]
    names = ["Comp A - T lymphocyte", "Comp B - Macrophage", "Comp B - Macrophage - Small set", "Comp B - Mast cell"]
    topgenes = []
    for cell in types:
        cell.columns = ['a', 'b', "Gene ID"]
        topgenes.append(pd.concat((cell.reindex(cell.b.abs().sort_values(ascending=False).index).iloc[:40], cell.reindex(cell.a.abs().sort_values(ascending=False).index).iloc[:40])))
    return types, names, topgenes


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    cells, name, topgene = fig_2_setup()
    # Get list of axis objects
    ax, f = getSetup((35, 22), (2, 2))
    #gene_df.columns = ["Component A", "Component B", "Gene ID"]
    for i in range(4):
        #b = sns.scatterplot(data=gene_df, x='Component A', y='Component B', ax=ax[i], color='cornflowerblue', s=50)
        b = sns.scatterplot(data=cells[i], x='a', y='b', ax=ax[i], color='lightcoral', s=50)
        b.set_xlabel("Component A", fontsize=25)
        b.set_ylabel("Component B", fontsize=25)
        b.tick_params(labelsize=20)
        b.set_title(name[i], fontsize=25)
        b.set_xlim(-1.25, 1.05)
        b.set_ylim(-1.05, .9)
        label_points(topgene[i], "Gene ID", ax[i])

    # Add subplot labels
    subplotLabel(ax)

    return f
