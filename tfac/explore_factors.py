import numpy as np
import pandas as pd
import seaborn as sns
import copy
import gseapy as gp
from pybiomart import Server
from .figures.figureCommon import getSetup


def plot_gene_components(factors, comp1, comp2, geneids, color, ax):
    geneA = factors.T[comp1]
    geneB = factors.T[comp2]
    gene_df = pd.DataFrame([geneA, geneB, geneids]).T
    gene_df.index = geneids
    gene_df.columns = ["Component A", "Component B", "Genes"]
    b = sns.scatterplot(data=gene_df, x='Component A', y='Component B', ax=ax, color=color) # blue
    b.set_xlabel("Component A",fontsize=25)
    b.set_ylabel("Component B",fontsize=25)
    b.tick_params(labelsize=20)


def label_points(df, names, ax):
    for i, point in df.iterrows():
        ax.text(point[df.columns[0]]+.002, point[df.columns[1]], str(point[names]), fontsize=13, fontweight="semibold", color='k')


def ensembl_convert(factors, geneids, decimals):
    convtable = pd.DataFrame()
    server = Server(host='http://www.ensembl.org')
    dataset = (server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl'])
    convtable = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    ourids = copy.deepcopy(geneids)
    if(decimals):
        for a in range(len(ourids)):
            ourids[a] = ourids[a][:ourids[a].index(".") ]
            
    
    newnames = []
    newtens = pd.DataFrame(factors)
    newtens["ensembl ids"] = ourids
    #droppedids = newtens[~newtens["ensembl ids"].isin(convtable["Gene stable ID"])]
    newtens = newtens[newtens["ensembl ids"].isin(convtable["Gene stable ID"])]
    for ensid in newtens["ensembl ids"]:
        table = convtable[convtable["Gene stable ID"] == ensid]
        table.reset_index(inplace = True)
        newnames.append(table.at[0, "Gene name"])

    newtens["Gene ID"] = newnames
    
    return newtens


def prerank(newtens, component, geneset):
    prtens = pd.concat((newtens["Gene ID"], newtens[newtens.columns[component]]), axis = 1)
    pre_res = gp.prerank(rnk=prtens, gene_sets=geneset, processes=16, min_size=10, max_size=5000, permutation_num=1000, weighted_score_type=0, outdir=None, seed=6)
    return pre_res.res2d


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=1,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
