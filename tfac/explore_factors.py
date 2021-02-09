"""Functions for exploring components to MRSA parafac2 decomposition"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gseapy as gp
from pybiomart import Server
from tensorly.parafac2_tensor import apply_parafac2_projections


def label_points(df, names, ax):
    """Label given points given df of coordinates and names"""
    for _, point in df.iterrows():
        ax.text(point[df.columns[0]] + .002, point[df.columns[1]], str(point[names]), fontsize=13, fontweight="semibold", color='k')


def ensembl_convert(factors, geneids):
    """Converts array of gene weights and list of ensembl ids to dataframe for gsea"""
    #Import ensembl for id conversion
    convtable = pd.DataFrame()
    server = Server(host='http://www.ensembl.org')
    dataset = (server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl'])
    convtable = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    ourids = copy.deepcopy(geneids)
    newnames = []
    newtens = pd.DataFrame(factors)
    newtens["ensembl ids"] = ourids
    #droppedids = newtens[~newtens["ensembl ids"].isin(convtable["Gene stable ID"])]
    newtens = newtens[newtens["ensembl ids"].isin(convtable["Gene stable ID"])]
    for ensid in newtens["ensembl ids"]:
        table = convtable[convtable["Gene stable ID"] == ensid]
        table.reset_index(inplace=True)
        newnames.append(table.at[0, "Gene name"])

    newtens["Gene ID"] = newnames

    return newtens


def prerank(newtens, component, geneset):
    """Runs prerank gsea on specific component/gene list"""
    prtens = pd.concat((newtens["Gene ID"], newtens[newtens.columns[component]]), axis=1)
    pre_res = gp.prerank(rnk=prtens, gene_sets=geneset, processes=16, min_size=1, max_size=5000, permutation_num=1000, weighted_score_type=0, outdir=None, seed=6)
    return pre_res.res2d


def Full_GSEA(best_decomp, best_comps, library, geneids):
    '''Perform GSEA on the gene factor matrix, with each component done individually. Each of the 2 best predicting components from SVC are then compared to the other 37 
    to find which gene set are uniquely enriched in that component.

    Some important things to note - the library chosen and the ultimate unique gene sets that are relevant depend upon the application. 
    For our purposes, we wanted to look at the immune system, and decided that observing enrichment of certain cell types would be a better method than looking at 
    smaller cellular programs, thus we chose libraries of cell types. The results were examined manually, as there were many gene sets irrelevant to us - such as 
    cell types that were not among the samples taken nor involved in the immune system. Gene sets that appeared as highly unique - aka with a normalized 
    enrichment score greater in magnitude than in other components - and relevant are those determined to be enriched.
    
    Parameters: 
    best_decomp (parafac2tensor): Decomposition with best status prediction
    best_comps (tuple): tuple of component pair that provides optimal prediction within SVC
    library (string): Name of Enricher library to use
    geneids (list): Ensembl ids for all genes in matrix
  
    Returns: 
    gseacompA, gseacompB (DataFrame): GSEA results for component pair and given gene set, with columns for "uniqueness" based on nes across components
    '''
    #Get full factor matrices from tensor object
    patient_mats_applied = apply_parafac2_projections(best_decomp)
    gene_matrix = patient_mats_applied[1][1][1]
    #Convert ensembl ids to gene names and construct DataFrame
    newtens = ensembl_convert(gene_matrix, geneids, False)
    preranked_all = []
    #Perform GSEA on each component
    for comp in best_decomp[1][2].shape[1]:
        preranked = prerank(newtens, comp, 'KEGG_2019_Human')
        preranked_all.append(preranked)
    #Construct DataFrame for comparing nes for gene sets
    alls = pd.DataFrame()
    for i in range(best_decomp[1][2].shape[1]):
        df = preranked_all[i][preranked_all[i]['fdr'] < .05]
        alls = pd.concat((alls, df['nes']), axis=1)
    alls.columns = range(38)
    gseacompA = preranked_all[best_comps[0]]
    gseacompB = preranked_all[best_comps[1]]
    find_unique(gseacompA, alls)
    find_unique(gseacompB, alls)
    return gseacompA, gseacompB


def find_unique(df, alls):
    "For given component prerank results, determines for each geneset how many components have larger positive/negative magnitude NES"
    largepos = []
    largeneg = []
    for geneset in df.index:
        counti = 0
        countn = 0
        if geneset in alls.index:
            for comp in alls.loc[geneset]:
                if comp > 1 * df.loc[geneset][1] and df.loc[geneset][1] > 0:
                    counti += 1
                if comp < -1 * df.loc[geneset][1] and df.loc[geneset][1] > 0:
                    countn += 1
                if comp < 1 * df.loc[geneset][1] and df.loc[geneset][1] < 0:
                    countn += 1
                if comp > -1 * df.loc[geneset][1] and df.loc[geneset][1] < 0:
                    counti += 1
        largepos.append(counti)
        largeneg.append(countn)
    df["largepos"] = largepos
    df["largeneg"] = largeneg
