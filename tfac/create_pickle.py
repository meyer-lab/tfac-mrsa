import pickle
import numpy as np
import pandas as pd
from .dataImport import form_missing_tensor
from .tensor import perform_TMTF
from .explore_factors import ensembl_convert, prerank


def pickle_all():
    """ Create and pickle the decomposition. """
    tensor_slices, _, _, _ = form_missing_tensor()
    tensor = np.stack((tensor_slices[0], tensor_slices[1])).T
    matrix = tensor_slices[2].T
    components = 40
    all_tensors = []
    #Run factorization at each component number up to chosen limit
    for component in range(1, components + 1):
        print(f"Starting decomposition with {component} components.")
        all_tensors.append(perform_TMTF(tensor, matrix, r=component))
    pickle.dump(all_tensors, open("Factorized.p", "wb"))


def Full_GSEA(gene_factors, best_comps, libraries, geneids):
    """Perform GSEA on the gene factor matrix, with each component done individually. Each of the 2 best predicting components from SVC are then compared to the other 37
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
    """
    # Convert ensembl ids to gene names and construct DataFrame
    newtens = ensembl_convert(gene_factors, geneids)
    preranked_all = []
    # Run for all libraries
    all_dfs = []
    for library in libraries:
        # Perform GSEA on each component
        for comp in gene_factors.shape[1]:
            preranked = prerank(newtens, comp, library)
            preranked_all.append(preranked)
        # Construct DataFrame for comparing nes for gene sets
        alls = pd.DataFrame()
        for i in range(gene_factors.shape[1]):
            df = preranked_all[i][preranked_all[i]["fdr"] < 0.05]
            alls = pd.concat((alls, df["nes"]), axis=1)
        alls.columns = range(38)
        gseacompA = preranked_all[best_comps[0]]
        gseacompB = preranked_all[best_comps[1]]
        find_unique(gseacompA, alls)
        find_unique(gseacompB, alls)
        all_dfs.append(gseacompA, gseacompB)
    return all_dfs


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
