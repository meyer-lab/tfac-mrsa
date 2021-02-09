import numpy as np
import pandas as pd
import pickle
from os.path import join, dirname
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from tensorly.parafac2_tensor import apply_parafac2_projections
from .MRSA_dataHelpers import form_MRSA_tensor, get_C1_patient_info, produce_outcome_bools, find_SVC_proba
from .tensor import MRSA_decomposition, R2Xparafac2
from .explore_factors import ensembl_convert, prerank


path_here = dirname(dirname(__file__))


def pickle_all():
    '''Create and pickle the best predicting decomposition, R2X, GSEA, and cell type deconvolution'''
    tensor_slices, _, geneIDs, _ = form_MRSA_tensor('serum')
    components = tensor_slices[0].shape[0]
    AllR2X = []
    parafac2tensors = []
    #Run factorization at each component number up to limit (38 due to 38 cytokines)
    for component in range(1, components + 1):
        parafac2tensor = MRSA_decomposition(tensor_slices, component, random_state=None)
        R2X = np.round(R2Xparafac2(tensor_slices, parafac2tensor), 6)
        parafac2tensors.append(parafac2tensor)
        AllR2X.append(R2X)
    best = Full_SVC(parafac2tensors, components)
    best_decomp = parafac2tensors[best[1] - 1]
    best_comps = best[2]
    gsea = Full_GSEA(best_decomp, best_comps, ['ARCHS4_Tissues', 'Jensen_TISSUES'], geneIDs)
    deconv = import_deconv()
    #pickle.dump([best_decomp, AllR2X, gsea, deconv], open("test.p", "wb"))


def import_deconv():
    '''Imports and returns cell deconvolution data.'''
    return pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter=",", index_col='sample').sort_index().drop(['gender', 'cell_type'], axis=1)


def Full_SVC(parafac2tensors, components):
    '''Perform cross validated SVC for each decomposition to determine optimal decomposition and component pair.
  
    Parameters: 
    parafac2tensors (list): list of 38 parafac2tensor objects
  
    Returns: 
    best_decomp (parafac2tensor): Decomposition with best status prediction
    comps (tuple): tuple of component pair that provides optimal prediction within SVC'''
    #import information on patient ID, P vs R status
    _, status_ID, _ = get_C1_patient_info()
    outcomes = produce_outcome_bools(status_ID)
    #For each decomposition, perform loo CV SVC
    pairs = []
    for comp in range(2, components + 1):
        patient_matrix = parafac2tensors[comp - 1][1][2]
        patient_matrix = patient_matrix[:61, :]
        loo = LeaveOneOut()
        bests = []
        choices = []
        for train, test in loo.split(patient_matrix):
            values_comps = []
            #for each component pair, perform SVC and score it
            for i in range(0, comp - 1):
                for j in range(i + 1, comp):
                    double = np.vstack((patient_matrix[train, i], patient_matrix[train, j])).T
                    decisions = find_SVC_proba(double, outcomes[train])        
                    auc = roc_auc_score(outcomes[train], decisions)
                    values_comps.append([i, j, auc])
            #Make it look nice and fit a model using the best pair, then predict
            df_comp = pd.DataFrame(values_comps)
            df_comp.columns = ["First", "Second", "AUC"]
            df_comp = df_comp.sort_values(by=["AUC"], ascending=False)
            best1 = df_comp.iloc[0, 0]
            best2 = df_comp.iloc[0, 1]
            clf = SVC().fit(np.vstack((patient_matrix[train, best1], patient_matrix[train, best2])).T, outcomes[train])
            choices.append(clf.decision_function(np.vstack((patient_matrix[test, best1], patient_matrix[test, best2])).T))
            bests.append(df_comp.iloc[:1, :])
        best_auc = roc_auc_score(outcomes, choices)
        pairs.append([best_auc, comp, bests])
    return sorted(pairs, reverse=True)[0]


def Full_GSEA(best_decomp, best_comps, libraries, geneids):
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
    newtens = ensembl_convert(gene_matrix, geneids)
    preranked_all = []
    #Run for all libraries
    all_dfs = []
    for library in libraries:
        #Perform GSEA on each component
        for comp in best_decomp[1][2].shape[1]:
            preranked = prerank(newtens, comp, library)
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
