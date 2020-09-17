"""
This creates Figure 6 - Individual Data performance
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from ..MRSA_dataHelpers import produce_outcome_bools, get_patient_info, form_MRSA_tensor, find_SVC_proba
from .figureCommon import subplotLabel, getSetup

def find_CV_proba(patient_matrix, outcomes, random_state=None, C=1):
    proba = cross_val_predict(LogisticRegression(penalty='l2', solver='lbfgs', C=C, random_state=random_state, max_iter=10000, fit_intercept=False), patient_matrix, outcomes, cv=30, method="predict_proba")
    return proba[:, 1]
def find_regularization(patient_matrix, outcomes, random_state=None):
    clf = LogisticRegressionCV(Cs=20, cv=10, random_state=random_state, fit_intercept=False, penalty='l2', solver='lbfgs', max_iter=100000).fit(patient_matrix, outcomes)
    reg = clf.C_
    return reg[0]
def fig_6_setup():

    tensor_slices, _, _ = form_MRSA_tensor()
    _, statusID = get_patient_info()
    outcomes = produce_outcome_bools(statusID)
    cytokines = tensor_slices[0].T
    genes = tensor_slices[1].T
    reg_genes = find_regularization(genes, outcomes)
    reg_cyto = find_regularization(cytokines, outcomes)
    proba_genes = find_CV_proba(genes, outcomes, random_state=None, C=reg_genes)
    proba_cyto = find_CV_proba(cytokines, outcomes, random_state=None, C=reg_cyto)
    auc_genes = roc_auc_score(outcomes, proba_genes)
    auc_cyto = roc_auc_score(outcomes, proba_cyto)

    cytokines = tensor_slices[0].T
    values_cyto = []
    for i in range(0, cytokines.shape[1] - 1):
        for j in range(i + 1, cytokines.shape[1]):
            double = np.vstack((cytokines[:, i], cytokines[:, j])).T
            decisions = find_SVC_proba(double, outcomes)        
            auc = roc_auc_score(outcomes, decisions)
            values_cyto.append([i, j, auc])
    df_cyto = pd.DataFrame(values_cyto)
    df_cyto.columns = ["First", "Second", "AUC"]
    svc_cyto = df_cyto.sort_values(by='AUC', ascending=False).iloc[0, 2]

    patient_matrices = pickle.load( open( "cyto_exp.p", "rb" ) )
    test = patient_matrices[37][1][2]
    values_comps = []
    for i in range(0, 37):
        for j in range(i + 1, 38):
            double = np.vstack((test[:, i], test[:, j])).T
            decisions = find_SVC_proba(double, outcomes)        
            auc = roc_auc_score(outcomes, decisions)
            values_comps.append([i, j, auc])
    df_comp = pd.DataFrame(values_comps)
    df_comp.columns = ["First", "Second", "AUC"]
    df_comp.sort_values(by='AUC', ascending=False)
    svc_both = df_comp.sort_values(by='AUC', ascending=False).iloc[0, 2]

    return auc_cyto, auc_genes, svc_cyto, svc_both

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    cyto_logit, genes_logit, cyto_svc, both_svc = fig_6_setup()
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))
    df = pd.DataFrame({'Data/Analysis': ["Log Reg - Cytokines", "Log Reg - Genes", "SVC - Cytokines", "SVC - Factorization"], 'AUC': [cyto_logit, genes_logit, cyto_svc, both_svc]})
    b = sns.barplot(data=df, x='Data/Analysis', y='AUC', ax=ax[0])
    b.set_xlabel("Data/Analysis", fontsize=25)
    b.set_ylabel("AUC", fontsize=25)
    b.tick_params(labelsize=20)
    # Add subplot labels
    subplotLabel(ax)

    return f
