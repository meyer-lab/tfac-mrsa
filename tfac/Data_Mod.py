"""Data pre-processing and tensor formation"""
import pandas as pd
import numpy as np
from .dataHelpers import importLINCSprotein, ohsu_data


def data_mod(x, df=None):
    """Creates a slice of the data tensor corresponding to the inputted treatment"""
    if not isinstance(df, pd.core.frame.DataFrame):
        df = importLINCSprotein()
    spec_df = df.loc[(df["Treatment"] == "Control") | (df["Treatment"] == x)]
    times = spec_df["Time"].to_numpy().tolist()
    spec_df = spec_df.drop(columns=["Sample description", "Treatment", "Time"])
    y = spec_df.to_numpy()
    return y, spec_df, times


def form_tensor():
    """Creates tensor in numpy array form and returns tensor, treatments, and time"""
    df = importLINCSprotein()
    tempindex = df["Sample description"]
    tempindex = tempindex[:36]
    i = 0
    for a in tempindex:
        tempindex[i] = a[3:]
        i += 1
    treatments = df["Treatment"][0:36]
    times = df["Time"][0:36]
    df = df.drop(["Sample description"], axis=1)
    by_row_index = df.groupby(df.index)
    df_means = by_row_index.mean()
    df_means.insert(0, "Treatment", value=treatments)
    df_means.insert(0, "Sample description", tempindex)
    unique_treatments = np.unique(df_means["Treatment"].values).tolist()
    unique_treatments.remove("Control")

    slices = []
    for treatment in unique_treatments:
        array, _, times = data_mod(treatment, df_means)
        slices.append(array)
    tensor = np.stack(slices)
    return tensor, unique_treatments, times


def LINCSCleanUp():
    """Cleaning up LINCS data for PARAFAC2 column order"""
    LINCSprotein = importLINCSprotein()
    ind = LINCSprotein.loc[LINCSprotein['Time'] >= 24]
    ind = ind.drop(columns='File')
    x = ['02_', '03_', '04_']
    y = ['24', '48']
    for a in range(0, 3):
        for b in range(0, 2):
            ind = ind.replace(x[a] + 'RPPA_BMP2_' + y[b], 'BMP2_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_EGF_' + y[b], 'EGF_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_HGF_' + y[b], 'HGF_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_IFNg_' + y[b], 'IFNg_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_OSM_' + y[b], 'OSM_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_TGFb_' + y[b], 'TGFb_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_pbs_' + y[b], 'PBS_' + y[b])
    ind = ind.drop(columns=['Treatment', 'Time'])
    ind = ind.groupby(['Sample description']).mean()
    ind = ind.sort_values('Sample description')
    indT = ind.T
    treatmentsTime = indT.columns.tolist()
    proteins = indT.index.tolist()
    indT = indT.to_numpy()
    return indT, treatmentsTime, proteins


def dataCleanUp():
    """Cleaning up OHSU data for PARAFAC2 column order"""
    atac, cycIF, GCP, _, L1000, RNAseq, RPPA = ohsu_data()
    tr = ['BMP2_', 'EGF_', 'HGF_', 'IFNG_', 'OSM_', 'TGFB_', 'PBS_', 'ctrl_0']
    for r in range(0, 7):
        cycIF = cycIF.drop(columns=[tr[r]+'1', tr[r]+'4', tr[r]+'8'])
        GCP = GCP.drop(columns=[tr[r]+'4', tr[r]+'8'])
        L1000 = L1000.drop(columns=[tr[r]+'1', tr[r]+'4', tr[r]+'8'])
        RPPA = RPPA.drop(columns=[tr[r]+'1', tr[r]+'4', tr[r]+'8'])
    atac = atac.drop(columns=tr[7])
    atac = atac.sort_index(axis=1)
    chromosomes = atac['peak'].to_list()
    atac = atac.drop(columns='peak').to_numpy()  
    cycIF = cycIF.drop(columns=tr[7])
    cycIF = cycIF.sort_index(axis=1)
    IFproteins = cycIF['feature'].to_list()
    cycIF = cycIF.drop(columns='feature').to_numpy()  
    GCP = GCP.drop(columns=tr[7])
    GCP = GCP.dropna()
    GCP = GCP.sort_index(axis=1)
    histones = GCP['histone'].to_list()
    GCP = GCP.drop(columns='histone').to_numpy()
    L1000 = L1000.drop(columns=tr[7])
    L1000 = L1000.sort_index(axis=1)
    geneExpression = L1000['probeset'].to_list()
    L1000 = L1000.drop(columns='probeset').to_numpy()
    RNAseq = RNAseq.drop(columns=tr[7])
    RNAseq = RNAseq.sort_index(axis=1)
    RNAGenes = RNAseq['ensembl_gene_id'].tolist()
    RNAseq = RNAseq.drop(columns='ensembl_gene_id').to_numpy()
    RPPA = RPPA.drop(columns=tr[7])
    RPPA = RPPA.sort_index(axis=1)
    RPPAProteins = RPPA['antibody'].tolist()
    RPPA = RPPA.drop(columns='antibody').to_numpy()
    return atac, cycIF, GCP, L1000, RNAseq, RPPA, chromosomes, IFproteins, histones, geneExpression, RNAGenes, RPPAProteins


def form_parafac2_tensor():
    """Creates tensor in numpy form and returns tensor, treatment by time, LINCS proteins, ATAC chromosomes, IF proteins, GCP histones, L1000 gene expression, RNA gene sequence, and RPPA proteins"""
    indTM, treatmentsTime, proteins = LINCSCleanUp()
    atacM, cycIFM, GCPM, L1000M, RNAseqM, RPPAM, chromosomes, IFproteins, histones, geneExpression, RNAGenes, RPPAProteins = dataCleanUp()
    p2slices = [indTM, atacM, cycIFM, GCPM, L1000M, RNAseqM, RPPAM]
    return p2slices, treatmentsTime, proteins, chromosomes, IFproteins, histones, geneExpression, RNAGenes, RPPAProteins
