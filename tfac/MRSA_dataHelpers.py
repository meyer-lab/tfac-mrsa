"""Data import and processing for the MRSA data"""
from os.path import join, dirname
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from tensorly.metrics.regression import variance as tl_var


path_here = dirname(dirname(__file__))


def find_SVC_proba(patient_matrix, outcomes):
    """Given a particular patient matrix and outcomes list, performs cross validation of SVC and returns the decision function to be used for AUC"""
    proba = cross_val_predict(SVC(kernel="rbf"), patient_matrix, outcomes, cv=30, method="decision_function")
    return proba


def produce_outcome_bools(statusID):
    """Returns a list of booleans for progressor/resolver status ready to use for logistic regression"""
    outcome_bools = []
    for outcome in statusID:
        if outcome == "APMB":
            outcome_bools.append(0)
        else:
            outcome_bools.append(1)

    return np.asarray(outcome_bools)


def get_C1_patient_info():
    """Return specific patient ID information"""
    dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter="\t")
    cohortID = list(dataCohort["sample"])
    statusID = list(dataCohort["outcome_txt"])
    type_ID = list(dataCohort["sampletype"])
    return cohortID, statusID, type_ID


def form_missing_tensor(variance1=1, variance2=1, variance3=1):
    '''Create list of normalized data matrices for parafac2: cytokines from serum, cytokines from plasma, RNAseq'''

    cyto_list, cytokines, dfExp, geneIDs = full_import()
    #Make initial data slices
    _, _, type_ID = get_C1_patient_info()
    cyto1 = cyto_list[0].T
    cyto1["type"] = type_ID
    dfCyto_serum = pd.concat([cyto1[cyto1["type"] == 'Serum'].T.drop("type"), cyto_list[1]], axis=1)
    dfCyto_plasma = pd.concat([cyto1[cyto1["type"] == 'Plasma'].T.drop("type"), cyto_list[2]], axis=1)
    #Eliminate normalization bias
    dfCyto_serum = dfCyto_serum * ((1 / tl_var(dfCyto_serum)) ** 0.5) * variance1
    dfCyto_plasma = dfCyto_plasma * ((1 / tl_var(dfCyto_plasma)) ** 0.5) * variance2
    dfExp = dfExp * ((1 / tl_var(dfExp)) ** 0.5) * variance3
    #Add in NaNs
    temp = pd.concat([dfCyto_serum, dfCyto_plasma, dfExp])
    dfCyto_serum = temp.iloc[:38, :]
    dfCyto_plasma = temp.iloc[38:76, :]
    dfExp = temp.iloc[76:, :]

    cohortID = dfExp.columns.to_list()
    serumNumpy = dfCyto_serum.to_numpy()
    plasmaNumpy = dfCyto_plasma.to_numpy()
    expNumpy = dfExp.to_numpy()
    tensor_slices = [serumNumpy, plasmaNumpy, expNumpy]
    return tensor_slices, cytokines, geneIDs, cohortID


def form_MRSA_tensor(sample_type, variance1=1, variance2=1):
    """Create list of data matrices for parafac2. The sample type argument chosen for cohort 3 is incorporated into the tensor, the data for the other type is not used.
    Keeps both types for cohort 1."""

    cyto_list, cytokines, dfExp, geneIDs = full_import()

    for cyto_idx in range(1, 3):
        cyto_list[cyto_idx].drop([patient for patient in cyto_list[cyto_idx].columns if patient not in dfExp.columns], axis=1, inplace=True)
    dfExp.drop([patient for patient in dfExp.columns if patient not in cyto_list[0].columns.to_list() + cyto_list[1].columns.to_list()], axis=1, inplace=True)
    #Uses sample type argument to construct tensor, specifically choosing which data set for cohort 3 will be used. 
    if sample_type == 'serum':
        dfCyto_serum = pd.concat([cyto_list[0], cyto_list[1]], axis=1)
        #Below line, as well as others in same format are to avoid the decomposition method biasing one of the slices due to its overall variance being large due to normalization changes.
        cytoNumpy = dfCyto_serum.to_numpy()
        cytoNumpy = cytoNumpy * ((1 / np.var(cytoNumpy)) ** 0.5) * variance1
        cohortID = dfExp.columns.to_list()
        expNumpy = dfExp.to_numpy()
        expNumpy = expNumpy * ((1 / np.var(expNumpy)) ** 0.5) * variance2
        tensor_slices = [cytoNumpy, expNumpy]
    elif sample_type == 'plasma':
        dfCyto_plasma = pd.concat([cyto_list[0], cyto_list[2]], axis=1)
        cytoNumpy = dfCyto_plasma.to_numpy()
        cytoNumpy = cytoNumpy * ((1 / np.var(cytoNumpy)) ** 0.5) * variance1
        dfExp = dfExp.drop(7008, axis=1)
        cohortID = dfExp.columns.to_list()
        expNumpy = dfExp.to_numpy()
        expNumpy = expNumpy * ((1 / np.var(expNumpy)) ** 0.5) * variance2
        tensor_slices = [cytoNumpy, expNumpy]
    else:
        raise ValueError("Bad sample type selection.")

    return tensor_slices, cytokines, geneIDs, cohortID


def full_import():
    '''Imports raw cytokine and RNAseq data for both cohort 1 and 3. Performs normalization and fixes bad values.'''
    #Import cytokines
    dfClin, dfCoh = importClinicalMRSA()
    dfCyto_c1 = clinicalCyto(dfClin, dfCoh)
    dfCyto_c1 = dfCyto_c1.set_index("sid")
    dfCyto_c3_serum, dfCyto_c3_plasma = import_C3_cyto()
    #Import RNAseq
    dfExp_c1 = importCohort1Expression()
    dfExp_c3 = importCohort3Expression()

    #Modify cytokines
    dfCyto_c1.columns = dfCyto_c3_serum.columns
    #Fix limit of detection error - bring to next lowest value
    dfCyto_c1["IL-12(p70)"] = [val * 16000000 if val < 1 else val for val in dfCyto_c1["IL-12(p70)"]]
    #normalize separately and extract cytokines
    cyto_list = [dfCyto_c1, dfCyto_c3_serum, dfCyto_c3_plasma]
    for idx, df in enumerate(cyto_list):
        df = df.div(df.apply(gmean, axis=1).to_list(), axis=0)
        df = df.apply(np.log, axis=0)
        df = df.sub(df.apply(np.mean, axis=0).to_list(), axis=1)
        cyto_list[idx] = df.T
    cytokines = dfCyto_c1.columns.to_list()

    #Modify RNAseq
    dfExp_c1 = removeC1_dupes(dfExp_c1)
    dfExp_c1 = dfExp_c1.set_index("Geneid")
    #Drop genes not shared
    dfExp_c1.sort_values("Geneid", inplace=True)
    dfExp_c3.sort_values("Geneid", inplace=True)
    drop_df = pd.concat([dfExp_c1, dfExp_c3], axis=1, join='inner')
    #Remove those with very few reads on average
    drop_df["Mean"] = drop_df.apply(np.mean, axis=1)
    mean_drop = drop_df[drop_df['Mean'] < 2].index
    dfExp_c1 = dfExp_c1.drop(mean_drop)
    dfExp_c3 = dfExp_c3.drop(mean_drop)
    #Store info for later
    pats_c1 = dfExp_c1.columns.astype(int)
    pats_c3 = dfExp_c3.columns.astype(int)
    genes_c1 = dfExp_c1.index
    genes_c3 = dfExp_c3.index
    #Normalize
    dfExp_c1 = scale(dfExp_c1, axis=0)
    dfExp_c1 = scale(dfExp_c1, axis=1)
    dfExp_c3 = scale(dfExp_c3, axis=0)
    dfExp_c3 = scale(dfExp_c3, axis=1)
    #Return to dataframe format after scaling
    dfExp_c1 = pd.DataFrame(dfExp_c1, index=genes_c1, columns=pats_c1)
    dfExp_c3 = pd.DataFrame(dfExp_c3, index=genes_c3, columns=pats_c3)
    dfExp = pd.concat([dfExp_c1, dfExp_c3], axis=1, join='inner')
    geneIDs = dfExp.index.to_list()
    return cyto_list, cytokines, dfExp, geneIDs


def import_methylation():
    """import methylation data"""
    dataMeth = pd.read_csv(join(path_here, "tfac/data/mrsa/MRSA.Methylation.txt.xz"), delimiter=" ", compression="xz")
    locs = dataMeth.values[:, 0]
    return dataMeth, locs


def importClinicalMRSA():
    """import clincal MRSA data"""
    dataClin = pd.read_csv(join(path_here, "tfac/data/mrsa/mrsa_s1s2_clin+cyto_073018.csv"))
    dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter="\t")
    return dataClin, dataCohort


def clinicalCyto(dataClinical, dataCohort):
    """isolate cytokine data from clinical"""
    rowSize, _ = dataClinical.shape
    patientID = list(dataClinical["sid"])

    dataClinical = dataClinical.drop(dataClinical.iloc[:, 0:3], axis=1)
    dataClinical = dataClinical.drop(dataClinical.iloc[:, 1:206], axis=1)

    # isolate patient IDs from cohort 1
    dataCohort = dataCohort.drop(columns=["age", "gender", "race", "sampletype", "pair", "outcome_txt"], axis=1)
    cohortID = list(dataCohort["sample"])
    IDSize, _ = dataCohort.shape

    cytokineData = pd.DataFrame()

    for y in range(0, rowSize):
        for z in range(0, IDSize):
            if (cohortID[z]).find(str(patientID[y])) != -1:
                temp = dataClinical.loc[dataClinical["sid"] == patientID[y]]
                cytokineData = pd.concat([temp, cytokineData])
    cytokineData.sort_values(by=["sid"], inplace=True)
    return cytokineData


def importCohort1Expression():
    """import expression data"""
    df = pd.read_table(join(path_here, "tfac/data/mrsa/expression_counts_cohort1.txt"))
    df.drop(["Chr", "Start", "End", "Strand", "Length"], inplace=True, axis=1)
    nodecimals = [val[:val.index(".")] for val in df["Geneid"]]
    df["Geneid"] = nodecimals
    df = df.set_index("Geneid")
    return df


def importCohort3Expression():
    '''Imports RNAseq data for cohort 3, sorted by patient ID.'''
    dfExp_c3 = pd.read_csv("tfac/data/mrsa/Genes_cohort3.csv")
    dfExp_c3 = dfExp_c3.set_index("Geneid")
    dfExp_c3 = dfExp_c3.reindex(sorted(dfExp_c3.columns), axis=1)
    return dfExp_c3


def removeC1_dupes(dfExp_c1):
    '''Removes duplicate genes from cohort 1 data. There are only a few (approx. 10) out of ~50,000, they are possibly different isoforms. The ones similar to cohort 3 are kept.'''
    dic = {}
    for id in dfExp_c1.index:
        if id in dic:
            dic[id] += 1
        else:
            dic[id] = 1
    mults = []
    for key, item in dic.items():
        if item > 1:
            mults.append(key)
    dfExp_c1 = dfExp_c1.reset_index()
    drop_dupes = [dfExp_c1[dfExp_c1["Geneid"] == mult].index[1] for mult in mults]
    dfExp_c1 = dfExp_c1.drop(drop_dupes)
    return dfExp_c1


def import_C3_cyto():
    '''Imports the cohort 3 cytokine data, giving separate dataframes for serum and plasma data. They overlap on many paitents.'''
    dfCyto_c3 = pd.read_csv("tfac/data/mrsa/CYTOKINES.csv")
    dfCyto_c3 = dfCyto_c3.set_index("sample ID")
    dfCyto_c3 = dfCyto_c3.rename_axis('sid')
    dfCyto_c3_serum = dfCyto_c3[dfCyto_c3["sample type"] == "serum"]
    dfCyto_c3_plasma = dfCyto_c3[dfCyto_c3["sample type"] == "plasma"]
    dfCyto_c3_serum.drop("sample type", axis=1, inplace=True)
    dfCyto_c3_plasma.drop("sample type", axis=1, inplace=True)
    return dfCyto_c3_serum, dfCyto_c3_plasma
