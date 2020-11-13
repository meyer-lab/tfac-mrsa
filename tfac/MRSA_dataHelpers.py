"""Data import and processing for the MRSA data"""
from os.path import join, dirname
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
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


def get_C1_patient_info(paired=False):
    """Return specific patient ID information"""
    if paired:
        dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter="\t")
        singles = ["SA04233", "SA04158", "SA04255", "SA04378", "SA04469", "SA04329", "SA05300", "SA04547", "SA05030"]
        dataCohort = dataCohort[~dataCohort["sample"].isin(singles)]
        cohortID = list(dataCohort["sample"])
        statusID = list(dataCohort["outcome_txt"])
        return cohortID, statusID

    dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter="\t")
    cohortID = list(dataCohort["sample"])
    statusID = list(dataCohort["outcome_txt"])
    type_ID = list(dataCohort["sampletype"])
    return cohortID, statusID, type_ID


def get_C3_patient_info(sample_type):
    dfExp_c3 = pd.read_csv("tfac/data/mrsa/Genes_cohort3.csv")
    dfExp_c3 = dfExp_c3.set_index("Geneid")
    dfExp_c3 = dfExp_c3.drop([patient for patient in dfExp_c3.columns if len(patient) > 4], axis=1)
    dfExp_c3 = dfExp_c3.reindex(sorted(dfExp_c3.columns), axis=1)
    cohortID = list(dfExp_c3.columns)
    if sample_type == 'serum':
        return cohortID
    elif sample_type == 'plasma':
        cohortID.remove('7008')
        return cohortID
    else:
        raise ValueError("Bad sample type selection.")


def form_MRSA_tensor(sample_type, variance1=1, variance2=1):
    """Create list of data matrices for parafac2"""
    #import cytokines and format
    dfClin, dfCoh = importClinicalMRSA()
    dfCyto_c1 = clinicalCyto(dfClin, dfCoh)
    dfCyto_c1 = dfCyto_c1.set_index("sid")
    dfCyto_c3_serum, dfCyto_c3_plasma = import_C3_cyto()
    dfCyto_c1.columns = dfCyto_c3_serum.columns
    #normalize separately and extract cytokines
    cyto_list = [dfCyto_c1, dfCyto_c3_serum, dfCyto_c3_plasma]
    for idx, df in enumerate(cyto_list):
        df = df.div(df.apply(gmean, axis=1).to_list(), axis=0)
        df = df.apply(np.log, axis=0)
        df = df.sub(df.apply(np.mean, axis=0).to_list(), axis=1)
        cyto_list[idx] = df
    cytokines = dfCyto_c1.columns

    dfExp_c1 = importCohort1Expression()
    dfExp_c3 = importCohort3Expression()
    #Drop genes not shared
    dfExp_c1 = dfExp_c1.drop([gene for gene in dfExp_c1.index if gene not in dfExp_c3.index])
    dfExp_c3 = dfExp_c3.drop([gene for gene in dfExp_c3.index if gene not in dfExp_c1.index])
    #Filter out duplicate genes from c1 - choosing to keep those most similar to c3
    dfExp_c1 = removeC1_dupes(dfExp_c1)
    #Extract Gene IDs and normalize
    geneIDs = dfExp_c1["Geneid"].to_list()
    dfExp_c1 = dfExp_c1.set_index("Geneid")
    dfExp_c1.sort_values("Geneid", inplace=True)
    dfExp_c3.sort_values("Geneid", inplace=True)
    dfExp_c1 = (dfExp_c1 - dfExp_c1.apply(np.mean)) / dfExp_c1.apply(np.std)
    dfExp_c1 = (dfExp_c1.sub(dfExp_c1.apply(np.mean, axis=1).to_list(), axis=0)).div(dfExp_c1.apply(np.std, axis=1).to_list(), axis=0)
    dfExp_c3 = (dfExp_c3 - dfExp_c3.apply(np.mean)) / dfExp_c3.apply(np.std)
    dfExp_c3 = (dfExp_c3.sub(dfExp_c3.apply(np.mean, axis=1).to_list(), axis=0)).div(dfExp_c3.apply(np.std, axis=1).to_list(), axis=0)
    dfExp = pd.concat([dfExp_c1, dfExp_c3], axis=1)

    if sample_type == 'serum':
        dfCyto_serum = pd.concat([cyto_list[0], cyto_list[1]])
        cytoNumpy = dfCyto_serum.to_numpy().T
        cytoNumpy = cytoNumpy * ((1 / tl_var(cytoNumpy)) ** 0.5) * variance1
        expNumpy = dfExp.to_numpy()
        expNumpy = expNumpy.astype(float)
        expNumpy = expNumpy * variance2
        tensor_slices = [cytoNumpy, expNumpy]
    elif sample_type == 'plasma':
        dfCyto_plasma = pd.concat([cyto_list[0], cyto_list[2]])
        cytoNumpy = dfCyto_plasma.to_numpy().T
        cytoNumpy = cytoNumpy * ((1 / tl_var(cytoNumpy)) ** 0.5) * variance1
        dfExp = dfExp.drop('7008', axis=1)
        expNumpy = dfExp.to_numpy()
        expNumpy = expNumpy.astype(float)
        expNumpy = expNumpy * variance2
        tensor_slices = [cytoNumpy, expNumpy]
    else:
        raise ValueError("Bad sample type selection.")

    return tensor_slices, cytokines, geneIDs


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
    dfExp_c3 = pd.read_csv("tfac/data/mrsa/Genes_cohort3.csv")
    dfExp_c3 = dfExp_c3.set_index("Geneid")
    dfExp_c3 = dfExp_c3.drop([patient for patient in dfExp_c3.columns if len(patient) > 4], axis=1)
    dfExp_c3 = dfExp_c3.reindex(sorted(dfExp_c3.columns), axis=1)
    return dfExp_c3


def removeC1_dupes(dfExp_c1):
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
    dfCyto_c3 = pd.read_csv("tfac/data/mrsa/CYTOKINES.csv")
    dfCyto_c3 = dfCyto_c3.set_index("sample ID")
    patientID = get_C3_patient_info(sample_type='serum')
    dfCyto_c3 = dfCyto_c3.drop([patient for patient in dfCyto_c3.index if str(patient) not in patientID])
    dfCyto_c3 = dfCyto_c3.rename_axis('sid')
    dfCyto_c3_serum = dfCyto_c3[dfCyto_c3["sample type"] == "serum"].drop("sample type", axis=1)
    dfCyto_c3_plasma = dfCyto_c3[dfCyto_c3["sample type"] == "plasma"].drop("sample type", axis=1)
    return dfCyto_c3_serum, dfCyto_c3_plasma
