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


def get_patient_info(paired=False):
    """Return specific patient ID information"""
    if paired:
        dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter="\t")
        singles = ["SA04233", "SA04158", "SA04255", "SA04378", "SA04469", "SA04329", "SA05300", "SA04547", "SA05030"]
        dataCohort = dataCohort[~dataCohort["sample"].isin(singles)]
        cohortID = list(dataCohort["sample"])
        statusID = list(dataCohort["outcome_txt"])
        return cohortID, statusID

    dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter="\t")
    singles = ["SA04233"]
    dataCohort = dataCohort[~dataCohort["sample"].isin(singles)]
    cohortID = list(dataCohort["sample"])
    statusID = list(dataCohort["outcome_txt"])
    return cohortID, statusID


def form_paired_tensor(variance1=1, variance2=1):
    """Create list of data matrices of paired data for parafac2"""
    dfClin, dfCoh = importClinicalMRSA()
    singles = [4, 7, 14, 19, 24, 25, 29, 31]
    remove = dfCoh[dfCoh["pair"].isin(singles)]["sample"].to_list()
    dfCoh = dfCoh[~dfCoh["pair"].isin(singles)]
    pairs = dfCoh["pair"]
    dfCyto = clinicalCyto(dfClin, dfCoh)
    dfCyto = dfCyto.sort_values(by="sid")
    dfCyto = dfCyto.set_index("sid")
    dfCyto = dfCyto.div(dfCyto.apply(gmean, axis=1).to_list(), axis=0)
    dfCyto = dfCyto.apply(np.log, axis=0)
    dfCyto = dfCyto.sub(dfCyto.apply(np.mean, axis=0).to_list(), axis=1)
    cytokines = dfCyto.columns

    dfExp = importExpressionData()
    dfExp = dfExp.drop(remove, axis=1)
    ser = dfExp.var(axis=1)
    drops = []
    for idx, element in enumerate(ser):
        if not element:
            drops.append(idx)
    dfExp = dfExp.drop(drops)
    geneIDs = dfExp["Geneid"].to_list()
    dfExp = dfExp.drop(["Geneid"], axis=1)
    dfExp = (dfExp - dfExp.apply(np.mean)) / dfExp.apply(np.std)
    dfExp = (dfExp.sub(dfExp.apply(np.mean, axis=1).to_list(), axis=0)).div(dfExp.apply(np.std, axis=1).to_list(), axis=0)

    # dataMeth, m_locations = import_methylation()
    # remove = ["4158", "4255", "4378", "4469", "4329", "5300", "4547", "5030"]
    # dataMeth = dataMeth.drop(remove, axis=1)

    cytoNumpy = dfCyto.to_numpy().T
    expNumpy = dfExp.to_numpy()
    # methNumpy = dataMeth.iloc[:, 1:].values

    # methNumpy = methNumpy.astype(float)
    expNumpy = expNumpy.astype(float)
    cytoNumpy = cytoNumpy * ((1 / tl_var(cytoNumpy)) ** 0.5) * variance1
    expNumpy = expNumpy * variance2
    # methNumpy = methNumpy * ((1 / tl_var(methNumpy)) ** .5) * variance3

    tensor_slices = [cytoNumpy, expNumpy]  # , methNumpy]

    return tensor_slices, cytokines, geneIDs, pairs


def form_MRSA_tensor(variance1=1, variance2=1):
    """Create list of data matrices for parafac2"""
    dfClin, dfCoh = importClinicalMRSA()
    dfCyto = clinicalCyto(dfClin, dfCoh)
    dfCyto = dfCyto.sort_values(by="sid")
    dfCyto = dfCyto.set_index("sid")
    dfCyto = dfCyto.drop(4233)
    dfCyto = dfCyto.div(dfCyto.apply(gmean, axis=1).to_list(), axis=0)
    dfCyto = dfCyto.apply(np.log, axis=0)
    dfCyto = dfCyto.sub(dfCyto.apply(np.mean, axis=0).to_list(), axis=1)
    cytokines = dfCyto.columns

    dfExp = importExpressionData()
    dfExp = dfExp.drop(["SA04233"], axis=1)
    ser = dfExp.var(axis=1)
    drops = []
    for idx, element in enumerate(ser):
        if not element:
            drops.append(idx)
    dfExp = dfExp.drop(drops)
    geneIDs = dfExp["Geneid"].to_list()
    dfExp = dfExp.drop(["Geneid"], axis=1)
    dfExp = (dfExp - dfExp.apply(np.mean)) / dfExp.apply(np.std)
    dfExp = (dfExp.sub(dfExp.apply(np.mean, axis=1).to_list(), axis=0)).div(dfExp.apply(np.std, axis=1).to_list(), axis=0)

    # dataMeth, m_locations = import_methylation()

    cytoNumpy = dfCyto.to_numpy().T
    expNumpy = dfExp.to_numpy()
    # methNumpy = dataMeth.iloc[:, 1:].values

    # methNumpy = methNumpy.astype(float)
    expNumpy = expNumpy.astype(float)
    cytoNumpy = cytoNumpy * ((1 / tl_var(cytoNumpy)) ** 0.5) * variance1
    expNumpy = expNumpy * variance2
    # methNumpy = methNumpy * ((1 / tl_var(methNumpy)) ** .5) * variance3

    tensor_slices = [cytoNumpy, expNumpy]  # , methNumpy]

    return tensor_slices, cytokines, geneIDs  # , m_locations


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
    cytokineData.sort_values(by=["sid"])
    return cytokineData


def importExpressionData():
    """import expression data"""
    df = pd.read_table(join(path_here, "tfac/data/mrsa/expression_counts_cohort1.txt"))
    df.drop(["Chr", "Start", "End", "Strand", "Length"], inplace=True, axis=1)
    return df
