"""Data import and processing for the MRSA data"""
from os.path import join, dirname
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

path_here = dirname(dirname(__file__))


def find_CV_decisions(patient_matrix, outcomes, n_splits=61, random_state=None, C=1):
    """Given the matrix of patients by components of a decomposition, returns the decision function results of logistic regression using L.O.O cross validation"""
    kf = KFold(n_splits=n_splits)
    decisions = []
    for train, test in kf.split(patient_matrix):
        clf = LogisticRegression(penalty='l1', solver='saga', C=C, random_state=random_state, max_iter=10000, fit_intercept=False).fit(patient_matrix[train], outcomes[train])
        decisions.append(clf.decision_function(patient_matrix[test]))
    score_y = decisions
    return score_y


def produce_outcome_bools(statusID):
    """Returns a list of booleans for progressor/resolver status ready to use for logistic regression"""
    outcome_bools = []
    for outcome in statusID:
        if outcome == 'APMB':
            outcome_bools.append(0)
        else:
            outcome_bools.append(1)

    return np.asarray(outcome_bools)


def get_patient_info():
    """Return specific patient ID information"""
    dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter="\t")
    cohortID = list(dataCohort["sample"])
    statusID = list(dataCohort["outcome_txt"])

    return cohortID, statusID


def form_MRSA_tensor(variance):
    """Create list of data matrices for parafac2"""
    dfClin, dfCoh = importClinicalMRSA()
    dfCyto = clinicalCyto(dfClin, dfCoh)
    dfCyto = dfCyto.sort_values(by="sid")
    dfCyto = dfCyto.set_index("sid")
    dfCyto = dfCyto.div(dfCyto.apply(gmean, axis=1).to_list(), axis=0)
    cytokines = dfCyto.columns

    dfExp = importExpressionData()
    geneIDs = dfExp["Geneid"].to_list()
    dfExp = dfExp.drop(["Geneid"], axis=1)
    ser = dfExp.var(axis=1)
    drops = []
    for idx, element in enumerate(ser):
        if not element:
            drops.append(idx)
    dfExp = dfExp.drop(drops)
    dfExp = (dfExp - dfExp.apply(np.mean)) / dfExp.apply(np.std)
    #dfExp = dfExp.sub(dfExp.apply(np.mean, axis=1).to_list(), axis=0)
    #dfExp = (dfExp.sub(dfExp.apply(np.mean, axis=1).to_list(), axis=0)).div(dfExp.apply(np.std, axis=1).to_list(), axis=0)

    cytoNumpy = dfCyto.to_numpy().T
    expNumpy = dfExp.to_numpy()

    expNumpy = expNumpy.astype(float)
    cytoNumpy = cytoNumpy * variance

    tensor_slices = [cytoNumpy, expNumpy]

    return tensor_slices, cytokines, geneIDs


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
