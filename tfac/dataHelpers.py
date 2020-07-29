"""Contains function for importing and handling OHSU data"""
from os.path import join, dirname
import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))


def importLINCSprotein():
    """ Import protein characterization from LINCS. """
    dataA = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_A.csv"))
    dataB = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_B.csv"))
    dataC = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_C.csv"))

    dataA["File"] = "A"
    dataB["File"] = "B"
    dataC["File"] = "C"

    return pd.concat([dataA, dataB, dataC])


def ohsu_data():
    """ Import OHSU data for PARAFAC2"""
    atac = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_ATACseq_Level4.csv"))
    cycIF = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_cycIF_Level4.csv"))
    GCP = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_GCP_Level4.csv"))
    IF = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_IF_Level4.csv"))
    L1000 = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_L1000_Level4.csv"))
    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RNAseq_Level4.csv"))
    RPPA = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RPPA_Level4.csv"))
    return atac, cycIF, GCP, IF, L1000, RNAseq, RPPA


def compProteins(comps):
    """Returns the top three weighted proteins for each component in input protein component matrix"""
    i = np.shape(comps)  # input tensor decomp output
    proteins = proteinNames()
    _, compNum = np.shape(comps[i[0] - 1])
    compName = []
    topProtein = []

    for x in range(0, compNum):
        compName.append("Col" + str(x + 1))

    dfComps = pd.DataFrame(data=comps[i[0] - 1], index=proteins, columns=compName)
    for y in range(0, compNum):
        topProtein.append(compName[y])
        rearranged = dfComps.sort_values(by=compName[y], ascending=False)
        rearrangedNames = list(rearranged.index.values)
        for z in range(0, 3):
            topProtein.append(rearrangedNames[z])

    return topProtein


def proteinNames():
    """Return protein names (data columns)"""
    data = importLINCSprotein()
    data = data.drop(columns=["Treatment", "Sample description", "File", "Time"], axis=1)
    proteinN = data.columns.values.tolist()
    return proteinN


def printOutliers(results):
    """Prints most extremem protein outliers of partial tucker decomposition of OHSU data based on IQR"""
    df = pd.DataFrame(results[1][0])
    proteins = importLINCSprotein()
    columns = proteins.columns[3:298]
    df["Proteins"] = columns
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    prots = {}
    for i in range(df.columns.size - 1):
        print("Component", str(i + 1), "1.5*IQR:", np.round((Q1[i] - 1.5 * IQR[i]), 2), np.round((Q3[i] + 1.5 * IQR[i]), 2))
        positives = []
        negatives = []
        for _, col in df.iterrows():
            if col[i] < (Q1[i] - 1.5 * IQR[i]):
                negatives.append((col[i], col["Proteins"]))
                if col["Proteins"] not in prots:
                    prots[col["Proteins"]] = 1
                else:
                    prots[col["Proteins"]] += 1
            elif col[i] > (Q3[i] + 1.5 * IQR[i]):
                positives.append((col[i], col["Proteins"]))
                if col["Proteins"] not in prots:
                    prots[col["Proteins"]] = 1
                else:
                    prots[col["Proteins"]] += 1
        print()
        negatives = sorted(negatives)[:7]
        positives = sorted(positives)[-7:]
        for tup in positives:
            print(tup[1])
        for tup in positives:
            print(np.round(tup[0], 2))
        print()
        for tup in negatives:
            print(tup[1])
        for tup in negatives:
            print(np.round(tup[0], 2))
        print()
    print(prots)
