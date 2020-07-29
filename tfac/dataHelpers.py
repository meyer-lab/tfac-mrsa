"""Contains function for importing and handling OHSU data"""
from os.path import dirname
import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))


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
