"""
This creates Figure 3 - Cytokine plots
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from .figureCommon import subplotLabel, getSetup
from ..dataImport import form_missing_tensor
from ..predict import SVC_predict


def fig_3_setup():
    tensor_slices, cytokines, _, patInfo = form_missing_tensor()
    test = pd.concat([pd.DataFrame(tensor_slices[0]), pd.DataFrame(tensor_slices[1])])
    test = test.dropna(axis=1)
    pears = []
    for i in range(38):
        pears.append([cytokines[i], pearsonr(test.iloc[i, :].to_numpy(dtype=float), test.iloc[i + 38, :].to_numpy(dtype=float))[0]])
    pears = pd.DataFrame(pears).sort_values(1)

    ser = pd.DataFrame(tensor_slices[0], index=cytokines, columns=patInfo.columns).iloc[:, :148].dropna(axis=1).T
    ser["Outcome"] = patInfo.loc["status"]
    out0 = ser[ser["Outcome"] == 0]
    out1 = ser[ser["Outcome"] == 1]
    ser = ser.append(pd.DataFrame(np.abs(out0.median() - out1.median()), columns=["diff"]).T).sort_values(by="diff", axis=1).drop("diff")
    ser = pd.melt(ser, id_vars=["Outcome"])

    plas = pd.DataFrame(tensor_slices[1], index=cytokines, columns=patInfo.columns).iloc[:, :148].dropna(axis=1).T
    plas["Outcome"] = patInfo.loc["status"]
    out0 = plas[plas["Outcome"] == 0]
    out1 = plas[plas["Outcome"] == 1]
    plas = plas.append(pd.DataFrame(np.abs(out0.median() - out1.median()), columns=["diff"]).T).sort_values(by="diff", axis=1).drop("diff")
    plas = pd.melt(plas, id_vars=["Outcome"])

    return pears, ser, plas


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 15), (3, 1))
    pears, ser, plas = fig_3_setup()
    a = sns.pointplot(data=pears, x=0, y=1, join=False, ax=ax[0])
    a.set_xticklabels(a.get_xticklabels(), rotation=45, fontsize=12, ha="right")
    b = sns.boxplot(data=ser, x="variable", y="value", hue="Outcome", ax=ax[1])
    b.set_xticklabels(b.get_xticklabels(), rotation=45, fontsize=12, ha="right")
    c = sns.boxplot(data=plas, x="variable", y="value", hue="Outcome", ax=ax[2])
    c.set_xticklabels(c.get_xticklabels(), rotation=45, fontsize=12, ha="right")
    # Add subplot labels
    subplotLabel(ax)

    return f
