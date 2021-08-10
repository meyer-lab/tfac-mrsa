"""
This creates Figure S1 - Full Cytokine plots
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from tfac.dataImport import form_tensor, import_cytokines
from .figureCommon import getSetup


def fig_S1_setup():
    tensor, matrix, patInfo = form_tensor()
    plasma, _ = import_cytokines()
    cytokines = plasma.index

    tensor = tensor.T
    patInfo = patInfo.T
    serum_slice = tensor[0, :, :]
    plasma_slice = tensor[1, :, :]
    
    test = pd.concat([pd.DataFrame(serum_slice), pd.DataFrame(plasma_slice)])
    test = test.dropna(axis=1)
    pears = []
    for i in range(38):
        pears.append([cytokines[i], pearsonr(test.iloc[i, :].to_numpy(dtype=float), test.iloc[i + 38, :].to_numpy(dtype=float))[0]])
    pears = pd.DataFrame(pears).sort_values(1)

    ser = pd.DataFrame(serum_slice, index=cytokines, columns=patInfo.columns).iloc[:, :148].dropna(axis=1).T
    ser["Outcome"] = patInfo.loc["status"]
    out0 = ser[ser["Outcome"] == 0]
    out1 = ser[ser["Outcome"] == 1]
    ser = ser.append(pd.DataFrame(np.abs(out0.median() - out1.median()), columns=["diff"]).T).sort_values(by="diff", axis=1).drop("diff")
    ser = pd.melt(ser, id_vars=["Outcome"])

    plas = pd.DataFrame(plasma_slice, index=cytokines, columns=patInfo.columns).iloc[:, :148].dropna(axis=1).T
    plas["Outcome"] = patInfo.loc["status"]
    out0 = plas[plas["Outcome"] == 0]
    out1 = plas[plas["Outcome"] == 1]
    plas = plas.append(pd.DataFrame(np.abs(out0.median() - out1.median()), columns=["diff"]).T).sort_values(by="diff", axis=1).drop("diff")
    plas = pd.melt(plas, id_vars=["Outcome"])

    return pears, ser, plas


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (3, 1))
    pears, ser, plas = fig_S1_setup()
    a = sns.pointplot(data=pears, x=0, y=1, join=False, ax=ax[0])
    a.set_xticklabels(a.get_xticklabels(), rotation=30, ha="right")
    a.set_xlabel("Cytokine")
    a.set_ylabel("Pearson's correlation")
    a.set_title("Serum-Plasma Cytokine Level Correlation")
    b = sns.boxplot(data=ser, x="variable", y="value", hue="Outcome", ax=ax[1])
    handles, _ = b.get_legend_handles_labels()
    b.legend(handles, ["Resolved", "Persisted"])
    b.set_xticklabels(b.get_xticklabels(), rotation=30, ha="right")
    b.set_xlabel("Cytokine")
    b.set_ylabel("Normalized cytokine level")
    b.set_title("Normalized Serum Cytokine Level by Outcome")
    c = sns.boxplot(data=plas, x="variable", y="value", hue="Outcome", ax=ax[2])
    handles, _ = c.get_legend_handles_labels()
    c.legend(handles, ["Resolved", "Persisted"])
    c.set_xticklabels(c.get_xticklabels(), rotation=30, ha="right")
    c.set_xlabel("Cytokine")
    c.set_ylabel("Normalized cytokine level")
    c.set_title("Normalized Plasma Cytokine Level by Outcome")

    return f
