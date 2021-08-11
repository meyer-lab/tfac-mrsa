"""
This creates Figure 3 - Cytokine plots
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from tfac.dataImport import form_tensor, import_cytokines
from .figureCommon import getSetup


def serum_vs_plasma_setup():
    tensor, _, patInfo = form_tensor()
    plasma, _ = import_cytokines()
    cytokines = plasma.index

    patInfo = patInfo.T
    tensor = tensor.T
    serum_slice = tensor[0, :, :]
    plasma_slice = tensor[1, :, :]

    test = pd.concat([pd.DataFrame(serum_slice), pd.DataFrame(plasma_slice)])
    test = test.dropna(axis=1)
    pears = []
    for i in range(38):
        pears.append([cytokines[i], pearsonr(test.iloc[i, :].to_numpy(dtype=float), test.iloc[i + 38, :].to_numpy(dtype=float))[0]])
    pears = pd.DataFrame(pears).sort_values(1)

    ser = pd.DataFrame(serum_slice, index=cytokines, columns=patInfo.columns).T
    ser["Outcome"] = patInfo.T["status"]
    ser = ser[ser["Outcome"].isin(['0', '1'])].dropna().astype(float)
    plas = pd.DataFrame(plasma_slice, index=cytokines, columns=patInfo.columns).T
    plas["Outcome"] = patInfo.T["status"]
    plas = plas[plas["Outcome"].isin(['0', '1'])].dropna().astype(float)
    plas["Outcome"] += 2
    cyto = pd.concat([ser, plas])
    out0 = cyto[cyto["Outcome"] == 0]
    out1 = cyto[cyto["Outcome"] == 1]
    out2 = cyto[cyto["Outcome"] == 2]
    out3 = cyto[cyto["Outcome"] == 3]
    cyto = cyto.append(pd.DataFrame(np.abs(out0.median() - out1.median()) + np.abs(out2.median() - out3.median()), columns=["diff"]).T).sort_values(by="diff", axis=1).drop("diff")
    cyto = pd.melt(cyto, id_vars=["Outcome"])

    return pears, cyto


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((5, 5), (2, 1))
    pears, cyto = serum_vs_plasma_setup()
    kines = ["IL-10", "VEGF", "MCP-3", "IL-4", "Fractalkine"]
    pears = pears[pears[0].isin(kines)]
    cyto = cyto[cyto["variable"].isin(kines)]
    a = sns.pointplot(data=pears, x=0, y=1, join=False, ax=ax[0])
    a.set_xlabel("Cytokine")
    a.set_ylabel("Pearson's correlation")
    a.set_title("Serum-Plasma Cytokine Level Correlation")
    a.set_ylim(0.0, 1.0)
    b = sns.boxplot(data=cyto, x="variable", y="value", hue="Outcome", ax=ax[1])
    handles, _ = b.get_legend_handles_labels()
    b.legend(handles, ["Serum - Resolved", "Serum - Persisted", "Plasma - Resolved", "Plasma - Persisted"])
    b.set_xlabel("Cytokine")
    b.set_ylabel("Normalized cytokine level")
    b.set_title("Normalized Serum Cytokine Level by Outcome")

    return f
