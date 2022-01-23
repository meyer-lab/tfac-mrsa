"""
This creates Figure S1 - Full Cytokine plots
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

from .common import getSetup
from ..dataImport import form_tensor, import_cytokines


def fig_S1_setup():
    tensor, _, patInfo = form_tensor()
    plasma, _ = import_cytokines()
    cytokines = plasma.index
    n_cytokines = len(cytokines)

    tensor = tensor.T
    patInfo = patInfo.T
    serum_slice = tensor[0, :, :]
    plasma_slice = tensor[1, :, :]

    test = pd.concat([pd.DataFrame(serum_slice), pd.DataFrame(plasma_slice)])
    test = test.dropna(axis=1)
    pears = []
    for i in range(n_cytokines):
        pears.append([cytokines[i], pearsonr(test.iloc[i, :].to_numpy(dtype=float), test.iloc[i + n_cytokines, :].to_numpy(dtype=float))[0]])
    pears = pd.DataFrame(pears).sort_values(1)

    return pears, serum_slice, plasma_slice, cytokines, patInfo


def cytokine_boxplot(cyto_slice, cytokines, patInfo, axx):
    ser = pd.DataFrame(cyto_slice, index=cytokines, columns=patInfo.columns).dropna(axis=1).T
    ser["Outcome"] = patInfo.loc["status"]
    out0 = ser[ser["Outcome"] == 0]
    out1 = ser[ser["Outcome"] == 1]
    ser = ser.append(pd.DataFrame(np.abs(out0.median() - out1.median()), columns=["diff"]).T).sort_values(by="diff", axis=1).drop("diff")
    ser = pd.melt(ser, id_vars=["Outcome"])

    b = sns.boxplot(data=ser, x="variable", y="value", hue="Outcome", ax=axx)
    handles, _ = b.get_legend_handles_labels()
    b.legend(handles, ["Resolved", "Persisted"])
    b.set_xticklabels(b.get_xticklabels(), rotation=30, ha="right")
    b.set_xlabel("Cytokine")
    b.set_ylabel("Normalized cytokine level")


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    fig_size = (8, 8)
    layout = {
        'ncols': 1,
        'nrows': 3
    }
    ax, f, _ = getSetup(
        fig_size,
        layout
    )
    pears, serum_slice, plasma_slice, cytokines, patInfo = fig_S1_setup()
    a = sns.pointplot(data=pears, x=0, y=1, join=False, ax=ax[0])
    a.set_xticklabels(a.get_xticklabels(), rotation=30, ha="right")
    a.set_xlabel("Cytokine")
    a.set_ylabel("Pearson's correlation")
    a.set_title("Serum-Plasma Cytokine Level Correlation")

    cytokine_boxplot(serum_slice, cytokines, patInfo, ax[1])
    ax[1].set_title("Normalized Serum Cytokine Level by Outcome")

    cytokine_boxplot(plasma_slice, cytokines, patInfo, ax[2])
    ax[2].set_title("Normalized Plasma Cytokine Level by Outcome")

    return f
