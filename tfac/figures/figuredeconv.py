"""
This creates Figure 3 - Deconvolution-Component Correlation
"""
from os.path import join, dirname
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from tfac.figures.figureCommon import subplotLabel, getSetup
from tfac.dataImport import form_missing_tensor
from tfac.tensor import perform_TMTF

path_here = dirname(dirname(__file__))

def fig_3_setup():
    tensor_slices, _, _, patInfo = form_missing_tensor()
    tensor = np.stack((tensor_slices[0], tensor_slices[1])).T
    matrix = tensor_slices[2].T
    component = 5
    Decomp = perform_TMTF(tensor, matrix, r=component)

    missing = [4721, 5030, 5101]

    facs = Decomp[0].factors[0]
    facs = pd.DataFrame(facs, index=patInfo.columns, columns=["Component " + str(x + 1) for x in range(facs.shape[1])])
    facs = pd.concat([patInfo, facs.T]).T
    facs = facs[facs["type"].str.contains("2RNAseq")].sort_index().drop(missing)
    facs = facs.drop(["type", "status", "cohort"], axis=1)
    dec1 = pd.read_csv(join(path_here, "tfac/data/mrsa/deconvo_cibersort_cohort3.csv", delimiter=",", index_col="cell_type")).T
    dec3 = pd.read_csv(join(path_here, "tfac/data/mrsa/deconvo_cibersort_APMB.csv", delimiter=",", index_col="sample")).sort_index().drop(["gender", "cell_type"], axis=1)
    deconv = pd.concat([dec1, dec3]).sort_index()
    plotable = pd.DataFrame(columns=["CellType", "Pearson's R", "Component"])
    for cell in deconv.columns:
        for comp in facs.columns:
            plotable = plotable.append({"CellType": type, "Pearson's R": pearsonr(deconv[cell], facs[comp])[0], "Component": comp}, ignore_index=True)
    return plotable


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    plotable = fig_3_setup()
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))
    a = sns.pointplot(data=plotable, x="CellType", y="Pearson's R", hue="Component", join=False, ax=ax[0])
    a.set_xticklabels(a.get_xticklabels(), rotation=45, ha="right")
    a.set_ylim(-1, 1)
    # Add subplot labels
    subplotLabel(ax)

    return f
