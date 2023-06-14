import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

from tfac.dataImport import get_cibersort_results, get_factors
from tfac.figures.common import getSetup


def main():
    cs_results = get_cibersort_results()
    t_fac, meta = get_factors()

    patient_factor = pd.DataFrame(
        t_fac.factors[0],
        index=meta.index,
        columns=np.arange(t_fac.factors[0].shape[1]) + 1
    )
    patient_factor = patient_factor.loc[cs_results.index, :]
    correlations = pd.DataFrame(
        index=cs_results.columns,
        columns=patient_factor.columns,
        dtype=float
    )

    for cell in cs_results.columns:
        for component in patient_factor.columns:
            # result = spearmanr(
            #     patient_factor.loc[:, component],
            #     cs_results.loc[:, cell]
            # )
            # correlations.loc[cell, component] = result.correlation
            corr, p = pearsonr(
                patient_factor.loc[:, component],
                cs_results.loc[:, cell]
            )
            correlations.loc[cell, component] = corr

    fig_size = (5, 5)
    layout = {
        'ncols': 1,
        'nrows': 1
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )
    ax = axs[0]

    sns.heatmap(
        correlations.astype(float),
        center=0,
        cmap='PRGn',
        ax=ax,
        cbar_kws={
            'label': 'Pearson Correlation'
        }
    )
    ax.set_yticks(np.arange(0.5, correlations.shape[0]))
    ax.set_yticklabels(correlations.index)

    ax.set_ylabel('Cell Type')
    ax.set_xlabel('Component')

    plt.show()


if __name__ == '__main__':
    main()
