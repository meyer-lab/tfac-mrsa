import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

from tfac.dataImport import get_cibersort_results, get_factors, reorder_table
from tfac.figures.common import getSetup


def main():
    cs_results = get_cibersort_results(citeseq=True)
    t_fac, _, meta = get_factors()

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
    p_values = correlations.copy(deep=True)

    for cell in cs_results.columns:
        for component in patient_factor.columns:
            corr, p = pearsonr(
                patient_factor.loc[:, component],
                cs_results.loc[:, cell]
            )
            correlations.loc[cell, component] = corr
            p_values.loc[cell, component] = p

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

    correlations = correlations.dropna(axis=0)
    correlations = reorder_table(correlations)
    sns.heatmap(
        correlations.astype(float),
        center=0,
        cmap='coolwarm',
        ax=ax,
        cbar_kws={
            'label': 'Pearson Correlation'
        }
    )

    ax.set_yticks(np.arange(0.5, correlations.shape[0]))
    ax.set_yticklabels(correlations.index)

    for xtick, comp in zip(ax.get_xticks(), correlations.columns):
        for ytick, cell in zip(ax.get_yticks(), correlations.index):
            if p_values.loc[cell, comp] < 0.05:
                ax.text(
                    xtick,
                    ytick + 0.25,
                    s='*',
                    ha='center',
                    va='center',
                    ma='center'
                )

    ax.set_ylabel('Cell Type')
    ax.set_xlabel('Component')

    plt.show()


if __name__ == '__main__':
    main()
