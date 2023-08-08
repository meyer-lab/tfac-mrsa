"""
This creates Figure S3 - RNA-Seq Associations to Components
"""
from os.path import abspath, dirname

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import getSetup
from ..dataImport import get_factors, import_rna, reorder_table

plt.rcParams["svg.fonttype"] = "none"

COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']
PATH_HERE = dirname(abspath(__file__))


def makeFigure():
    rna = import_rna()
    t_fac, _, _ = get_factors()
    mod_expression = pd.DataFrame(
        t_fac.mFactor,
        index=rna.columns,
        columns=range(1, 9)
    )
    mod_expression = reorder_table(mod_expression)

    fig_size = (4, 8)
    layout = {
        'ncols': 1,
        'nrows': 1
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )
    ax = axs[0]

    bound = abs(mod_expression).max().max()
    sns.heatmap(
        mod_expression.astype(float),
        center=0,
        cmap='PRGn',
        vmax=bound,
        vmin=-bound,
        cbar_kws={
            'label': 'Component Association'
        },
        ax=ax
    )

    ax.set_xticklabels(range(1, mod_expression.shape[1] + 1))
    ax.set_yticks(
        np.arange(
            0.5,
            mod_expression.shape[0]
        ),
    )
    ax.set_yticklabels(mod_expression.index, rotation=0)
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Module', fontsize=12)

    return fig
