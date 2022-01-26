from argparse import Namespace
import os
from os.path import abspath, dirname, exists, join
import sys

import gseapy as gp
from iterativeWGCNA.iterativeWGCNA import IterativeWGCNA
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import mygene
from natsort import natsorted
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale
from tensorpack import perform_CMTF

from .common import getSetup
from ..dataImport import form_tensor, import_rna

plt.rcParams["svg.fonttype"] = "none"

COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']
PATH_HERE = dirname(abspath(__file__))


def makeFigure():
    rna = import_rna()
    tensor, matrix, _ = form_tensor()
    t_fac = perform_CMTF(tensor, matrix)
    mod_expression = t_fac.mFactor

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
        cmap='vlag',
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
    ax.set_yticklabels(rna.columns, rotation=0)
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Module', fontsize=12)

    return fig
