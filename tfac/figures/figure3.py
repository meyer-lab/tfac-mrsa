"""
This creates Figure 3 - Individual Data performance
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..predict import SVC_predict

def fig_3_setup():
    return


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))
    # Add subplot labels
    subplotLabel(ax)

    return f
