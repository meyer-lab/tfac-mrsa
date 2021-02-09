"""
This creates Figure 5 - SVC visualization.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from .figureCommon import subplotLabel, getSetup
from ..dataImport import get_patient_info, produce_outcome_bools


def fig_5_setup():
    patient_matrices, _, _, _ = pickle.load(open("MRSA_pickle.p", "rb"))
    _, statusID = get_patient_info()
    outcomes = produce_outcome_bools(statusID)
    cytoA = patient_matrices[1][2].T[8]
    cytoB = patient_matrices[1][2].T[32]
    cyto_df = pd.DataFrame([cytoA, cytoB, outcomes]).T
    cyto_df.columns = ["Component A", "Component B", "Outcomes"]
    double = np.vstack((cytoA, cytoB)).T
    clf = SVC()
    clf.fit(double, outcomes)

    return cyto_df, clf


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    df, func = fig_5_setup()
    # Get list of axis objects
    ax, f = getSetup((15, 8), (1, 1))
    b = sns.scatterplot(data=df, x="Component A", y="Component B", hue="Outcomes", ax=ax[0])  # blue
    b.set_xlabel("Component A", fontsize=20)
    b.set_ylabel("Component B", fontsize=20)
    b.tick_params(labelsize=14)

    plot_svc_decision_function(func, ax=ax[0])

    # Add subplot labels
    subplotLabel(ax)

    return f


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, facecolors="none")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
