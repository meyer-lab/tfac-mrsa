"""
This creates Figure 3. - Components vs AUC
"""
import pickle
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import get_patient_info, produce_outcome_bools

_, outcomeID = get_patient_info()
true_y = produce_outcome_bools(outcomeID)
all_vars = pickle.load(open("common_decomp.p", "rb"))


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 20), (3, 1))
    for idx, var in enumerate([.0007, .007, .07]):
        b = sns.pointplot(data=all_vars[idx], x='Components', y='AUC', ax=ax[idx], s=70, join=False) # blue
        b.set_xlabel("Components", fontsize=20)
        b.set_ylabel("AUC", fontsize=20)
        b.set_title("AUC with Cytokine Variance Scaling = " + str(var), fontsize=20)
        b.tick_params(labelsize=14)
        ax[idx].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
