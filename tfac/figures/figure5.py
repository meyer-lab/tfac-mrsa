"""
Creates Figure 5 -- Reduced Model
"""
import datashader as ds
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
from os.path import abspath, dirname
import pandas as pd
from sklearn.metrics import roc_curve

from .common import getSetup
from ..dataImport import import_validation_patient_metadata, get_factors, \
    import_cytokines
from ..predict import get_accuracy, predict_known

SCATTER_COLORS = {
    '2': 'red',
    '4': 'cyan',
    '6': 'green',
    'shared': 'black',
    'neither': '#D3D3D3'
}
COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
PATH_HERE = dirname(dirname(abspath(__file__)))
PERSISTENCE_COMPONENTS = [1, 2, 4, 6]


def run_cv(components, patient_data, svc=False):
    """
    Predicts samples with known outcomes via cross-validation.

    Parameters:
        components (numpy.array): CMTF components
        patient_data (pandas.DataFrame): patient metadata
        svc (bool, default: True): use svc for classification

    Returns:
        predictions (pandas.Series): predictions for each data source
    """
    labels = patient_data.loc[components.index, 'status'].astype(int)

    predictions = pd.DataFrame(
        index=patient_data.index
    )
    probabilities = predictions.copy()

    predictions.loc[:, 'Full'], _ = predict_known(components, labels, svc=svc)
    probabilities.loc[:, 'Full'], _ = predict_known(
        components,
        labels,
        method='predict_proba',
        svc=svc
    )

    predictions.loc[:, '1, 2, 4 & 6'], _ = predict_known(
        components.loc[:, PERSISTENCE_COMPONENTS],
        labels,
        svc=svc
    )
    probabilities.loc[:, '1, 2, 4 & 6'], _ = predict_known(
        components.loc[:, PERSISTENCE_COMPONENTS],
        labels,
        svc=svc,
        method='predict_proba'
    )

    summed = pd.concat(
        [
            components.loc[:, 2],
            components.loc[:, [4, 6]].sum(axis=1)
        ],
        axis=1
    )

    predictions.loc[:, '2, 4 + 6'], model = predict_known(
        summed,
        labels,
        svc=svc
    )
    probabilities.loc[:, '2, 4 + 6'], _ = predict_known(
        summed,
        labels,
        svc=svc,
        method='predict_proba'
    )

    for i in np.arange(len(PERSISTENCE_COMPONENTS)):
        reduced = PERSISTENCE_COMPONENTS[:i] + PERSISTENCE_COMPONENTS[i+1:]
        name = ', '.join([str(i) for i in reduced[:-1]]) + f' & {reduced[-1]}'
        predictions[name], _ = predict_known(
            components.loc[:,  reduced],
            labels,
            svc=svc
        )
        probabilities[name], _ = \
            predict_known(
                components.loc[:, reduced],
                labels,
                method='predict_proba',
                svc=svc
            )

    predictions.loc[:, 'Actual'] = patient_data.loc[:, 'status']

    return predictions.astype(int), probabilities, model


def get_accuracies(samples):
    """
    Calculates prediction accuracy for samples with known outcomes.

    Parameters:
        samples (pandas.DataFrame): predictions for different models

    Returns:
        accuracies (pandas.Series): model accuracy w/r to each model type
    """
    actual = samples.loc[:, 'Actual']
    samples = samples.drop('Actual', axis=1)

    d_types = samples.columns
    accuracies = pd.Series(
        index=d_types,
        dtype=float
    )

    for d_type in d_types:
        col = samples[d_type]
        accuracies[d_type] = get_accuracy(col, actual)

    return accuracies


def plot_results(predictions, probabilities, model, components,
                 t_fac, patient_data):
    """
    Plots prediction model performance.

    Parameters:
        predictions (pandas.DataFrame): predictions for training samples
        probabilities (pandas.DataFrame): predicted probability of
            persistence for training samples
        model (sklearn.SVM): Best reduced SVM model
        components (pandas.DataFrame): CMTF components
        t_fac (CPTensor): CMTF factorization result
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        fig (matplotlib.Figure): figure depicting predictions for all samples
    """
    fig_size = (6, 6)
    layout = {
        'ncols': 3,
        'nrows': 3,
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )

    # Cross-validation Accuracies

    model_axs = axs[:3]
    lr_accuracies = get_accuracies(predictions)
    model_axs[0].bar(
        np.arange(len(lr_accuracies)),
        lr_accuracies,
        color=COLOR_CYCLE[:len(lr_accuracies)],
        width=1,
    )

    model_axs[0].set_ylim(0, 1)
    model_axs[0].set_xticks(
        np.arange(len(lr_accuracies))
    )

    labels = lr_accuracies.index
    model_axs[0].set_xticklabels(
        labels,
        rotation=45,
        ha='right',
        va='top'
    )

    model_axs[0].set_xlabel('Components')
    model_axs[0].set_ylabel('Balanced Accuracy')

    # AUC-ROC Curves

    for i, reduced in enumerate(probabilities.columns):
        fpr, tpr, _ = roc_curve(
            predictions.loc[:, 'Actual'],
            probabilities[reduced]
        )

        model_axs[1].plot(fpr, tpr, color=COLOR_CYCLE[i])

    model_axs[1].set_xticks(np.linspace(0, 1, 6))
    model_axs[1].set_yticks(np.linspace(0, 1, 6))
    model_axs[1].set_xlim(0, 1)
    model_axs[1].set_ylim(0, 1)
    model_axs[1].set_xlabel('False Positive Rate')
    model_axs[1].set_ylabel('True Positive Rate')
    model_axs[1].legend(labels)
    model_axs[1].plot([0, 1], [0, 1], color='k', linestyle='--')

    # Best Model Scatter

    color = patient_data.loc[:, 'status'].astype(int)
    color = color.replace(0, 'blue')
    color = color.replace(1, 'red')

    style = patient_data.loc[:, 'gender'].astype(int)
    style = style.replace(0, 's')
    style = style.replace(1, 'o')

    xx, yy = np.meshgrid(
        np.linspace(-1.1, 1.1, 23),
        np.linspace(-1.1, 1.1, 23)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    prob_map = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    model_axs[2].contourf(
        xx,
        yy,
        prob_map,
        cmap='coolwarm',
        linestyles='--'
    )

    for marker in ['s', 'o']:
        index = style.loc[style == marker].index
        model_axs[2].scatter(
            components.loc[index, 2],
            components.loc[index, [4, 6]].sum(axis=1),
            c=color.loc[index],
            s=10,
            edgecolors='k',
            marker=marker,
            zorder=2
        )

    model_axs[2].set_xticks(np.arange(-1, 1.1, 0.5))
    model_axs[2].set_xlim([-1.1, 1.1])
    model_axs[2].set_yticks(np.arange(-1, 1.1, 0.5))
    model_axs[2].set_ylim([-1.1, 1.1])

    model_axs[2].set_xlabel(f'Component 2')
    model_axs[2].set_ylabel(f'Components 4 + 6')

    legend_markers = [
        Line2D([0], [0], lw=4, color='blue', label='Resolver'),
        Line2D([0], [0], lw=4, color='red', label='Persister'),
        Line2D([0], [0], marker='s', color='k', label='Male'),
        Line2D([0], [0], marker='o', color='k', label='Female')
    ]
    model_axs[2].legend(handles=legend_markers)
    
    # RNA & Cytokine Factor Comparisons
    
    rna_factors = pd.DataFrame(
        t_fac.mFactor,
        columns=np.arange(1, t_fac.rank + 1)
    )
    rna_factors = rna_factors.loc[:, PERSISTENCE_COMPONENTS[1:]]

    plasma, _ = import_cytokines()
    cyto_factors = pd.DataFrame(
        t_fac.factors[1],
        index=plasma.index,
        columns=np.arange(1, t_fac.rank + 1)
    )
    cyto_factors = cyto_factors.loc[:, PERSISTENCE_COMPONENTS[1:]]

    comp_axs = np.reshape(axs[3:], (2, 3))
    for factor, name, row in zip(
        [cyto_factors, rna_factors],
        ['Cytokines', 'RNA'],
        comp_axs
    ):
        for component, ax in zip(PERSISTENCE_COMPONENTS[1:], row):
            matrix = factor.drop(component, axis=1)
            if name == 'Cytokines':
                ax.set_xticks(np.arange(-1.5, 1.6, 0.5))
                ax.set_yticks(np.arange(-1.5, 1.6, 0.5))
                ax.grid(True)

                ax.set_axisbelow(True)
                important = matrix.loc[abs(matrix).max(axis=1) > 0.75, :].index
                ax.scatter(
                    matrix.drop(important).iloc[:, 0],
                    matrix.drop(important).iloc[:, 1],
                    alpha=0.5,
                    c='grey',
                    edgecolors='k',
                    s=10
                )
                ax.scatter(
                    matrix.loc[important, :].iloc[:, 0],
                    matrix.loc[important, :].iloc[:, 1],
                    c='green',
                    edgecolors='k',
                    s=10
                )
                for cyto in important:
                    ax.text(
                        matrix.loc[cyto, :].iloc[0],
                        matrix.loc[cyto, :].iloc[1],
                        s=cyto,
                        fontsize=8,
                        ha='right',
                        va='top',
                        ma='right'
                    )
                    ax.set_xlim([-1.1, 1.1])
                    ax.set_ylim([-1.1, 1.1])
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)

                top_overlap = []
                bot_overlap = []
                matrix.columns = matrix.columns.astype(str)
                matrix.loc[:, 'label'] = 'neither'
                for comp in matrix.columns[:-1]:
                    matrix = matrix.sort_values(by=comp, ascending=True)
                    important = pd.concat(
                        [
                            matrix.iloc[:500, :],
                            matrix.iloc[-500:, :]
                        ],
                        axis=0
                    )
                    matrix.loc[important.index, 'label'] = comp
                    top_overlap.append(set(important.index[-500:]))
                    bot_overlap.append(set(important.index[:500]))

                overlap = list(top_overlap[0] & top_overlap[1])
                overlap.extend(list(bot_overlap[0] & bot_overlap[1]))
                matrix.loc[overlap, 'label'] = 'shared'
                matrix.loc[:, 'label'] = matrix.loc[
                    :,
                    'label'
                ].astype('category')

                cvs = ds.Canvas(
                    plot_width=200,
                    plot_height=200,
                    x_range=(-1.6, 1.6),
                    y_range=(-1.6, 1.6)
                )
                agg = cvs.points(
                    matrix,
                    matrix.columns[0],
                    matrix.columns[1],
                    agg=ds.count_cat('label')
                )
                result = ds.tf.shade(
                    agg,
                    color_key=SCATTER_COLORS,
                    how='eq_hist',
                    min_alpha=255
                )
                result = ds.tf.set_background(result, 'white')
                img_rev = result.data[::-1]
                mpl_img = np.dstack(
                    [img_rev & 0x0000FF, (img_rev & 0x00FF00) >> 8,
                     (img_rev & 0xFF0000) >> 16]
                )
                ax.imshow(mpl_img)

            ax.set_xlabel(f'Component {matrix.columns[0]}')
            ax.set_ylabel(f'Component {matrix.columns[1]}')

    return fig


def makeFigure():
    t_fac, _, patient_data = get_factors()
    val_data = import_validation_patient_metadata()
    patient_data.loc[val_data.index, 'status'] = val_data.loc[:, 'status']

    components = t_fac[1][0]
    components = pd.DataFrame(
        components,
        index=patient_data.index,
        columns=list(np.arange(1, components.shape[1] + 1))
    )

    predictions, probabilities, model = run_cv(
        components,
        patient_data,
        svc=False
    )

    fig = plot_results(
        predictions,
        probabilities,
        model,
        components,
        t_fac,
        patient_data
    )

    return fig
