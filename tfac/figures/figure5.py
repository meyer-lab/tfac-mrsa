from matplotlib.patches import Patch
from os.path import abspath, dirname, join
import pandas as pd
import seaborn as sns

from tfac.figures.figureCommon import getSetup, OPTIMAL_SCALING
from tfac.dataImport import form_tensor, import_cytokines
from tfac.predict import predict_known, predict_validation
from tensorpac import perform_CMTF

PATH_HERE = dirname(dirname(abspath(__file__)))


def makeFigure():
    data_types, patient_data = get_data_types()
    train_samples = run_cv(data_types, patient_data)
    validation_samples = run_unknown(data_types, patient_data)
    export_results(train_samples, validation_samples)
    fig = plot_results(train_samples, validation_samples)

    return fig


def export_results(train_samples, validation_samples):
    """
    Reformats prediction DataFrames and saves as .csv.

    Parameters:
        train_samples (pandas.Series): predictions for training samples
        validation_samples (pandas.Series): predictions for validation samples

    Returns:
        None
    """
    validation_samples = validation_samples.astype(str)
    train_samples = train_samples.astype(str)

    validation_samples = validation_samples.replace('0', 'ARMB')
    validation_samples = validation_samples.replace('1', 'APMB')
    train_samples = train_samples.replace('0', 'ARMB')
    train_samples = train_samples.replace('1', 'APMB')

    validation_samples.to_csv(
        join(
            PATH_HERE,
            '..',
            'output',
            'validation_predictions.txt'
        )
    )
    train_samples.to_csv(
        join(
            PATH_HERE,
            '..',
            'output',
            'train_predictions.txt'
        )
    )


def get_data_types():
    """
    Creates data for classification.

    Parameters:
        None

    Returns:
        data_types (list[tuple]): data sources and their names
        patient_data (pandas.DataFrame): patient metadata
    """
    plasma_cyto, serum_cyto = import_cytokines()
    tensor, matrix, patient_data = form_tensor(OPTIMAL_SCALING)
    patient_data = patient_data.loc[:, ['status', 'type']]

    components = perform_CMTF(tensor, matrix)
    components = components[1][0]

    data_types = [
        ('Plasma Cytokines', plasma_cyto.T),
        ('Plasma IL-10', plasma_cyto.loc['IL-10', :]),
        ('Serum Cytokines', serum_cyto.T),
        ('Serum IL-10', serum_cyto.loc['IL-10', :]),
        ('CMTF', pd.DataFrame(
            components,
            index=patient_data.index,
            columns=list(range(1, components.shape[1] + 1))
        )
        )
    ]

    return data_types, patient_data


def run_unknown(data_types, patient_data):
    """
    Predicts samples with unknown outcomes.

    Parameters:
        data_types (list[tuple]): data sources to predict
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        predictions (pandas.Series): predictions for each data source
    """
    predictions = pd.DataFrame(
        index=patient_data.index
    )
    predictions = predictions.loc[patient_data['status'] == 'Unknown']

    for data_type in data_types:
        source = data_type[0]
        data = data_type[1]
        labels = patient_data.loc[data.index, 'status']

        _predictions = predict_validation(data, labels)
        predictions.loc[_predictions.index, source] = _predictions

    return predictions


def run_cv(data_types, patient_data):
    """
    Predicts samples with known outcomes via cross-validation.

    Parameters:
        data_types (list[tuple]): data sources to predict
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        predictions (pandas.Series): predictions for each data source
    """
    predictions = pd.DataFrame(
        index=patient_data.index
    )

    for data_type in data_types:
        source = data_type[0]
        data = data_type[1]
        labels = patient_data.loc[data.index, 'status']

        _predictions = predict_known(data, labels)
        predictions.loc[_predictions.index, source] = _predictions

    predictions.loc[:, 'Actual'] = patient_data.loc[:, 'status']

    return predictions


def plot_results(cv_results, val_results):
    """
    Plots predictions as heatmaps.

    Parameters:
        cv_results (pandas.DataFrame): predictions for samples with known
            outcomes
        val_results (pandas.DataFrame): predictions for samples with unknown
            outcomes

    Returns:
        fig (matplotlib.Figure): figure depicting predictions for all samples
    """
    fig_size = (8, 8)
    layout = {
        'ncols': 1,
        'nrows': 2
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )

    # Cross-validation predictions

    cv_results = cv_results.loc[cv_results['Actual'] != 'Unknown']
    cv_results = cv_results.fillna(-1).astype(int)
    axs[0] = sns.heatmap(
        cv_results.T,
        ax=axs[0],
        cbar=False,
        cmap=['dimgrey', '#ffd2d2', '#9caeff'],
        vmin=-1
    )

    axs[0].set_xticks([])
    axs[0].set_xlabel('Patient', fontsize=12)

    legend_elements = [Patch(facecolor='dimgrey', edgecolor='dimgrey', label='Data Not Available'),
                       Patch(facecolor='#ffd2d2', edgecolor='#ffd2d2', label='Persistor'),
                       Patch(facecolor='#9caeff', edgecolor='#9caeff', label='Resolver')]
    axs[0].legend(handles=legend_elements, loc=[1.05, 0.4])

    # Validation set predictions

    val_results = val_results.fillna(-1).astype(int)
    sns.heatmap(
        val_results.T,
        ax=axs[1],
        cbar=False,
        cmap=['dimgrey', '#ffd2d2', '#9caeff'],
        vmin=-1,
        linewidths=1
    )

    axs[1].set_xlabel('Patient', fontsize=12)

    legend_elements = [Patch(facecolor='dimgrey', edgecolor='dimgrey', label='Data Not Available'),
                       Patch(facecolor='#ffd2d2', edgecolor='#ffd2d2', label='Persistor'),
                       Patch(facecolor='#9caeff', edgecolor='#9caeff', label='Resolver')]
    axs[1].legend(handles=legend_elements, loc=[1.05, 0.4])

    return fig
