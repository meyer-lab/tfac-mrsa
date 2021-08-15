from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tfac.dataImport import form_tensor, import_cytokines, import_rna
from tfac.predict import predict_known, predict_unknown
from tfac.tensor import perform_CMTF

OPTIMAL_SCALING = 32


def makeFigure():
    train_samples = run_cv()
    validation_samples = run_unknown()
    plot_results(train_samples, validation_samples)


def run_unknown():
    plasma_cyto, serum_cyto = import_cytokines()
    rna = import_rna()
    tensor, matrix, patient_data = form_tensor(OPTIMAL_SCALING)
    patient_data = patient_data.loc[:, ['status', 'type']]

    components = perform_CMTF(tensor, matrix, 9)
    components = components[1][0]

    data_types = [
        ('Plasma Cytokines', plasma_cyto.T),
        ('Plasma IL-10', plasma_cyto.loc['IL-10', :]),
        ('Serum Cytokines', serum_cyto.T),
        ('Serum IL-10', serum_cyto.loc['IL-10', :]),
        ('RNA', rna.T),
        ('CMTF', pd.DataFrame(
            components,
            index=patient_data.index,
            columns=list(range(1, 10))
        )
        )
    ]

    predicted = pd.DataFrame(
        index=patient_data.index
    )
    predicted = predicted.loc[patient_data['status'] == 'Unknown']

    for data_type in data_types:
        source = data_type[0]
        data = data_type[1]
        labels = patient_data.loc[data.index, 'status']

        predictions = predict_unknown(data, labels)
        predicted.loc[predictions.index, source] = predictions

    return predicted


def run_cv():
    plasma_cyto, serum_cyto = import_cytokines()
    rna = import_rna()
    tensor, matrix, patient_data = form_tensor(OPTIMAL_SCALING)
    patient_data = patient_data.loc[:, ['status', 'type']]

    components = perform_CMTF(tensor, matrix, 9)
    components = components[1][0]

    data_types = [
        ('Plasma Cytokines', plasma_cyto.T),
        ('Plasma IL-10', plasma_cyto.loc['IL-10', :]),
        ('Serum Cytokines', serum_cyto.T),
        ('Serum IL-10', serum_cyto.loc['IL-10', :]),
        ('RNA', rna.T),
        ('CMTF', pd.DataFrame(
            components,
            index=patient_data.index,
            columns=list(range(1, 10))
        )
        )
    ]

    predicted = pd.DataFrame(
        index=patient_data.index
    )

    for data_type in data_types:
        source = data_type[0]
        data = data_type[1]
        labels = patient_data.loc[data.index, 'status']

        predictions = predict_known(data, labels)
        predicted.loc[predictions.index, source] = predictions

    predicted.loc[:, 'Actual'] = patient_data.loc[:, 'status']

    return predicted


def plot_results(cv_results, val_results):
    cv_results = cv_results.loc[cv_results['Actual'] != 'Unknown']
    cv_results = cv_results.fillna(-1).astype(int)

    plt.figure(figsize=(10, 4))
    sns.heatmap(cv_results.T, cbar=False, cmap=['dimgrey', '#ffd2d2', '#9caeff'], vmin=-1)

    plt.xticks([], [])
    plt.yticks(fontsize=12)
    plt.xlabel('Patient', fontsize=12)

    legend_elements = [Patch(facecolor='dimgrey', edgecolor='dimgrey', label='Data Not Available'),
                       Patch(facecolor='#ffd2d2', edgecolor='#ffd2d2', label='Persistor'),
                       Patch(facecolor='#9caeff', edgecolor='#9caeff', label='Resolver')]
    plt.legend(handles=legend_elements, loc=[1.05, 0.4])

    plt.subplots_adjust(left=0.2, right=0.75, top=0.9)
    plt.show()

    val_results = val_results.fillna(-1)

    plt.figure(figsize=(10, 4))
    sns.heatmap(val_results.T, cbar=False, cmap=['dimgrey', '#ffd2d2', '#9caeff'], vmin=-1, linewidths=1)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Patient', fontsize=12)

    legend_elements = [Patch(facecolor='dimgrey', edgecolor='dimgrey', label='Data Not Available'),
                       Patch(facecolor='#ffd2d2', edgecolor='#ffd2d2', label='Persistor'),
                       Patch(facecolor='#9caeff', edgecolor='#9caeff', label='Resolver')]
    plt.legend(handles=legend_elements, loc=[1.05, 0.4])
    plt.subplots_adjust(left=0.2, right=0.75, bottom=0.25, top=0.9)

    plt.show()


if __name__ == '__main__':
    makeFigure()
