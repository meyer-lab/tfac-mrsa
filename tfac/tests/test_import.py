"""
Test that we can successfully import the datasets.
"""
import pytest
import numpy as np
import pandas as pd
from ..dataImport import importCohort1Expression, importCohort3Expression, full_import, form_missing_tensor, get_C1_patient_info, form_MRSA_tensor


@pytest.mark.parametrize("call", [importCohort1Expression, importCohort3Expression, get_C1_patient_info])
def test_importBases(call):
    """ Test that the most basic imports work. """
    data = call()

    assert isinstance(data, pd.DataFrame)


def test_fullImport():
    """ Test that we can import the full dataset. """
    cyto_list, cytokines, dfExp, geneIDs = full_import()

    assert isinstance(geneIDs, list)
    assert isinstance(dfExp, pd.DataFrame)
    assert isinstance(cyto_list, list)


def test_formMissing():
    """ Test that we can form the missing tensor. """
    tensor_slices, cytokines, geneIDs, cohortID = form_missing_tensor()

    assert isinstance(tensor_slices, list)

    for i in range(3):
        assert isinstance(tensor_slices[i], np.ndarray)
        assert np.any(np.isnan(tensor_slices[i]))

    assert isinstance(geneIDs, list)
    assert tensor_slices[0].shape[1] == tensor_slices[1].shape[1]
    assert tensor_slices[0].shape[1] == tensor_slices[2].shape[1]


@pytest.mark.parametrize("dataType", ["serum", "plasma"])
def test_formTensor(dataType):
    """ Test that we can form the standard tensor. """
    tensor_slices, cytokines, geneIDs, cohortID = form_MRSA_tensor(dataType)

    assert len(tensor_slices) == 2
    assert np.all(np.isfinite(tensor_slices[1])) # Gene expression should be complete
    assert isinstance(cohortID, list)
    assert isinstance(geneIDs, list)
    assert tensor_slices[0].shape[1] == tensor_slices[1].shape[1]
