"""
Test that we can successfully import the datasets.
"""
import pytest
import numpy as np
import pandas as pd
from ..dataImport import importCohort1Expression, importCohort3Expression, full_import, form_missing_tensor, get_C1C2_patient_info, get_C3_patient_info, import_deconv


@pytest.mark.parametrize("call", [importCohort1Expression, importCohort3Expression, get_C1C2_patient_info, get_C3_patient_info, import_deconv])
def test_importBases(call):
    """ Test that the most basic imports work. """
    data = call()

    assert isinstance(data, pd.DataFrame)


# TODO: Update to use new imports
def test_fullImport():
    """ Test that we can import the full dataset. """
    cyto_list, cytokines, dfExp, geneIDs = full_import()

    assert isinstance(geneIDs, list)
    assert isinstance(dfExp, pd.DataFrame)
    assert isinstance(cyto_list, list)


# TODO: Update to use new form_tensor
def test_formMissing():
    """ Test that we can form the missing tensor. """
    tensor_slices, cytokines, geneIDs, patInfo = form_missing_tensor()

    assert isinstance(tensor_slices, list)

    for dd in tensor_slices:
        assert isinstance(dd, np.ndarray)
        assert np.any(np.isnan(dd))
        assert np.any(np.isfinite(dd))  # At least one value should be finite

    assert isinstance(geneIDs, list)
    assert tensor_slices[0].shape == tensor_slices[1].shape
    assert tensor_slices[0].shape[1] == tensor_slices[2].shape[1]
