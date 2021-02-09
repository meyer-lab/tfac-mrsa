"""
Test that we can successfully import the datasets.
"""
import pytest
import pandas as pd
from ..dataImport import importCohort1Expression, importCohort3Expression, full_import


@pytest.mark.parametrize("call", [importCohort1Expression, importCohort3Expression])
def test_importBases(call):
    """ Test that the most basic imports work. """
    data = call()

    assert isinstance(data, pd.DataFrame)


def test_fullImport():
    """ Test that we can import the full dataset. """
    cyto_list, cytokines, dfExp, geneIDs = full_import()

    assert isinstance(dfExp, pd.DataFrame)