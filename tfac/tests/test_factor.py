"""
Test that we can factor the data.
"""
import numpy as np
from ..dataImport import form_missing_tensor
from ..tensor import perform_CMTF


def test_CMTF():
    """ Test that we can form the missing tensor. """
    tensor_slices, _, _, _ = form_missing_tensor()

    tensor = np.stack((tensor_slices[0], tensor_slices[1])).T
    matrix = tensor_slices[2].T

    tFac = perform_CMTF(tensor, matrix, r=2)

    assert tFac.R2X > 0.0
