"""
Test that we can factor the data.
"""
import numpy as np
from ..dataImport import get_scaled_tensors
from ..tensor import perform_CMTF


def test_CMTF():
    """ Test that we can form the missing tensor. """
    tensor, matrix, _ = get_scaled_tensors()

    tFac = perform_CMTF(tensor, matrix, r=3)

    assert tFac.R2X > 0.0
