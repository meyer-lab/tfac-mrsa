"""
Test that we can factor the data.
"""
from ..dataImport import form_tensor
from ..cmtf import perform_CMTF


def test_CMTF():
    """ Test that we can form the missing tensor. """
    tensor, matrix, _ = form_tensor()
    tFac = perform_CMTF(tensor, matrix, r=9)
    assert tFac.R2X > 0.0
