"""
Test that we can factor the data.
"""
import numpy as np
from ..dataImport import form_missing_tensor
from ..tensor import perform_TMTF


def test_TMTF():
    """ Test that we can form the missing tensor. """
    tensor_slices, _, _, _ = form_missing_tensor()

    tensor = np.stack((tensor_slices[0], tensor_slices[1])).T
    matrix = tensor_slices[2].T

    print(tensor.shape)
    print(matrix.shape)

    tFac, mFac, R2X = perform_TMTF(tensor, matrix, r=1)

    assert R2X > 0.0
