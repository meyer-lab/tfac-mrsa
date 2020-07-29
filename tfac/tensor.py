"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac2
from tensorly.metrics.regression import variance as tl_var
from tensorly.parafac2_tensor import parafac2_to_slice


tl.set_backend("numpy")  # Set the backend


def z_score_values(A, cell_dim):
    """ Function that takes in the values tensor and z-scores it. """
    assert cell_dim < tl.ndim(A)
    convAxes = tuple([i for i in range(tl.ndim(A)) if i != cell_dim])
    convIDX = [None] * tl.ndim(A)
    convIDX[cell_dim] = slice(None)

    sigma = tl.tensor(np.std(tl.to_numpy(A), axis=convAxes))
    return A / sigma[tuple(convIDX)]


def R2X(reconstructed, original):
    """ Calculates R2X of two tensors. """
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)


def R2Xparafac2(tensor_slices, decomposition):
    """Calculate the R2X of parafac2 decomposition"""
    R2X = np.zeros(len(tensor_slices))
    for idx, tensor_slice in enumerate(tensor_slices):
        reconstruction = parafac2_to_slice(decomposition, idx, validate=False)
        R2X[idx] = 1.0 - tl_var(reconstruction - tensor_slice) / tl_var(tensor_slice)
    return R2X


def reorient_factors(factors):
    """ Reorient factors based on the sign of the mean so that only the last factor can have negative means. """
    for index in range(len(factors) - 1):
        meann = np.sign(np.mean(factors[index], axis=0))
        assert meann.size == factors[0].shape[1]

        factors[index] *= meann
        factors[index + 1] *= meann

    return factors


#### Decomposition Methods ###################################################################


def MRSA_decomposition(tensor_slices, components, random_state=None):
    """Perform tensor formation and decomposition for particular variance and component number
    ---------------------------------------------
    Returns
        parafac2tensor object
        tensor_slices list
    """
    parafac2tensor = parafac2(tensor_slices, components, random_state=random_state, verbose=False)
    return parafac2tensor


###### To Flip Factors #########################################################################


def flip_factors(tucker_output):
    """For partial tucker OHSU factorization, flips protein and treatment/time factors if both negative for important values"""
    for component in range(tucker_output[0].shape[2]):
        av = 0.0
        for i in range(tucker_output[0].shape[0]):
            av += np.mean(tucker_output[0][i][:, component] ** 5)

        if av < 0 and tucker_output[1][0][:, component].mean() < 0:
            tucker_output[1][0][:, component] *= -1
            for j in range(tucker_output[0].shape[0]):
                tucker_output[0][j][:, component] *= -1
    return tucker_output
