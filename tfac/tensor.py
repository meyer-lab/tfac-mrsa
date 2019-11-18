'''
Contains functions having to do wth tensor decomposition
'''
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from tensorly.metrics.regression import variance as tl_var

tl.set_backend('numpy')

def perform_parafac(tens, rank):
    '''Run Canonical Polyadic Decomposition on a tensor
    ---------------------------------------------------------
    Parameters:
        tens: numpy tensor
            Data tensor with which to perform factorization
        rank: int
            Number of component vectors desired along each axis during factorization

    Returns:
        factors: list of numpy arrays (length: dimensionality of the tensor)
            List of arrays containing the components for each axis (i.e. Factors[0] is the array containing the components for the first axis)

    '''
    _, factors = parafac(tens, rank)
    return factors

def calc_R2X_parafac(tens, rank):
    '''Calculate R2X of the decomposition of a tensor'''
    output = parafac(tens, rank)
    reconstructed = tl.kruskal_to_tensor(output)
    R2X = 1.0 - tl_var(reconstructed - tens)/tl_var(tens)
    return R2X
