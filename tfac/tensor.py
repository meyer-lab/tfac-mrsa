'''
Contains functions having to do wth tensor decomposition
'''
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from tensorly.metrics.regression import variance as tl_var


def perform_parafac(tens, rank):
    '''Run Canonical Polyadic Decomposition on a tensor'''
    _, factors = parafac(tens, rank)
    return factors


def calc_R2X_parafac(tens, rank):
    '''Calculate R2X of the decomposition of a tensor'''
    output = parafac(tens, rank)
    reconstructed = tl.kruskal_to_tensor(output)
    R2X = 1.0 - tl_var(reconstructed - tens) / tl_var(tens)
    return R2X
