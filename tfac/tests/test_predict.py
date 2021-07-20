"""
Test our prediction methods.
"""
from sklearn.datasets import load_breast_cancer


def test_CV():
    """ Test that the nested CV function is working. """
    data = load_breast_cancer()
