"""
Test our prediction methods.
"""
from sklearn.datasets import load_breast_cancer
from ..predict import run_model


def test_CV():
    """ Test that the nested CV function is working. """
    X, y = load_breast_cancer(return_X_y=True)
    score, _ = run_model(X, y)
    assert score > 0.9
