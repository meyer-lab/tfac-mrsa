"""
Test our prediction methods.
"""
from sklearn.datasets import load_breast_cancer
from ..predict import SVC_predict


def test_CMTF():
    """ Test that the nested CV function is working. """
    data = load_breast_cancer()

    nested_pred, nested_score, _ = SVC_predict(data.data, data.target)

    assert nested_score > 0.4
    assert len(nested_pred) == len(data.target)
