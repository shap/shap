import shap
from nose.tools import *

def test_compute_indices_from_names():
    indices = shap.compute_display_feature_indices(
        display_features=["feature_0", "feature_2"],
        feature_names=["feature_0", "feature_1", "feature_2"]
    )
    assert (indices == [0,2]).all()


def test_compute_indices_from_indices():
    indices = shap.compute_display_feature_indices(
        display_features=[0, 2],
        feature_names=["feature_0", "feature_1", "feature_2"]
    )
    assert (indices == [0,2]).all()


def test_compute_indices_from_indices_and_names():
    indices = shap.compute_display_feature_indices(
        display_features=[0, "feature_2"],
        feature_names=["feature_0", "feature_1", "feature_2"]
    )
    assert (indices == [0, 2]).all()


def test_compute_indices_from_indices_and_names_repeating():
    indices = shap.compute_display_feature_indices(
        display_features=[0, "feature_0"],
        feature_names=["feature_0", "feature_1", "feature_2"]
    )
    assert len(indices) == 1
    assert (indices == [0]).all()


@raises(AssertionError)
def test_compute_indices_from_name_unknown_name():
    _ = shap.compute_display_feature_indices(
        display_features=["feature_0", "unknwon_feature"],
        feature_names=["feature_0", "feature_1", "feature_2"]
    )

