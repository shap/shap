import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # non-interactive backend for testing

from shap.plots._group_difference import group_difference


@pytest.fixture
def basic_data():
    shap_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    group_mask = np.array([True, True, False])
    return shap_values, group_mask


def test_basic_call(basic_data):
    shap_values, group_mask = basic_data
    _, ax = plt.subplots()
    group_difference(shap_values, group_mask, ax=ax, show=False)
    plt.close()


def test_feature_names_none(basic_data):
    shap_values, group_mask = basic_data
    _, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=None, ax=ax, show=False)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "Feature 0" in labels
    assert "Feature 1" in labels
    assert "Feature 2" in labels
    plt.close()


def test_1d_input():
    shap_values = np.array([0.1, 0.3, 0.5])
    group_mask = np.array([True, True, False])
    _, ax = plt.subplots()
    group_difference(shap_values, group_mask, ax=ax, show=False)
    plt.close()


def test_sort_true(basic_data):
    shap_values, group_mask = basic_data
    feature_names = ["a", "b", "c"]
    _, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=feature_names, sort=True, ax=ax, show=False)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels != ["a", "b", "c"]  # order should have changed
    plt.close()


def test_sort_false(basic_data):
    shap_values, group_mask = basic_data
    feature_names = ["a", "b", "c"]
    _, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=feature_names, sort=False, ax=ax, show=False)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["a", "b", "c"]
    plt.close()


def test_max_display(basic_data):
    shap_values, group_mask = basic_data
    _, ax = plt.subplots()
    group_difference(shap_values, group_mask, max_display=2, ax=ax, show=False)
    assert len(ax.patches) == 2
    plt.close()


def test_custom_ax(basic_data):
    shap_values, group_mask = basic_data
    _, ax = plt.subplots()
    group_difference(shap_values, group_mask, ax=ax, show=False)
    assert ax is not None
    plt.close()


def test_custom_xlabel(basic_data):
    shap_values, group_mask = basic_data
    _, ax = plt.subplots()
    group_difference(shap_values, group_mask, xlabel="my custom label", ax=ax, show=False)
    assert ax.get_xlabel() == "my custom label"
    plt.close()


def test_xmin_xmax(basic_data):
    shap_values, group_mask = basic_data
    _, ax = plt.subplots()
    group_difference(shap_values, group_mask, xmin=0.2, xmax=0.7, ax=ax, show=False)
    assert ax.get_xlim() == (0.2, 0.7)
    plt.close()