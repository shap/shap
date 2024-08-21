from typing import Any

import numpy as np
import pandas as pd

import shap


def test_dependence_one_string_feature():
    """Test the dependence plot with a string feature."""
    X = _create_sample_dataset(string_features={"Sex"})

    shap.dependence_plot("Sex", np.random.randn(*X.values.shape), X, interaction_index="Age", show=False)


def test_dependence_two_string_features():
    """Test the dependence plot with two string features."""
    X = _create_sample_dataset(string_features={"Sex", "Blood group"})

    shap.dependence_plot("Sex", np.random.randn(*X.values.shape), X, interaction_index="Blood group", show=False)


def test_dependence_one_string_feature_no_interaction():
    """Test the dependence plot with no interactions."""
    X = _create_sample_dataset(string_features={"Sex"})

    shap.dependence_plot("Sex", np.random.randn(*X.values.shape), X, interaction_index=None, show=False)


def test_dependence_one_string_feature_auto_interaction():
    """Test the dependence plot with auto interaction detection."""
    X = _create_sample_dataset(string_features={"Sex"})

    shap.dependence_plot("Sex", np.random.randn(*X.values.shape), X, interaction_index="auto", show=False)


def test_approximate_interactions():
    """Test the approximate interaction detector."""
    X_no_string_features = _create_sample_dataset(string_features={})
    X_one_string_feature = _create_sample_dataset(string_features={"Sex"})
    X_two_string_features = _create_sample_dataset(string_features={"Sex", "Blood group"})

    shap_values = np.random.randn(*X_one_string_feature.values.shape)

    interactions_no_features = shap.approximate_interactions(0, shap_values, X_no_string_features)
    interactions_one_string_feature = shap.approximate_interactions(0, shap_values, X_one_string_feature)
    interactions_two_string_feature = shap.approximate_interactions(0, shap_values, X_two_string_features)

    assert (interactions_no_features == interactions_one_string_feature).all()
    assert (interactions_no_features == interactions_two_string_feature).all()


def _create_sample_dataset(string_features):
    sex_values: list[Any]
    if "Sex" in string_features:
        sex_values = ["Male", "Female", "Male", "Male", "Female", "Female"]
    else:
        sex_values = [0, 1, 0, 0, 1, 1]

    blood_values: list[Any]
    if "Blood group" in string_features:
        blood_values = ["A", "B", "A", "O", "O", "O"]
    else:
        blood_values = [1, 2, 1, 3, 3, 3]

    X = pd.DataFrame(
        {
            "Sex": sex_values,
            "Blood group": blood_values,
            "Age": [10, 15, 28, 3, 84, 56],
            "Height": [130, 170, 185, 40, 150, 164],
        }
    )
    return X
