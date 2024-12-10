import re

import numpy as np
import pandas as pd
import pytest
from _pytest.python_api import raises
from sklearn.linear_model import LinearRegression

import shap
from shap.maskers import Causal
from shap.maskers._tabular import Causal as CausalMasker
from shap.maskers._tabular import CausalOrdering
from shap.utils._exceptions import DimensionError, TypeError


@pytest.fixture
def custom_causal_dataset():
    # Parameters
    n_samples = 1 * 10**6  # Number of samples
    alpha = 0.5  # Coefficient for E[X2 | X1]

    # P(X1) ~ N(0, 1)
    X1 = np.random.normal(loc=0, scale=1, size=n_samples)

    # Generate P(X_2 | X_1) using E[X2 | X1] = alpha * X1 (+ noise)
    epsilon = np.random.normal(loc=0, scale=1, size=n_samples)  # Noise
    X2 = alpha * X1 + epsilon

    # Verify the means
    assert -0.05 < np.mean(X1) < 0.05, f"Mean of X1 ({np.mean(X1)}) is not approximately 0!"
    assert -0.05 < np.mean(X2) < 0.05, f"Mean of X2 ({np.mean(X2)} is not approximately 0!"
    assert (
        -0.05 < np.cov(X1, X2)[0, 1] / np.var(X1) - alpha < 0.05
    ), f"Computed alpha ({np.cov(X1, X2)[0, 1] / np.var(X1)}) is not approximately alpha ({alpha})!"

    X = np.stack([X1, X2], axis=1)
    df = pd.DataFrame(X, columns=["X1", "X2"])

    return df, alpha


# ========================================== #
# === Causal =============================== #
# ========================================== #
class TestCausal:
    tiny_X, _ = shap.datasets.california(n_points=10)

    seed = 42

    def test_should_correctly_count_the_number_of_features(self):
        """Check that n_features is equal to the data's number of features."""
        # Arrange
        expected = 8
        ordering = [["HouseAge"], ["AveRooms"]]

        # Act
        actual = Causal(self.tiny_X, ordering)

        # Assert
        assert actual.n_features == expected

    def test_mask_applies_coalition_all(self):
        """Check if the mask sets in-coalition features to the value of x"""
        # Arrange
        X = np.array([[0.004051, 0.01215], [0.01215, 0.036451]])
        x = np.array([0.5, 1.15])
        sut = CausalMasker(X, seed=self.seed)

        mask = [1, 1]

        # Act
        actual = sut(mask, x)[0]

        # Assert
        assert np.all(actual == x)

    def test_mask_applies_coalition_some(self):
        """Check if the mask sets in-coalition features to the value of x for some in-coalition features"""
        # Arrange
        X = np.array([[0.004051, 0.01215], [0.01215, 0.036451]])
        x = np.array([0.5, 1.15])
        sut = CausalMasker(X, seed=self.seed)

        mask = [1, 0]

        # Act
        actual = sut(mask, x)[0][0]

        # Assert
        assert np.all(actual[:, 0] == x[0])
        assert not np.all(actual[:, 1] == x[1])

    def test_mask_conditions_on_parent(self, custom_causal_dataset):
        # Arrange
        data, alpha = custom_causal_dataset
        n_samples = 1 * 10**6

        ordering = [["X1"], ["X2"]]
        confounding = [False, False]
        sut = CausalMasker(data, ordering=ordering, confounding=confounding, max_samples=n_samples, seed=42)

        # Gather a sample and a mask for testing distribution
        x = np.array([0.276311, 0.510056])
        mask = [1, 0]

        # Act
        results = sut(mask, x)

        # Assert
        conditional_mean = np.mean(results["X2"])
        expected_mean = alpha * x[0]  # E[X2 | X1]
        assert np.isclose(
            conditional_mean, expected_mean, atol=0.001
        ), f"Conditional mean ({conditional_mean}) is not approximately alpha * X1 ({expected_mean})!"

        conditional_variance = np.var(results["X2"])
        expected_variance = 1  # Variance of noise
        assert np.isclose(
            conditional_variance, expected_variance, atol=0.005
        ), f"Conditional variance ({conditional_variance}) is not approximately 1!"

    def test_mask_conditions_on_sibling_when_not_confounding(self, custom_causal_dataset):
        # Arrange
        data, alpha = custom_causal_dataset
        n_samples = 1 * 10**6

        ordering = [["X1", "X2"]]
        confounding = [False]
        sut = CausalMasker(data, ordering=ordering, confounding=confounding, max_samples=n_samples, seed=42)

        # Gather a sample and a mask for testing distribution
        x = np.array([0.276311, 0.510056])
        mask = [1, 0]

        # Act
        results = sut(mask, x)

        # Assert
        conditional_mean = np.mean(results["X2"])
        expected_mean = alpha * x[0]  # E[X2 | X1]
        assert np.isclose(
            conditional_mean, expected_mean, atol=0.005
        ), f"Conditional mean ({conditional_mean}) is not approximately alpha * X1 ({expected_mean})!"

        conditional_variance = np.var(results["X2"])
        expected_variance = 1  # Variance of noise
        assert np.isclose(
            conditional_variance, expected_variance, atol=0.005
        ), f"Conditional variance ({conditional_variance}) is not approximately 1!"

        # also check X1

    def test_mask_does_not_condition_on_sibling_when_confounding(self, custom_causal_dataset):
        # Arrange
        data, alpha = custom_causal_dataset
        n_samples = 1 * 10**6

        ordering = [["X1", "X2"]]
        confounding = [False]
        sut = CausalMasker(data, ordering=ordering, confounding=confounding, max_samples=n_samples, seed=42)

        # Gather a sample and a mask for testing distribution
        x = np.array([0.276311, 0.510056])
        mask = [1, 0]

        # Act
        results = sut(mask, x)

        # Assert
        conditional_mean = np.mean(results["X2"])
        expected_mean = alpha * x[0]  # E[X2 | X1]
        assert np.isclose(
            conditional_mean, expected_mean, atol=0.005
        ), f"Conditional mean ({conditional_mean}) is not approximately alpha * X1 ({expected_mean})!"

        conditional_variance = np.var(results["X2"])
        expected_variance = 1  # Variance of noise
        assert np.isclose(
            conditional_variance, expected_variance, atol=0.005
        ), f"Conditional variance ({conditional_variance}) is not approximately 1!"

        # also check X1

    def test_mask_samples_features_not_in_ordering(self, custom_causal_dataset):
        # Arrange
        data, alpha = custom_causal_dataset
        n_samples = 100

        ordering = [["X1"]]
        confounding = [False]
        sut = CausalMasker(data, ordering=ordering, confounding=confounding, max_samples=n_samples, seed=42)

        # Gather a sample and a mask for testing distribution
        x = np.array([0.276311, 0.510056])
        mask = [1, 0]

        # Act
        results = sut(mask, x)

        # Assert
        assert not np.all(results["X2"] == 0)
        assert not np.all(results["X2"] is None)

    def test_chain_causal_shapley_values(self, custom_causal_dataset):
        # Arrange
        data, alpha = custom_causal_dataset

        # Generate labels
        beta = 2.0
        y = beta * data["X2"]

        # Train a linear model
        model = LinearRegression().fit(data, y)

        # Verify model coefficients
        assert np.isclose(model.intercept_, 0.00, 0.001)  # Bias
        assert np.isclose(model.coef_[0], 0.00, 0.001)  # X1 coefficient
        assert np.isclose(model.coef_[1], beta, 0.001)  # X2 coefficient

        # Use SHAP to analyze the contribution of each feature
        def model_predict(*args):
            return model.predict(*args)

        n_samples = 1 * 10**5
        masker = CausalMasker(
            data, ordering=[["X1"], ["X2"]], confounding=[False, False], max_samples=n_samples, seed=42
        )
        explainer = shap.ExactExplainer(model_predict, masker=masker)

        x1 = 0.276311
        x2 = 0.510056

        expected_x1_shap = (1 / 2) * beta * alpha * x1
        expected_x2_shap = beta * x2 - expected_x1_shap

        # Act
        actual = explainer(np.array([[x1, x2]])).values[0]

        # Assert
        assert np.isclose(actual[0], expected_x1_shap, atol=0.005)
        assert np.isclose(actual[1], expected_x2_shap, atol=0.005)

    def test_fork_causal_shapley_values(self, custom_causal_dataset):
        # Arrange
        data, alpha = custom_causal_dataset

        # Generate labels
        beta = 2.0
        y = beta * data["X2"]

        # Train a linear model
        model = LinearRegression().fit(data, y)

        # Verify model coefficients
        assert np.isclose(model.intercept_, 0.00, 0.001)  # Bias
        assert np.isclose(model.coef_[0], 0.00, 0.001)  # X1 coefficient
        assert np.isclose(model.coef_[1], beta, 0.001)  # X2 coefficient

        # Use SHAP to analyze the contribution of each feature
        def model_predict(*args):
            return model.predict(*args)

        n_samples = 1 * 10**5
        masker = CausalMasker(
            data, ordering=[["X2"], ["X1"]], confounding=[False, False], max_samples=n_samples, seed=42
        )
        explainer = shap.ExactExplainer(model_predict, masker=masker)

        x1 = 0.276311
        x2 = 0.510056

        expected_x1_shap = 0
        expected_x2_shap = beta * x2

        # Act
        actual = explainer(np.array([[x1, x2]])).values[0]

        # Assert
        assert np.isclose(actual[0], expected_x1_shap, atol=0.005)
        assert np.isclose(actual[1], expected_x2_shap, atol=0.005)

    def test_confounder_causal_shapley_values(self, custom_causal_dataset):
        # Arrange
        data, alpha = custom_causal_dataset

        # Generate labels
        beta = 2.0
        y = beta * data["X2"]

        # Train a linear model
        model = LinearRegression().fit(data, y)

        # Verify model coefficients
        assert np.isclose(model.intercept_, 0.00, 0.001)  # Bias
        assert np.isclose(model.coef_[0], 0.00, 0.001)  # X1 coefficient
        assert np.isclose(model.coef_[1], beta, 0.001)  # X2 coefficient

        # Use SHAP to analyze the contribution of each feature
        def model_predict(*args):
            return model.predict(*args)

        n_samples = 1 * 10**5
        masker = CausalMasker(data, ordering=[["X2", "X1"]], confounding=[True], max_samples=n_samples, seed=42)
        explainer = shap.ExactExplainer(model_predict, masker=masker)

        x1 = 0.276311
        x2 = 0.510056

        expected_x1_shap = 0
        expected_x2_shap = beta * x2

        # Act
        actual = explainer(np.array([[x1, x2]])).values[0]

        # Assert
        assert np.isclose(actual[0], expected_x1_shap, atol=0.005)
        assert np.isclose(actual[1], expected_x2_shap, atol=0.005)


class TestCausalOrdering:
    tiny_X_feature_names = pd.Index(
        [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
    )
    tiny_X_n_features = len(tiny_X_feature_names)

    ### Ordering init ###
    def test_should_raise_error_if_ordering_named_feature_does_not_exist(self):
        """Check that ordering feature names are all in the dataset's feature names."""
        # Arrange
        ordering = [["HouseAge"], ["Happiness"]]
        expected_message = "Feature Happiness does not appear in the provided dataset."

        # Act & Assert
        with raises(Exception, match=expected_message):
            CausalOrdering(
                ordering=ordering, n_features=self.tiny_X_n_features, feature_names=self.tiny_X_feature_names
            )

    def test_should_raise_error_if_ordering_feature_idx_is_too_high(self):
        """Check that ordering feature indices are not higher than the number of features."""
        # Arrange
        ordering = [[8]]
        expected_message = "Feature 8 does not appear in the provided dataset."

        # Act & Assert
        with raises(Exception, match=expected_message):
            CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

    def test_should_raise_error_if_ordering_feature_idx_is_too_low(self):
        """Check that ordering feature indices are not lower than 0."""
        # Arrange
        ordering = [[-1]]
        expected_message = "Feature -1 does not appear in the provided dataset."

        # Act & Assert
        with raises(Exception, match=expected_message):
            CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

    def test_should_default_to_empty_ordering_when_none_is_provided(self):
        """Check that the default ordering is an empty list when none is provided."""
        # Arrange
        expected = []
        sut = CausalOrdering()

        # Act
        actual = sut.ordering

        # Assert
        assert actual == expected

    def test_should_raise_error_if_ordering_is_not_list(self):
        """Check that a TypeError is raised if ordering is not a list."""
        # Arrange
        ordering = "Not a list"
        expected_message = "Ordering must be a list."

        # Act & Assert
        with raises(TypeError, match=expected_message):
            CausalOrdering(ordering=ordering)

    def test_should_raise_error_for_1d_ordering(self):
        """Ensure DimensionError is raised for 1D ordering."""
        # Arrange
        ordering = [1, 2, 3]
        expected_message = "Ordering must be a 2d list."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            CausalOrdering(ordering=ordering)

    def test_should_raise_error_for_3d_ordering(self):
        """Ensure DimensionError is raised for 3D ordering."""
        # Arrange
        ordering = [[1], [[2], 3]]
        expected_message = "Ordering must be a 2d list."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            CausalOrdering(ordering=ordering)

    def test_should_accept_valid_2d_ordering(self):
        """Ensure 2D ordering is accepted."""
        # Arrange
        ordering = [[1], [2, 3]]
        sut = CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

        # Act
        actual = sut.ordering

        # Assert
        assert actual == ordering

    def test_should_raise_error_for_duplicate_features_in_same_group(self):
        """Ensure DimensionError is raised for duplicate features within the same group."""
        # Arrange
        ordering = [[1, 1], [3]]
        expected_message = "Feature 1 occurs multiple times in the provided ordering."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

    def test_should_raise_error_for_duplicate_features_across_groups(self):
        """Ensure DimensionError is raised for duplicate features across groups."""
        # Arrange
        ordering = [[1], [1, 3]]
        expected_message = "Feature 1 occurs multiple times in the provided ordering."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

    def test_should_raise_error_for_ordering_that_contains_other_than_string_or_integer(self):
        """Ensure TypeError is raised for ordering that contains other than strings or integers."""
        # Arrange
        ordering = [[3.14]]
        expected_message = "The ordering must consist of either feature names or feature indices."

        with raises(TypeError, match=expected_message):
            CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

    def test_should_raise_error_for_ordering_that_contains_both_strings_and_integers(self):
        """Ensure TypeError is raised for ordering that contains both feature names and feature indices."""
        # Arrange
        ordering = [[1], ["HouseAge"]]
        expected_message = "Mixing feature names and features indices is not supported."

        # Act & Assert
        with raises(TypeError, match=expected_message):
            CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

    def test_should_raise_error_when_feature_names_in_ordering_but_not_in_dataset(self):
        """Ensure Exception is raised for ordering that contains feature names, while dataset does not have any."""
        # Arrange
        ordering = [["Worries"]]
        expected_message = "Provided ordering contained feature names, but the given dataset does not have any."

        # Act & Assert
        with raises(Exception, match=re.escape(expected_message)):
            CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

    def test_should_map_ordering_feature_names_to_indices(self):
        """Ensure feature names passed into ordering are converted to corresponding indices."""
        # Arrange
        ordering = [["HouseAge"], ["AveRooms"]]
        expected_ordering = [[1], [2]]

        # Act
        actual = CausalOrdering(
            ordering=ordering, n_features=self.tiny_X_n_features, feature_names=self.tiny_X_feature_names
        )

        # Assert
        assert actual.ordering == expected_ordering

    def test_len_method(self):
        """Ensure __len__ returns the correct number of causal groups."""
        # Arrange
        ordering = [[1], [2, 3], [4, 5]]
        sut = CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)
        expected = 3

        # Act
        actual = len(sut)

        # Assert
        assert actual == expected

    def test_getitem_method(self):
        """Ensure __getitem__ correctly returns the specified causal group."""
        # Arrange
        ordering = [[1], [2, 3], [4, 5]]
        sut = CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

        expected_1 = [1]
        expected_2 = [2, 3]
        expected_3 = [4, 5]

        # Act
        actual_1 = sut[0]
        actual_2 = sut[1]
        actual_3 = sut[2]

        # Act & Assert
        assert actual_1 == expected_1
        assert actual_2 == expected_2
        assert actual_3 == expected_3

    def test_getitem_slicing(self):
        """Ensure __getitem__ supports slicing."""
        # Arrange
        ordering = [[1], [2, 3], [4, 5]]
        sut = CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

        expected_1 = [[1], [2, 3]]
        expected_2 = [[2, 3], [4, 5]]

        # Act
        actual_1 = sut[:2]
        actual_2 = sut[1:]

        # Act & Assert
        assert actual_1 == expected_1
        assert actual_2 == expected_2

    def test_get_ancestors_method(self):
        """Ensure get_ancestors correctly returns a flattened list of all preceding causal groups."""
        # Arrange
        ordering = [[1], [2, 3], [4, 5]]
        sut = CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

        expected_1 = []
        expected_2 = [1]
        expected_3 = [1, 2, 3]

        # Act
        actual_1 = sut.get_ancestors(0)
        actual_2 = sut.get_ancestors(1)
        actual_3 = sut.get_ancestors(2)

        # Act & Assert
        assert np.array_equal(actual_1, expected_1)
        assert np.array_equal(actual_2, expected_2)
        assert np.array_equal(actual_3, expected_3)

    ### Confounding init ###
    def test_should_default_confounding_to_true_for_all_groups_when_unspecified(self):
        """Ensure confounding is set to True for all groups if not provided."""
        # Arrange
        ordering = [[1], [2, 3]]
        expected_confounding = [True, True]

        # Act
        sut = CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

        # Assert
        assert np.array_equal(sut.confounding, expected_confounding)

    def test_should_warn_when_confounding_is_unspecified(self, caplog):
        """Ensure a warning is logged when confounding is not provided."""
        # Arrange
        ordering = [[1], [2, 3]]
        expected_warning = (
            "No confounding provided. Assuming that all causal groups have confounding variables present."
        )

        # Act
        with caplog.at_level("WARNING"):
            CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)

        # Assert
        assert expected_warning in caplog.text

    def test_should_raise_error_if_confounding_is_not_list_or_numpy_array(self):
        """Ensure a TypeError is raised if confounding is not a list or numpy array."""
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = "Not a valid type"
        expected_message = "Confounding must be a list or a numpy.array."

        # Act & Assert
        with raises(TypeError, match=expected_message):
            CausalOrdering(ordering=ordering, confounding=confounding, n_features=self.tiny_X_n_features)

    def test_should_raise_error_if_confounding_list_is_not_1_dimensional(self):
        """Ensure a DimensionError is raised if confounding is not 1 dimensional."""
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = [1, [3], [2]]
        expected_message = "Confounding must be a 1d array."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            CausalOrdering(ordering=ordering, confounding=confounding, n_features=self.tiny_X_n_features)

    def test_should_raise_error_if_confounding_array_is_not_1_dimensional(self):
        """Ensure a DimensionError is raised if confounding is not 1 dimensional."""
        # Arrange
        ordering = [[1], [2, 3, 4]]
        confounding = np.array([[1, 2], [3, 4]])
        expected_message = "Confounding must be a 1d array."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            CausalOrdering(ordering=ordering, confounding=confounding, n_features=self.tiny_X_n_features)

    def test_should_convert_list_confounding_to_numpy_array(self):
        """Ensure list-type confounding is converted to a numpy array."""
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = [True, False]

        # Act
        sut = CausalOrdering(ordering=ordering, confounding=confounding, n_features=self.tiny_X_n_features)

        # Assert
        assert isinstance(sut.confounding, np.ndarray)
        assert np.array_equal(sut.confounding, np.array(confounding))

    def test_should_raise_error_if_confounding_shape_does_not_match_ordering(self):
        """Ensure a DimensionError is raised if confounding shape does not match ordering groups."""
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = [True]  # Mismatched length
        expected_message = "Provided confounding shape is (1,), which does not match the number of causal groups 2. Please specify confounding for each group."

        # Act & Assert
        with raises(DimensionError, match=re.escape(expected_message)):
            CausalOrdering(ordering=ordering, confounding=confounding, n_features=self.tiny_X_n_features)

    def test_should_raise_error_if_confounding_is_not_boolean(self):
        """Ensure a TypeError is raised if confounding is not a boolean array."""
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = [1, 0]  # Not boolean
        expected_message = "Confounding must be a boolean array."

        # Act & Assert
        with raises(TypeError, match=expected_message):
            CausalOrdering(ordering=ordering, confounding=confounding, n_features=self.tiny_X_n_features)

    def test_is_group_confounding_should_return_confounding_for_group_idx(self):
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = [True, False]
        sut = CausalOrdering(ordering=ordering, confounding=confounding, n_features=self.tiny_X_n_features)
        expected1, expected2 = True, False

        # Act
        actual1 = sut.is_group_confounding(0)
        actual2 = sut.is_group_confounding(1)

        # Assert
        assert actual1 == expected1
        assert actual2 == expected2

    def test_empty_causal_groups_are_filtered(self):
        # Arrange
        ordering = [[1], [], [2, 3]]
        sut = CausalOrdering(ordering=ordering, n_features=self.tiny_X_n_features)
        expected = [[1], [2, 3]]

        # Act
        actual = sut.ordering

        # Assert
        assert actual == expected
