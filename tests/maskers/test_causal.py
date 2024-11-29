import re

import numpy as np
from _pytest.python_api import raises

import shap
from shap.maskers import Causal
from shap.utils._exceptions import DimensionError, TypeError


# ========================================== #
# === Causal =============================== #
# ========================================== #
class TestCausal:
    tiny_X, _ = shap.datasets.california(n_points=10)

    def test_should_correctly_count_the_number_of_features(self):
        """Check that n_features is equal to the data's number of features."""
        # Arrange
        expected = 8
        ordering = [["HouseAge"], ["AveRooms"]]

        # Act
        actual = Causal(self.tiny_X, ordering)

        # Assert
        assert actual.n_features == expected  # TODO what to assert here?

    def test_should_raise_error_if_ordering_named_feature_does_not_exist(self):
        """Check that ordering feature names are all in the datasets feature names."""
        # Arrange
        ordering = [["HouseAge"], ["Happiness"]]
        expected_message = "Feature Happiness does not appear in the provided dataset."

        # Act & Assert
        with raises(Exception, match=expected_message):
            Causal(self.tiny_X, ordering)

    def test_should_raise_error_if_ordering_feature_idx_is_too_high(self):
        """Check that ordering feature indices are not higher than the number of features."""
        # Arrange
        ordering = [[8]]
        expected_message = "Feature 8 does not appear in the provided dataset."

        # Act & Assert
        with raises(Exception, match=expected_message):
            Causal(self.tiny_X, ordering)

    def test_should_raise_error_if_ordering_feature_idx_is_too_low(self):
        """Check that ordering feature indices are not lower than  0."""
        # Arrange
        ordering = [[-1]]
        expected_message = "Feature -1 does not appear in the provided dataset."

        # Act & Assert
        with raises(Exception, match=expected_message):
            Causal(self.tiny_X, ordering)

    def test_should_default_to_empty_ordering_when_none_is_provided(self):
        """Check that the default ordering is an empty list when none is provided."""
        # Arrange
        expected = []

        # Act
        actual = Causal(self.tiny_X)

        # Assert
        assert actual.ordering == expected

    def test_should_raise_error_if_ordering_is_not_list(self):
        """Check that a TypeError is raised if ordering is not a list."""
        # Arrange
        ordering = "Not a list"
        expected_message = "Ordering must be a list."

        # Act & Assert
        with raises(TypeError, match=expected_message):
            Causal(self.tiny_X, ordering)

    def test_should_raise_error_for_1d_ordering(self):
        """Ensure DimensionError is raised for 1D ordering."""
        # Arrange
        ordering = [1, 2, 3]
        expected_message = "Ordering must be a 2d list."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            Causal(self.tiny_X, ordering)

    def test_should_raise_error_for_3d_ordering(self):
        """Ensure DimensionError is raised for 3D ordering."""
        # Arrange
        ordering = [[1], [[2], 3]]
        expected_message = "Ordering must be a 2d list."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            Causal(self.tiny_X, ordering)

    def test_should_accept_valid_2d_ordering(self):
        """Ensure 2D ordering is accepted."""
        # Arrange
        ordering = [[1], [2, 3]]

        # Act & Assert
        Causal(self.tiny_X, ordering)

    def test_should_raise_error_for_duplicate_features_in_same_group(self):
        """Ensure DimensionError is raised for duplicate features within the same group."""
        # Arrange
        ordering = [[1, 1], [3]]
        expected_message = "Feature 1 occurs multiple times in the provided ordering."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            Causal(self.tiny_X, ordering)

    def test_should_raise_error_for_duplicate_features_across_groups(self):
        """Ensure DimensionError is raised for duplicate features across groups."""
        # Arrange
        ordering = [[1], [1, 3]]
        expected_message = "Feature 1 occurs multiple times in the provided ordering."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            Causal(self.tiny_X, ordering)

    def test_should_raise_error_for_ordering_that_contains_other_than_string_or_integer(self):
        """Ensure TypeError is raised for ordering that contains other than strings or integers"""
        # Arrange
        ordering = [[Causal(self.tiny_X)]]
        expected_message = "Ordering features must either be feature names or feature indices."

        with raises(TypeError, match=expected_message):
            Causal(self.tiny_X, ordering)

    def test_should_raise_error_for_ordering_that_contains_both_strings_and_integers(self):
        """Ensure TypeError is raised for ordering that contains both feature names and feature indices"""
        # Arrange
        ordering = [[1], ["HouseAge"]]
        expected_message = "Mixing feature names and features indices is not supported."

        # Act & Assert
        with raises(TypeError, match=expected_message):
            Causal(self.tiny_X, ordering)

    def test_should_raise_error_when_feature_names_in_ordering_but_not_in_dataset(self):
        """Ensure TypeError is raised for ordering that contains feature names, while dataset does not have any."""
        # Arrange
        dataset = np.array([[10.0, 20.0, 30.0], [200.0, 400.0, 600.0]])
        ordering = [["Worries"]]
        expected_message = "Provided ordering contained feature names, but the given dataset does not have any."

        # Act & Assert
        with raises(Exception, match=re.escape(expected_message)):
            Causal(dataset, ordering)

    def test_should_default_confounding_to_true_for_all_groups_when_unspecified(self):
        """Ensure confounding is set to True for all groups if not provided."""
        # Arrange
        ordering = [[1], [2, 3]]
        expected_confounding = [True, True]

        # Act
        masker = Causal(self.tiny_X, ordering)

        # Assert
        assert np.array_equal(masker.confounding, expected_confounding)

    def test_should_warn_when_confounding_is_unspecified(self, caplog):
        """Ensure a warning is logged when confounding is not provided."""
        # Arrange
        ordering = [[1], [2, 3]]

        # Act
        with caplog.at_level("WARNING"):
            Causal(self.tiny_X, ordering)

        # Assert
        assert (
            "No confounding provided. Assuming that all causal groups contain have confounders present." in caplog.text
        )

    def test_should_raise_error_if_confounding_is_not_list_or_numpy_array(self):
        """Ensure a TypeError is raised if confounding is not a list or numpy array."""
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = "Not a valid type"
        expected_message = "Confounding must be a list or a numpy.array."

        # Act & Assert
        with raises(TypeError, match=expected_message):
            Causal(self.tiny_X, ordering, confounding)

    def test_should_raise_error_if_confounding_list_is_not_1_dimensional(self):
        """Ensure a DimensionError is raised if confounding is not 1 dimensional."""
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = [1, [3], [2]]
        expected_message = "Confounding must be a 1d array."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            Causal(self.tiny_X, ordering, confounding)

    def test_should_raise_error_if_confounding_array_is_not_1_dimensional(self):
        """Ensure a DimensionError is raised if confounding is not 1 dimensional."""
        # Arrange
        ordering = [[1], [2, 3, 4]]
        confounding = np.array([[1, 2], [3, 4]])
        expected_message = "Confounding must be a 1d array."

        # Act & Assert
        with raises(DimensionError, match=expected_message):
            Causal(self.tiny_X, ordering, confounding)

    def test_should_convert_list_confounding_to_numpy_array(self):
        """Ensure list-type confounding is converted to a numpy array."""
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = [True, False]

        # Act
        masker = Causal(self.tiny_X, ordering, confounding)

        # Assert
        assert isinstance(masker.confounding, np.ndarray)
        assert np.array_equal(masker.confounding, np.array(confounding))

    def test_should_raise_error_if_confounding_shape_does_not_match_ordering(self):
        """Ensure a DimensionError is raised if confounding shape does not match ordering groups."""
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = [True]  # Mismatched length
        expected_message = "Provided confounding shape is (1,), which does not match the number of causal groups 2. Please specify confounding for each group."

        # Act & Assert
        with raises(DimensionError, match=re.escape(expected_message)):
            Causal(self.tiny_X, ordering, confounding)

    def test_should_raise_error_if_confounding_is_not_boolean(self):
        """Ensure a TypeError is raised if confounding is not a boolean array."""
        # Arrange
        ordering = [[1], [2, 3]]
        confounding = [1, 0]  # Not boolean
        expected_message = "Confounding must be a boolean array."

        # Act & Assert
        with raises(TypeError, match=expected_message):
            Causal(self.tiny_X, ordering, confounding)
