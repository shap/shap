"""Tests for shap.utils._feature_selection."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import shap


# ──────────────────── Fixtures ────────────────────


@pytest.fixture
def classification_data():
    """Create a simple classification dataset with known informative features."""
    X, y = make_classification(
        n_samples=100,
        n_features=6,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def trained_model(classification_data):
    """Train a RandomForest on the classification data."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def shap_explanation(trained_model, classification_data):
    """Compute SHAP values for the classification data."""
    X, _ = classification_data
    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer(X)
    return shap_values


# ──────────────────── rank_features ────────────────────


class TestRankFeatures:
    """Tests for shap.utils.rank_features."""

    def test_basic_ranking(self, shap_explanation):
        """rank_features should return all expected keys."""
        result = shap.utils.rank_features(shap_explanation)

        assert "ranked_indices" in result
        assert "importance_scores" in result
        assert "feature_names" in result

    def test_ranked_indices_shape(self, shap_explanation):
        """Ranked indices should cover all features exactly once."""
        result = shap.utils.rank_features(shap_explanation)
        n_features = shap_explanation.shape[1]

        assert len(result["ranked_indices"]) == n_features
        assert set(result["ranked_indices"]) == set(range(n_features))

    def test_importance_is_sorted_descending(self, shap_explanation):
        """Importance scores should be in descending order."""
        result = shap.utils.rank_features(shap_explanation)
        scores = np.abs(result["importance_scores"])

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at position {i} ({scores[i]:.4f}) is less than "
                f"score at position {i + 1} ({scores[i + 1]:.4f})"
            )

    def test_custom_feature_names(self, shap_explanation):
        """User-supplied feature names should override defaults."""
        names = ["a", "b", "c", "d", "e", "f"]
        result = shap.utils.rank_features(shap_explanation, feature_names=names)

        # All returned names should be from the supplied list
        assert set(result["feature_names"]) == set(names)

    def test_mean_abs_method(self, shap_explanation):
        """The default 'mean_abs' method should produce non-negative scores."""
        result = shap.utils.rank_features(shap_explanation, method="mean_abs")
        assert np.all(result["importance_scores"] >= 0)

    def test_mean_method(self, shap_explanation):
        """The 'mean' method should allow negative scores."""
        result = shap.utils.rank_features(shap_explanation, method="mean")
        # Just ensure it runs without error
        assert len(result["importance_scores"]) == shap_explanation.shape[1]

    def test_max_abs_method(self, shap_explanation):
        """The 'max_abs' method should produce non-negative scores."""
        result = shap.utils.rank_features(shap_explanation, method="max_abs")
        assert np.all(result["importance_scores"] >= 0)

    def test_invalid_method_raises(self, shap_explanation):
        """An invalid aggregation method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            shap.utils.rank_features(shap_explanation, method="invalid")


# ──────────────────── select_features ────────────────────


class TestSelectFeatures:
    """Tests for shap.utils.select_features."""

    def test_backward_selection(
        self, shap_explanation, trained_model, classification_data
    ):
        """Backward selection should return a valid result dict."""
        X, y = classification_data
        result = shap.utils.select_features(
            shap_explanation,
            trained_model,
            X,
            y,
            method="backward",
            cv=3,
        )

        assert "selected_indices" in result
        assert "selected_names" in result
        assert "scores" in result
        assert "best_score" in result

    def test_forward_selection(
        self, shap_explanation, trained_model, classification_data
    ):
        """Forward selection should return a valid result dict."""
        X, y = classification_data
        result = shap.utils.select_features(
            shap_explanation,
            trained_model,
            X,
            y,
            method="forward",
            cv=3,
        )

        assert "selected_indices" in result
        assert len(result["selected_indices"]) >= 1

    def test_selected_is_subset(
        self, shap_explanation, trained_model, classification_data
    ):
        """Selected features should be a subset of all features."""
        X, y = classification_data
        result = shap.utils.select_features(
            shap_explanation,
            trained_model,
            X,
            y,
            method="backward",
            cv=3,
        )

        n_features = X.shape[1]
        for idx in result["selected_indices"]:
            assert 0 <= idx < n_features

    def test_min_features_respected(
        self, shap_explanation, trained_model, classification_data
    ):
        """The min_features constraint should be respected."""
        X, y = classification_data
        result = shap.utils.select_features(
            shap_explanation,
            trained_model,
            X,
            y,
            method="backward",
            min_features=3,
            cv=3,
        )

        assert len(result["selected_indices"]) >= 3

    def test_scores_track_all_steps(
        self, shap_explanation, trained_model, classification_data
    ):
        """The scores list should have one entry per evaluation step."""
        X, y = classification_data
        n_features = X.shape[1]

        result = shap.utils.select_features(
            shap_explanation,
            trained_model,
            X,
            y,
            method="backward",
            min_features=1,
            cv=3,
        )

        # backward: evaluates from n_features down to 1
        assert len(result["scores"]) == n_features

    def test_invalid_method_raises(
        self, shap_explanation, trained_model, classification_data
    ):
        """An invalid selection method should raise ValueError."""
        X, y = classification_data
        with pytest.raises(ValueError, match="Unknown selection method"):
            shap.utils.select_features(
                shap_explanation,
                trained_model,
                X,
                y,
                method="invalid",
                cv=3,
            )

    def test_invalid_min_features_raises(
        self, shap_explanation, trained_model, classification_data
    ):
        """Invalid min_features should raise ValueError."""
        X, y = classification_data
        with pytest.raises(ValueError, match="min_features must be between"):
            shap.utils.select_features(
                shap_explanation,
                trained_model,
                X,
                y,
                min_features=0,
                cv=3,
            )

    def test_best_score_is_max(
        self, shap_explanation, trained_model, classification_data
    ):
        """best_score should equal the maximum score across all steps."""
        X, y = classification_data
        result = shap.utils.select_features(
            shap_explanation,
            trained_model,
            X,
            y,
            method="backward",
            cv=3,
        )

        max_from_scores = max(s[1] for s in result["scores"])
        assert result["best_score"] == pytest.approx(max_from_scores)
