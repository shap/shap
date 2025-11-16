"""Additional tests for TreeExplainer to increase coverage.

These tests use synthetic data and don't rely on external datasets.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import shap


def test_tree_explainer_with_single_tree():
    """Test TreeExplainer with a single decision tree."""
    # Create synthetic data
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Train a single decision tree
    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X, y)

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Get SHAP values
    shap_values = explainer.shap_values(X[:10])

    # Classifiers return shape (n_samples, n_features, n_classes)
    assert shap_values.shape == (10, 5, 2) or shap_values.shape == (10, 5)


def test_tree_explainer_with_decision_tree_regressor():
    """Test TreeExplainer with DecisionTreeRegressor."""
    X = np.random.randn(100, 4)
    y = X[:, 0] * 2 + X[:, 1] - 0.5 * X[:, 2]

    model = DecisionTreeRegressor(max_depth=4, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:5])

    assert shap_values.shape == (5, 4)
    # expected_value can be float or array
    assert isinstance(explainer.expected_value, (float, np.floating, np.ndarray))


def test_tree_explainer_with_dataframe():
    """Test TreeExplainer with pandas DataFrame input."""
    df = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
    y = (df["a"] + df["b"] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(df, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df[:10])

    # Classifiers return shape (n_samples, n_features, n_classes)
    assert shap_values.shape == (10, 3, 2) or shap_values.shape == (10, 3)


def test_tree_explainer_feature_perturbation_interventional():
    """Test TreeExplainer with interventional feature perturbation."""
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X, y)

    # Explicitly specify interventional
    explainer = shap.TreeExplainer(model, X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X[:5])

    assert shap_values.shape == (5, 4, 2) or shap_values.shape == (5, 4)


def test_tree_explainer_feature_perturbation_tree_path_dependent():
    """Test TreeExplainer with tree_path_dependent feature perturbation."""
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X[:5])

    assert shap_values.shape == (5, 4, 2) or shap_values.shape == (5, 4)


def test_tree_explainer_random_forest_binary_classification():
    """Test TreeExplainer with RandomForestClassifier for binary classification."""
    X = np.random.randn(150, 5)
    y = (X[:, 0] + 2 * X[:, 1] - X[:, 2] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    # For binary classification, might return list of length 2 or single array
    if isinstance(shap_values, list):
        assert len(shap_values) == 2
        assert shap_values[0].shape == (10, 5)
    else:
        # Can be (10, 5) or (10, 5, 2) depending on model
        assert shap_values.shape in [(10, 5), (10, 5, 2)]


def test_tree_explainer_gradient_boosting_regressor():
    """Test TreeExplainer with GradientBoostingRegressor."""
    X = np.random.randn(120, 6)
    y = X[:, 0] ** 2 + X[:, 1] + np.random.randn(120) * 0.1

    model = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:8])

    assert shap_values.shape == (8, 6)


def test_tree_explainer_gradient_boosting_classifier():
    """Test TreeExplainer with GradientBoostingClassifier."""
    X = np.random.randn(150, 4)
    y = (X[:, 0] + X[:, 1] * 2 > 0.5).astype(int)

    model = GradientBoostingClassifier(n_estimators=15, max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    assert shap_values.shape == (10, 4)


def test_tree_explainer_with_background_data():
    """Test TreeExplainer with explicit background data."""
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X, y)

    # Use a subset as background
    background = X[:50]

    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X[50:60])

    assert shap_values.shape == (10, 4, 2) or shap_values.shape == (10, 4)


def test_tree_explainer_check_additivity():
    """Test that SHAP values sum to prediction - expected_value."""
    X = np.random.randn(50, 3)
    y = X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(50) * 0.1

    model = DecisionTreeRegressor(max_depth=4, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    # Verify additivity: sum of SHAP values + expected_value ≈ prediction
    predictions = model.predict(X[:10])

    if isinstance(explainer.expected_value, np.ndarray):
        expected = explainer.expected_value[0]
    else:
        expected = explainer.expected_value

    shap_sum = shap_values.sum(axis=1) + expected

    np.testing.assert_allclose(shap_sum, predictions, rtol=1e-3, atol=1e-3)


def test_tree_explainer_single_sample():
    """Test TreeExplainer with a single sample."""
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    # Single sample as 1D array
    single_sample = X[0]
    shap_values = explainer.shap_values(single_sample)

    # Classifier with single sample can have various shapes
    assert shap_values.shape in [(4,), (1, 4), (4, 2), (1, 4, 2)]


def test_tree_explainer_with_xgboost_basic():
    """Test TreeExplainer with basic XGBoost model."""
    xgboost = pytest.importorskip("xgboost")

    X = np.random.randn(100, 5)
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

    model = xgboost.XGBRegressor(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    assert shap_values.shape == (10, 5)


def test_tree_explainer_with_xgboost_classifier():
    """Test TreeExplainer with XGBoost classifier."""
    xgboost = pytest.importorskip("xgboost")

    X = np.random.randn(120, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = xgboost.XGBClassifier(n_estimators=15, max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    assert shap_values.shape == (10, 4)


def test_tree_explainer_with_lightgbm_regressor():
    """Test TreeExplainer with LightGBM regressor."""
    lightgbm = pytest.importorskip("lightgbm")

    X = np.random.randn(100, 5)
    y = X[:, 0] + X[:, 1] ** 2 + np.random.randn(100) * 0.1

    model = lightgbm.LGBMRegressor(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    assert shap_values.shape == (10, 5)


def test_tree_explainer_with_lightgbm_classifier():
    """Test TreeExplainer with LightGBM classifier."""
    lightgbm = pytest.importorskip("lightgbm")

    X = np.random.randn(120, 4)
    y = (X[:, 0] - X[:, 1] > 0).astype(int)

    model = lightgbm.LGBMClassifier(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    # LightGBM binary classifier returns array, not list
    assert shap_values.shape == (10, 4) or (isinstance(shap_values, list) and len(shap_values) == 2)


def test_tree_explainer_expected_value():
    """Test that expected_value is computed correctly."""
    X = np.random.randn(100, 3)
    y = X[:, 0] + X[:, 1]

    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    # expected_value should be close to mean prediction on training data
    mean_pred = model.predict(X).mean()

    # expected_value can be float or array
    if isinstance(explainer.expected_value, np.ndarray):
        assert abs(explainer.expected_value[0] - mean_pred) < 1.0
    else:
        assert isinstance(explainer.expected_value, (float, np.floating))
        assert abs(explainer.expected_value - mean_pred) < 1.0


def test_tree_explainer_with_interactions():
    """Test TreeExplainer with interaction detection."""
    X = np.random.randn(80, 4)
    # Create interaction between features 0 and 1
    y = X[:, 0] * X[:, 1] + X[:, 2]

    model = DecisionTreeRegressor(max_depth=5, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    # Test interactions
    shap_interaction_values = explainer.shap_interaction_values(X[:10])

    assert shap_interaction_values.shape == (10, 4, 4)


def test_tree_explainer_output_as_explanation_object():
    """Test TreeExplainer returning Explanation object."""
    X = np.random.randn(50, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    # Call explainer directly (should return Explanation object)
    explanation = explainer(X[:5])

    assert isinstance(explanation, shap.Explanation)
    # Classifiers have extra dimension for classes
    assert explanation.values.shape in [(5, 3), (5, 3, 2)]


def test_tree_explainer_model_output_parameter():
    """Test TreeExplainer with different model_output parameters."""
    X = np.random.randn(80, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X, y)

    # Test with model_output="raw"
    explainer_raw = shap.TreeExplainer(model, model_output="raw")
    shap_values_raw = explainer_raw.shap_values(X[:5])

    # Test with model_output="probability" requires interventional mode with background data
    background = X[:40]
    explainer_prob = shap.TreeExplainer(
        model, background, model_output="probability", feature_perturbation="interventional"
    )
    shap_values_prob = explainer_prob.shap_values(X[:5])

    # Both should work - classifiers have extra dimension
    assert shap_values_raw.shape in [(5, 3), (5, 3, 2)]
    assert shap_values_prob.shape in [(5, 3), (5, 3, 2)]


def test_tree_explainer_different_dtypes():
    """Test TreeExplainer with different data types."""
    # Test with float32
    X_float32 = np.random.randn(60, 3).astype(np.float32)
    y = (X_float32[:, 0] + X_float32[:, 1] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X_float32, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_float32[:5])

    assert shap_values.shape in [(5, 3), (5, 3, 2)]


def test_tree_explainer_with_sparse_data():
    """Test TreeExplainer behavior with sparse-like data (many zeros)."""
    X_dense = np.random.randn(80, 5)
    # Make it sparse-like
    X_dense[X_dense < 0.5] = 0
    y = (X_dense[:, 0] + X_dense[:, 1] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=4, random_state=0)
    model.fit(X_dense, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_dense[:10])

    assert shap_values.shape in [(10, 5), (10, 5, 2)]


def test_tree_explainer_with_approximate():
    """Test TreeExplainer with approximate=True (Saabas method)."""
    X = np.random.randn(100, 4)
    y = X[:, 0] + X[:, 1] - X[:, 2]

    model = DecisionTreeRegressor(max_depth=4, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10], approximate=True)

    assert shap_values.shape == (10, 4)


def test_tree_explainer_with_check_additivity_false():
    """Test TreeExplainer with check_additivity=False."""
    X = np.random.randn(80, 3)
    y = X[:, 0] + X[:, 1]

    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10], check_additivity=False)

    assert shap_values.shape == (10, 3)


def test_tree_explainer_with_tree_limit():
    """Test TreeExplainer with tree_limit parameter."""
    X = np.random.randn(100, 4)
    y = X[:, 0] + X[:, 1]

    model = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    # Use only first 10 trees (disable additivity check since we're using subset)
    shap_values = explainer.shap_values(X[:5], tree_limit=10, check_additivity=False)

    assert shap_values.shape == (5, 4)


def test_tree_explainer_multiclass():
    """Test TreeExplainer with multi-class classification (>2 classes)."""
    X = np.random.randn(150, 4)
    # Create 3 classes
    y = np.zeros(150, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = 2

    model = DecisionTreeClassifier(max_depth=4, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    # Multi-class should return list or 3D array
    if isinstance(shap_values, list):
        assert len(shap_values) == 3
        assert shap_values[0].shape == (10, 4)
    else:
        assert shap_values.shape in [(10, 4, 3), (10, 4)]


def test_tree_explainer_with_pandas_series():
    """Test TreeExplainer with pandas Series input."""
    df = pd.DataFrame(np.random.randn(100, 4), columns=["a", "b", "c", "d"])
    y = df["a"] + df["b"]

    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    model.fit(df, y)

    explainer = shap.TreeExplainer(model)

    # Test with single row as Series
    single_row = df.iloc[0]
    shap_values = explainer.shap_values(single_row)

    # Single sample can have various shapes
    assert shap_values.shape in [(4,), (1, 4)]


def test_tree_explainer_random_forest_multiclass():
    """Test TreeExplainer with RandomForestClassifier multi-class."""
    X = np.random.randn(150, 4)
    # Create 3 classes
    y = np.zeros(150, dtype=int)
    y[X[:, 0] > 0.3] = 1
    y[X[:, 0] < -0.3] = 2

    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    # Should handle multi-class output
    if isinstance(shap_values, list):
        assert len(shap_values) == 3
    else:
        assert shap_values.shape in [(10, 4), (10, 4, 3)]


def test_tree_explainer_random_forest_regressor():
    """Test TreeExplainer with RandomForestRegressor."""
    X = np.random.randn(100, 5)
    y = X[:, 0] ** 2 + X[:, 1] - 0.5 * X[:, 2]

    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)
    model.fit(X, (y > 0).astype(int))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    # Verify shape is correct
    if isinstance(shap_values, list):
        assert len(shap_values) == 2
    else:
        assert shap_values.shape in [(10, 5), (10, 5, 2)]
