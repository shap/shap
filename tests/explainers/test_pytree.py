import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import shap
from shap.explainers import pytree


def test_pytree_decision_tree_regressor():
    """Tests pytree against a simple DecisionTreeRegressor for additivity."""
    X, y = shap.datasets.california(n_points=100)
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)

    # Use the pure python implementation
    explainer = pytree.TreeExplainer(model)

    # Explain first 5 instances
    X_test = X.values[:5]
    shap_values = explainer.shap_values(X_test)

    # The returned shape for single output is (instances, features + 1)
    # where the last column is the expected_value
    assert shap_values.shape == (5, X.shape[1] + 1)

    # Check additivity: sum(shap_values) == prediction
    preds = model.predict(X_test)
    np.testing.assert_allclose(np.sum(shap_values, axis=1), preds, atol=1e-10)


def test_pytree_vs_main_tree_explainer():
    """Comparison test between pure python pytree and optimized TreeExplainer."""
    X, y = shap.datasets.california(n_points=50)
    model = RandomForestRegressor(n_estimators=2, max_depth=3, random_state=42)
    model.fit(X, y)

    # Standard optimized TreeExplainer
    explainer_ref = shap.TreeExplainer(model)
    shap_values_ref = explainer_ref.shap_values(X.values[:5])
    expected_value_ref = (
        explainer_ref.expected_value[0]
        if isinstance(explainer_ref.expected_value, np.ndarray)
        else explainer_ref.expected_value
    )

    # Pure python pytree
    explainer_py = pytree.TreeExplainer(model)
    shap_values_py_with_base = explainer_py.shap_values(X.values[:5])

    # Split pytree's output into values and base
    shap_values_py = shap_values_py_with_base[:, :-1]
    expected_value_py = shap_values_py_with_base[0, -1]

    # Compare base values
    assert np.abs(expected_value_ref - expected_value_py) < 1e-10

    # Compare SHAP values
    # Note: main TreeExplainer returns (instances, features)
    # pytree returns (instances, features + 1) or a list depending on outputs.
    # For RandomForestRegressor (1 output), both should align.
    np.testing.assert_array_almost_equal(shap_values_ref, shap_values_py, decimal=10)

