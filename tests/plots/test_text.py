import re

import numpy as np

import shap


def test_text_plot_rgba_css_uses_plain_floats():
    """Regression: NumPy 2 scalars must not appear in CSS (see GH #4146)."""
    test_values = np.array([[1.0, -0.5], [0.2, 0.1]])
    test_base_values = np.array([0.0, 0.0])
    test_data = np.array(["a", "b"], dtype="<U1")
    exp = shap.Explanation(
        values=test_values,
        base_values=test_base_values,
        data=test_data,
        output_names=["x", "y"],
    )
    html = shap.plots.text(exp[:, 0], display=False)
    assert "np.float64" not in html
    assert "np.int" not in html
    assert re.search(
        r"rgba\(\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+\s*\)",
        html,
    )


def test_single_text_to_text():
    """Just make sure the test_plot function doesn't crash."""
    test_values = np.array([[10.61284012, 3.28389317], [-3.77245945, 10.76889759], [0.0, 0.0]])

    test_base_values = np.array([-6.12535715, -12.87049389])

    test_data = np.array(["▁Hello ", "▁world ", " "], dtype="<U7")

    test_output_names = np.array(["▁Hola", "▁mundo"], dtype="<U6")

    test_clustering = np.array([[0.0, 1.0, 12.0, 2.0], [3.0, 2.0, 13.0, 3.0]])

    test_hierarchical_values = np.array(
        [[13.91739416, 7.09603131], [-0.4679054, 14.58103573], [0.0, 0.0], [-6.60910809, -7.62427628], [0.0, 0.0]]
    )

    shap_values_test = shap.Explanation(
        values=[test_values],
        base_values=[test_base_values],
        data=[test_data],
        output_names=test_output_names,
        feature_names=test_base_values,
        clustering=[test_clustering],
        hierarchical_values=[test_hierarchical_values],
    )
    shap.plots.text(shap_values_test)
