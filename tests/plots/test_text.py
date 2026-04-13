import numpy as np
import pytest

import shap


def test_text_plot_input_is_explanation():
    with pytest.raises(
        TypeError,
        match="text plot requires an `Explanation` object",
    ):
        shap.plots.text(np.array([[1.0, -1.0]]))


def test_text_plot_returns_html_when_display_false():
    explanation = shap.Explanation(
        values=np.array([0.25, -0.1]),
        base_values=0.0,
        data=np.array(["hello ", "world"]),
    )
    html = shap.plots.text(explanation, display=False)

    assert isinstance(html, str)
    assert "<div" in html


def test_text_plot_multirow_returns_html_when_display_false():
    explanation = shap.Explanation(
        values=np.array([[0.25, -0.1], [0.3, -0.2]]),
        base_values=np.array([0.0, 0.0]),
        data=np.array([["hello ", "world"], ["goodbye ", "world"]], dtype=object),
    )
    html = shap.plots.text(explanation, display=False)

    assert isinstance(html, str)
    assert "<div" in html


def test_text_plot_display_requires_ipython(monkeypatch):
    explanation = shap.Explanation(
        values=np.array([0.25, -0.1]),
        base_values=0.0,
        data=np.array(["hello ", "world"]),
    )
    monkeypatch.setattr(shap.plots._text, "have_ipython", False)

    with pytest.raises(ImportError, match="IPython is required for this function"):
        shap.plots.text(explanation, display=True)


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
