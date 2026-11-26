import re
import warnings

import numpy as np
import pytest

import shap
from shap.plots._text import _css_rgba, process_shap_values, svg_force_plot, text, unpack_shap_explanation_contents


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


def test_css_rgba_basic():
    """Test the _css_rgba function."""
    assert _css_rgba(255, 0, 0, 0.5) == "rgba(255.0, 0.0, 0.0, 0.5)"


def test_css_rgba_numpy_input():
    """Test the _css_rgba function with numpy inputs."""
    result = _css_rgba(np.float64(255), np.float64(0), np.float64(0), np.float64(0.5))
    assert "np.float64" not in result
    assert result == "rgba(255.0, 0.0, 0.0, 0.5)"


def test_text_returns_html_string():
    """Test the text function returns an HTML string."""
    sv = shap.Explanation(values=np.array([0.5, -0.2]), base_values=0.0, data=np.array(["good", "bad"]))
    result = text(sv, display=False)

    assert isinstance(result, str)
    assert "good" in result
    assert "bad" in result


def test_text_escapes_html_tokens():
    """Test the text function escapes HTML tokens."""
    sv = shap.Explanation(values=np.array([0.5]), base_values=0.0, data=np.array(["<script>"]))
    result = text(sv, display=False)

    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_text_display_true_returns_none():
    """Test the text function returns None when display is True."""
    sv = shap.Explanation(values=np.array([0.5]), base_values=0.0, data=np.array(["good"]))
    assert text(sv, display=True) is None


def test_process_shap_values_basic():
    """Test the process_shap_values function."""
    tokens = np.array(["good", "bad"])
    values = np.array([0.5, -0.2])

    out_tokens, out_values, group_sizes = process_shap_values(tokens, values, grouping_threshold=0.01, separator="")

    assert list(out_tokens) == ["good", "bad"]
    assert list(out_values) == [0.5, -0.2]
    assert list(group_sizes) == [1.0, 1.0]


def test_process_shap_values_raises_error_without_clustering():
    """Test the process_shap_values function raises an error without clustering."""
    tokens = np.array(["a", "b"])
    values = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        process_shap_values(tokens, values, grouping_threshold=0.01, separator="")


def test_text_with_output_names():
    """Test the text function with output names."""
    sv = shap.Explanation(
        values=np.array([[0.5, -0.2]]),
        base_values=np.array([0.0, 0.0]),
        data=np.array(["good"]),
        output_names=["positive", "negative"],
    )
    result = text(sv, display=False)

    assert "positive" in result
    assert "negative" in result


def test_svg_force_plot_returns_valid_svg():
    """Test the svg_force_plot function returns valid SVG."""
    result = svg_force_plot(
        values=np.array([0.5, -0.2]),
        base_values=0.0,
        fx=0.3,
        tokens=["good", "bad"],
        uuid="test",
        xmin=-1,
        xmax=1,
        output_name="",
    )
    assert result.strip().startswith("<svg")
    assert "</svg>" in result


def test_text_single_token():
    """Test the text function with a single token."""
    sv = shap.Explanation(values=np.array([0.0]), base_values=0.0, data=np.array(["test"]))
    result = text(sv, display=False)

    assert isinstance(result, str)
    assert "test" in result


def test_process_shap_values_return_metadata():
    """Test the process_shap_values function with return_meta_data=True."""
    tokens = np.array(["hello", "world"])
    values = np.array([0.3, -0.1])

    out_tokens, out_values, group_sizes, token_map, collapsed_ids = process_shap_values(
        tokens, values, grouping_threshold=0.01, separator="", return_meta_data=True
    )

    assert list(out_tokens) == ["hello", "world"]
    assert len(collapsed_ids) == 2


def test_positive_and_negative_tokens_get_different_colors():
    """Test that positive and negative tokens get different colors."""
    pos = shap.Explanation(values=np.array([0.9]), base_values=0.0, data=np.array(["great"]))
    neg = shap.Explanation(values=np.array([-0.9]), base_values=0.0, data=np.array(["great"]))

    assert text(pos, display=False) != text(neg, display=False)


def test_css_rgba_each_part_is_a_plain_number():
    """Test that each part of the RGBA color is a plain number."""
    result = _css_rgba(np.float64(128), np.float64(64), np.float64(32), np.float64(0.7))
    inner = result[5:-1]
    for part in inner.split(","):
        float(part.strip())


def test_unpack_shap_explanation_contents_basic():
    """Test the unpack_shap_explanation_contents function with basic inputs."""
    sv = shap.Explanation(values=np.array([0.5, -0.2]), base_values=0.0, data=np.array(["good", "bad"]))
    values, clustering = unpack_shap_explanation_contents(sv)

    assert list(values) == [0.5, -0.2]
    assert clustering is None


def test_unpack_shap_explanation_contents_uses_hierarchical_values_when_present():
    """Test the unpack_shap_explanation_contents function with hierarchical values."""
    sv = shap.Explanation(
        values=np.array([0.5, -0.2]),
        base_values=0.0,
        data=np.array(["good", "bad"]),
        hierarchical_values=np.array([0.9, -0.4]),
    )
    values, clustering = unpack_shap_explanation_contents(sv)

    assert list(values) == [0.9, -0.4]


def test_unpack_shap_explanation_contents_returns_clustering():
    """Test the unpack_shap_explanation_contents function returns clustering."""
    clustering = np.array([[0.0, 1.0, 1.0, 2.0]])
    sv = shap.Explanation(
        values=np.array([0.5, -0.2]), base_values=0.0, data=np.array(["good", "bad"]), clustering=clustering
    )
    values, returned_clustering = unpack_shap_explanation_contents(sv)

    assert returned_clustering is not None
    assert returned_clustering.shape == clustering.shape


def test_text_old_emits_future_warning():
    """Test that the text_old function emits a FutureWarning."""
    from shap.plots._text import text_old

    shap_values = np.array([0.5, -0.2])
    tokens = np.array(["good", "bad"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            text_old(shap_values, tokens)
        except Exception:
            pass
        warning_types = [w.category for w in caught]
        assert FutureWarning in warning_types


def test_text_num_starting_labels():
    """Test that the text function with num_starting_labels works correctly."""
    sv = shap.Explanation(
        values=np.array([0.9, 0.1, -0.8]), base_values=0.0, data=np.array(["great", "okay", "terrible"])
    )
    result = text(sv, num_starting_labels=1, display=False)

    assert isinstance(result, str)
    assert "great" in result or "terrible" in result


def test_text_with_separator():
    """Test that the text function with a separator works correctly."""
    tokens = np.array(["hello", "world"])
    values = np.array([0.3, -0.1])

    out_tokens, out_values, group_sizes = process_shap_values(tokens, values, grouping_threshold=0.01, separator=" ")

    assert list(out_tokens) == ["hello", "world"]


def test_svg_force_plot_contains_base_value_label():
    """Test that the svg_force_plot function contains a base value label."""
    result = svg_force_plot(
        values=np.array([0.5, -0.2]),
        base_values=0.0,
        fx=0.3,
        tokens=["good", "bad"],
        uuid="test",
        xmin=-1,
        xmax=1,
        output_name="",
    )
    assert "base value" in result
