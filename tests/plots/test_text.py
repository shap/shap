import numpy as np
import pytest

import shap


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


def test_text_display_false():
    """Test text plot with display=False returns HTML."""
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
    html_output = shap.plots.text(shap_values_test, display=False)
    assert isinstance(html_output, str)
    assert len(html_output) > 0


def test_text_with_custom_params():
    """Test text plot with custom xmin, xmax, cmax parameters."""
    test_values = np.array([[10.0, 3.0], [-3.0, 10.0], [0.0, 0.0]])
    test_base_values = np.array([-6.0, -12.0])
    test_data = np.array(["Hello ", "world ", " "], dtype="<U7")
    test_output_names = np.array(["Output1", "Output2"], dtype="<U10")

    shap_values_test = shap.Explanation(
        values=[test_values],
        base_values=[test_base_values],
        data=[test_data],
        output_names=test_output_names,
    )
    html_output = shap.plots.text(
        shap_values_test,
        xmin=-20,
        xmax=20,
        cmax=15,
        display=False
    )
    assert isinstance(html_output, str)


def test_text_with_num_starting_labels():
    """Test text plot with num_starting_labels parameter."""
    test_values = np.array([[10.0, 3.0], [-3.0, 10.0], [0.0, 0.0]])
    test_base_values = np.array([-6.0, -12.0])
    test_data = np.array(["Hello ", "world ", " "], dtype="<U7")
    test_output_names = np.array(["Output1", "Output2"], dtype="<U10")

    shap_values_test = shap.Explanation(
        values=[test_values],
        base_values=[test_base_values],
        data=[test_data],
        output_names=test_output_names,
    )
    html_output = shap.plots.text(
        shap_values_test,
        num_starting_labels=2,
        display=False
    )
    assert isinstance(html_output, str)


def test_text_with_grouping_threshold():
    """Test text plot with custom grouping_threshold."""
    test_values = np.array([[10.0, 3.0], [-3.0, 10.0], [0.0, 0.0]])
    test_base_values = np.array([-6.0, -12.0])
    test_data = np.array(["Hello ", "world ", " "], dtype="<U7")
    test_output_names = np.array(["Output1", "Output2"], dtype="<U10")

    shap_values_test = shap.Explanation(
        values=[test_values],
        base_values=[test_base_values],
        data=[test_data],
        output_names=test_output_names,
    )
    html_output = shap.plots.text(
        shap_values_test,
        grouping_threshold=0.05,
        separator=" ",
        display=False
    )
    assert isinstance(html_output, str)


def test_text_multi_row_single_output():
    """Test text plot with multiple rows and single output (or string output_names)."""
    # This tests the path: len(shap_values.shape) == 2 and (output_names is None or isinstance(output_names, str))
    test_values = np.array([
        [[10.0, 3.0], [-3.0, 10.0], [0.0, 0.0]],
        [[5.0, 2.0], [-2.0, 5.0], [0.0, 0.0]]
    ])
    test_base_values = np.array([[-6.0, -12.0], [-5.0, -11.0]])
    test_data = np.array([
        ["Hello ", "world ", " "],
        ["Goodbye ", "world ", " "]
    ], dtype="<U10")

    shap_values_test = shap.Explanation(
        values=test_values,
        base_values=test_base_values,
        data=test_data,
        output_names=None,  # This will trigger multi-row path
    )
    html_output = shap.plots.text(shap_values_test, display=False)
    assert isinstance(html_output, str)
    # Should contain separators between rows
    assert "[0]" in html_output
    assert "[1]" in html_output


def test_text_multi_output_with_names():
    """Test text plot with multiple outputs."""
    # Create 2D data with output_names (not None, not string)
    test_values = np.array([[10.0, 3.0], [-3.0, 10.0], [0.0, 0.0]])
    test_base_values = np.array([-6.0, -12.0])
    test_data = np.array(["Hello ", "world ", " "], dtype="<U7")
    test_output_names = ["Output1", "Output2"]  # List, not None

    shap_values_test = shap.Explanation(
        values=[test_values],
        base_values=[test_base_values],
        data=[test_data],
        output_names=test_output_names,
    )

    # Test with display=True (but we can't actually display in tests)
    # This should still generate HTML even if display fails
    try:
        result = shap.plots.text(shap_values_test, display=True)
        # If display=True and have_ipython, it returns None
        # If no ipython, it might error
    except:
        pass  # Expected if not in IPython

    # Test with display=False
    html_output = shap.plots.text(shap_values_test, display=False)
    assert isinstance(html_output, str)
    assert "Output1" in html_output
    assert "Output2" in html_output
