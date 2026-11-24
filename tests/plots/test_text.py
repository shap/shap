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


def test_text_plot_rtl_container_direction_and_hr_markup():
    """Verify RTL token sequences get dir='rtl' on the container and valid <hr> markup."""
    rtl_exp = shap.Explanation(
        values=np.array([0.8, -0.3]),
        base_values=0.1,
        data=np.array(["مرحبا ", "بالعالم"]),
    )
    rtl_html = shap.plots.text(rtl_exp, display=False)
    assert "<div align='center' dir='rtl'>" in rtl_html
    assert (
        '<div dir="ltr" style="color: rgb(120,120,120); font-size: 12px; margin-top: -15px;">inputs</div>' in rtl_html
    )
    assert "dir='auto'" in rtl_html
    assert "unicode-bidi: isolate;" in rtl_html

    ltr_exp = shap.Explanation(
        values=np.array([0.5, -0.2]),
        base_values=0.0,
        data=np.array(["Hello ", "world"]),
    )
    ltr_html = shap.plots.text(ltr_exp, display=False)
    assert "<div align='center' dir='ltr'>" in ltr_html

    multi_exp = shap.Explanation(
        values=np.array([[0.5, -0.2], [0.1, 0.3]]),
        base_values=np.array([0.0, 0.0]),
        data=np.array([["Hello ", "world"], ["مرحبا ", "بالعالم"]]),
    )
    multi_html = shap.plots.text(multi_exp, display=False)
    assert '#ccc;"">' not in multi_html
    assert 'border-top: 1px dashed #ccc;">' in multi_html
    assert "<div align='center' dir='ltr'>" in multi_html
    assert "<div align='center' dir='rtl'>" in multi_html
