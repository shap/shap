import matplotlib.pyplot as pl
import numpy as np
import pytest

import shap
from shap.plots import colors
from shap.plots._utils import (
    convert_color,
    convert_ordering,
    get_sort_order,
    parse_axis_limit,
)


@pytest.mark.parametrize(
    ("color", "expected_result"),
    [
        ("shap_red", colors.red_rgb),
        ("shap_blue", colors.blue_rgb),
        ("null", "null"),
        ("jet", pl.get_cmap("jet")),
    ],
)
def test_convert_color(color, expected_result):
    result = convert_color(color)

    if isinstance(result, np.ndarray):
        assert np.allclose(result, expected_result)
    else:
        assert result == expected_result


def test_convert_color_numpy_passthrough():
    arr = np.array([0.1, 0.2, 0.3])
    result = convert_color(arr)

    assert np.array_equal(result, arr)


def test_convert_color_invalid_string():
    result = convert_color("not-a-color")
    assert result == "not-a-color"


def test_parse_axis_limit_percentile():
    values = np.array([0, 1, 2, 3])
    result = parse_axis_limit("percentile(50)", values, is_shap_axis=True)

    assert np.isclose(result, np.nanpercentile(values, 50))


def test_parse_axis_limit_invalid_string():
    values = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        parse_axis_limit("percentile(bad)", values, is_shap_axis=True)


def test_parse_axis_limit_explanation_shap():
    exp = shap.Explanation(values=np.array([2.0]), data=np.array([5.0]))
    result = parse_axis_limit(exp, np.array([1, 2]), is_shap_axis=True)

    assert result == 2.0


def test_parse_axis_limit_explanation_data():
    exp = shap.Explanation(values=np.array([2.0]), data=np.array([5.0]))
    result = parse_axis_limit(exp, np.array([1, 2]), is_shap_axis=False)

    assert result == 5.0


def test_parse_axis_limit_float():
    result = parse_axis_limit(3.14, np.array([1, 2]), is_shap_axis=True)
    assert result == 3.14


def test_parse_axis_limit_none():
    result = parse_axis_limit(None, np.array([1, 2]), is_shap_axis=True)
    assert result is None


def test_convert_ordering_numpy():
    ordering = np.array([2, 0, 1])
    values = np.array([0.2, 0.1, 0.3])

    result = convert_ordering(ordering, values)

    assert np.array_equal(result, ordering)


def test_convert_ordering_explanation_argsort():
    exp = shap.Explanation(values=np.array([3.0, 1.0, 2.0]))
    exp = exp.argsort  # triggers OpChain path

    result = convert_ordering(exp, np.array([3.0, 1.0, 2.0]))

    assert isinstance(result, np.ndarray)


def test_get_sort_order_basic():
    dist = np.array(
        [
            [0, 0.1, 0.9],
            [0.1, 0, 0.2],
            [0.9, 0.2, 0],
        ]
    )
    clust_order = np.array([2, 0, 1])
    feature_order = np.array([0, 1, 2])

    result = get_sort_order(dist, clust_order, 0.5, feature_order)

    assert isinstance(result, np.ndarray)
    assert len(result) == len(feature_order)
