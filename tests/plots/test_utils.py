import matplotlib.pyplot as pl
import numpy as np
import pytest

from shap.plots import colors
from shap.plots._utils import aggregate_features_into_coalitions as _aggregate_features_into_coalitions
from shap.plots._utils import convert_color


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


def test_aggregate_features_into_coalitions():

    values = np.array([[1.0, 2.0, -0.5, 0.75], [1.0, 2.0, -0.5, 0.75]])
    feature_names = ["a", "b", "c", "d"]
    partition_tree = np.array(
        [
            [0, 1, 0.1, 2],
            [2, 3, 0.1, 2],
            [4, 5, 1.0, 4],
        ]
    )

    coalition_values, coalition_name = _aggregate_features_into_coalitions(
        values, feature_names=feature_names, clustering_cutoff=0.2, partition_tree=partition_tree
    )

    np.testing.assert_allclose(coalition_values, np.array([[3.0, 0.25], [3.0, 0.25]]))
    assert list(coalition_name) == ["a + b", "c + d"]
