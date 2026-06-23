import matplotlib.pyplot as pl
import numpy as np
import pytest

from shap import Explanation
from shap.plots import colors
from shap.plots._utils import (
    convert_color,
    convert_ordering,
    dendrogram_coords,
    fill_counts,
    fill_internal_max_values,
    get_sort_order,
    merge_nodes,
    parse_axis_limit,
    sort_inds,
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


def test_convert_color_array_passthrough():
    color = np.array([0.1, 0.2, 0.3])
    result = convert_color(color)
    assert result is color


def test_convert_ordering_with_opchain_and_explanation():
    shap_values = np.array([3.0, 1.0, 2.0])

    np.testing.assert_array_equal(convert_ordering(Explanation.argsort, shap_values), np.array([1, 2, 0]))

    explanation = Explanation(shap_values)
    np.testing.assert_array_equal(convert_ordering(explanation, shap_values), np.array([0, 2, 1]))


def test_get_sort_order_respects_cluster_constraints():
    dist = np.array(
        [
            [0.0, 0.9, 0.2, 0.3],
            [0.9, 0.0, 0.8, 0.6],
            [0.2, 0.8, 0.0, 0.4],
            [0.3, 0.6, 0.4, 0.0],
        ],
    )
    clust_order = np.array([2, 0, 3, 1])
    feature_order = np.array([0, 1, 2, 3])

    result = get_sort_order(dist, clust_order, 0.5, feature_order)

    np.testing.assert_array_equal(result, np.array([0, 2, 3, 1]))
    np.testing.assert_array_equal(feature_order, np.array([0, 1, 2, 3]))


def test_merge_nodes_reindexes_and_recomputes_counts():
    values = np.array([5.0, 5.0, 0.1, 5.0, 0.2, 5.0])
    partition_tree = np.array(
        [
            [0, 1, 0.1, 2],
            [4, 2, 0.05, 2],
            [5, 4, 0.2, 2],
            [8, 8, 0.3, 3],
            [9, 9, 0.4, 5],
        ],
        dtype=float,
    )

    merged_tree, ind1, ind2 = merge_nodes(values, partition_tree)

    assert (ind1, ind2) == (2, 4)
    np.testing.assert_allclose(
        merged_tree,
        np.array(
            [
                [0.0, 1.0, 0.1, 2.0],
                [4.0, 2.0, 0.2, 2.0],
                [6.0, 6.0, 0.3, 4.0],
                [7.0, 7.0, 0.4, 8.0],
            ],
        ),
    )


def test_merge_nodes_rewrites_direct_references_to_removed_internal_node():
    values = np.array([5.0, 5.0, 0.1, 5.0, 0.2, 5.0])
    partition_tree = np.array(
        [
            [0, 1, 0.1, 2],
            [4, 2, 0.05, 2],
            [7, 7, 0.2, 2],
            [8, 8, 0.3, 3],
            [9, 9, 0.4, 5],
        ],
        dtype=float,
    )

    merged_tree, ind1, ind2 = merge_nodes(values, partition_tree)

    assert (ind1, ind2) == (2, 4)
    np.testing.assert_allclose(
        merged_tree,
        np.array(
            [
                [0.0, 1.0, 0.1, 2.0],
                [2.0, 2.0, 0.2, 2.0],
                [6.0, 6.0, 0.3, 4.0],
                [7.0, 7.0, 0.4, 8.0],
            ],
        ),
    )


def test_dendrogram_coords_returns_expected_segments():
    leaf_positions = np.array([0.0, 1.0, 2.0, 3.0])
    partition_tree = np.array(
        [
            [0, 1, 0.1, 2],
            [2, 3, 0.2, 2],
            [4, 5, 0.3, 4],
        ],
        dtype=float,
    )

    x_coords, y_coords = dendrogram_coords(leaf_positions, partition_tree)

    np.testing.assert_allclose(
        x_coords,
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [2.0, 2.0, 3.0, 3.0],
                [0.5, 0.5, 2.5, 2.5],
            ],
        ),
    )
    np.testing.assert_allclose(
        y_coords,
        np.array(
            [
                [0.0, 0.1, 0.1, 0.0],
                [0.0, 0.2, 0.2, 0.0],
                [0.1, 0.3, 0.3, 0.2],
            ],
        ),
    )


def test_fill_internal_max_values_uses_leaf_and_internal_nodes():
    partition_tree = np.array(
        [
            [0, 1, 0.1, 0],
            [2, 3, 0.2, 0],
            [4, 5, 0.3, 0],
        ],
        dtype=float,
    )
    leaf_values = np.array([-1.0, 3.0, -2.0, 4.0])

    new_tree = fill_internal_max_values(partition_tree, leaf_values)

    np.testing.assert_allclose(
        new_tree,
        np.array(
            [
                [0.0, 1.0, 0.1, 3.0],
                [2.0, 3.0, 0.2, 4.0],
                [4.0, 5.0, 0.3, 4.0],
            ],
        ),
    )
    np.testing.assert_allclose(partition_tree[:, 3], np.zeros(3))


def test_fill_counts_updates_cluster_sizes():
    partition_tree = np.array(
        [
            [0, 1, 0.1, 0],
            [2, 3, 0.2, 0],
            [4, 5, 0.3, 0],
        ],
        dtype=float,
    )

    fill_counts(partition_tree)

    np.testing.assert_allclose(partition_tree[:, 3], np.array([2.0, 2.0, 4.0]))


def test_sort_inds_orders_by_internal_max_values():
    partition_tree = np.array(
        [
            [0, 1, 0.1, 0],
            [2, 3, 0.2, 0],
            [4, 5, 0.3, 0],
        ],
        dtype=float,
    )
    leaf_values = np.array([-1.0, 3.0, -2.0, 4.0])

    inds = sort_inds(partition_tree, leaf_values)

    assert inds == [3, 2, 1, 0]


def test_parse_axis_limit_percentile_and_passthrough():
    ax_values = np.array([0.0, 10.0, 20.0, 30.0])

    assert parse_axis_limit("percentile(50)", ax_values, is_shap_axis=True) == 15.0
    assert parse_axis_limit(1.23, ax_values, is_shap_axis=True) == 1.23
    assert parse_axis_limit(None, ax_values, is_shap_axis=False) is None


def test_parse_axis_limit_from_explanation_data_and_values():
    ax_values = np.array([1.0, 2.0, 3.0])
    explanation = Explanation(np.array([2.5]), data=np.array([7.5]))[0]

    assert parse_axis_limit(explanation, ax_values, is_shap_axis=True) == 2.5
    assert parse_axis_limit(explanation, ax_values, is_shap_axis=False) == 7.5


def test_parse_axis_limit_rejects_invalid_percentile_string():
    with pytest.raises(ValueError, match="Only strings of the format `percentile\\(x\\)` are supported\\."):
        parse_axis_limit("percentile(not-a-number)", np.array([1.0]), is_shap_axis=True)
