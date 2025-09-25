from __future__ import annotations

from typing import TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import TypeAlias

from .. import Explanation
from ..utils import OpChain
from . import colors

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


def convert_color(color: str | np.ndarray) -> np.ndarray | Colormap | str:
    """Converts a color specification alias into its actual representation"""
    if isinstance(color, np.ndarray):
        return color
    elif isinstance(color, str) and color == "shap_red":
        return colors.red_rgb
    elif isinstance(color, str) and color == "shap_blue":
        return colors.blue_rgb
    else:
        try:
            return plt.get_cmap(color)
        except ValueError:
            return color


def convert_ordering(ordering, shap_values):
    if issubclass(type(ordering), OpChain):
        ordering = ordering.apply(Explanation(shap_values))
    if issubclass(type(ordering), Explanation):
        if any(op.name == "argsort" for op in ordering.op_history):
            ordering = ordering.values
        else:
            ordering = ordering.argsort.flip.values
    return ordering


def get_sort_order(dist, clust_order, cluster_threshold, feature_order):
    """Returns a sorted order of the values where we respect the clustering order when dist[i,j] < cluster_threshold"""
    # feature_imp = np.abs(values)

    # if partition_tree is not None:
    #     new_tree = fill_internal_max_values(partition_tree, shap_values)
    #     clust_order = sort_inds(new_tree, np.abs(shap_values))
    clust_inds = np.argsort(clust_order)

    feature_order = feature_order.copy()  # order.apply(Explanation(shap_values))
    # print("feature_order", feature_order)
    for i in range(len(feature_order) - 1):
        ind1 = feature_order[i]
        next_ind = feature_order[i + 1]
        next_ind_pos = i + 1
        for j in range(i + 1, len(feature_order)):
            ind2 = feature_order[j]

            # if feature_imp[ind] >
            # if ind1 == 2:
            #     print(ind1, ind2, dist[ind1,ind2])
            if dist[ind1, ind2] <= cluster_threshold:
                # if ind1 == 2:
                #     print(clust_inds)
                #     print(ind1, ind2, next_ind, dist[ind1,ind2], clust_inds[ind2], clust_inds[next_ind])
                if dist[ind1, next_ind] > cluster_threshold or clust_inds[ind2] < clust_inds[next_ind]:
                    next_ind = ind2
                    next_ind_pos = j
            # print("next_ind", next_ind)
            # print("next_ind_pos", next_ind_pos)

        # insert the next_ind next
        for j in range(next_ind_pos, i + 1, -1):
            # print("j", j)
            feature_order[j] = feature_order[j - 1]
        feature_order[i + 1] = next_ind
        # print(feature_order)

    return feature_order


def merge_nodes(values, partition_tree):
    """This merges the two clustered leaf nodes with the smallest total value."""
    M = partition_tree.shape[0] + 1

    ptind = 0
    min_val = np.inf
    for i in range(partition_tree.shape[0]):
        ind1 = int(partition_tree[i, 0])
        ind2 = int(partition_tree[i, 1])
        if ind1 < M and ind2 < M:
            val = np.abs(values[ind1]) + np.abs(values[ind2])
            if val < min_val:
                min_val = val
                ptind = i
                # print("ptind", ptind, min_val)

    ind1 = int(partition_tree[ptind, 0])
    ind2 = int(partition_tree[ptind, 1])
    if ind1 > ind2:
        tmp = ind1
        ind1 = ind2
        ind2 = tmp

    partition_tree_new = partition_tree.copy()
    for i in range(partition_tree_new.shape[0]):
        i0 = int(partition_tree_new[i, 0])
        i1 = int(partition_tree_new[i, 1])
        if i0 == ind2:
            partition_tree_new[i, 0] = ind1
        elif i0 > ind2:
            partition_tree_new[i, 0] -= 1
            if i0 == ptind + M:
                partition_tree_new[i, 0] = ind1
            elif i0 > ptind + M:
                partition_tree_new[i, 0] -= 1

        if i1 == ind2:
            partition_tree_new[i, 1] = ind1
        elif i1 > ind2:
            partition_tree_new[i, 1] -= 1
            if i1 == ptind + M:
                partition_tree_new[i, 1] = ind1
            elif i1 > ptind + M:
                partition_tree_new[i, 1] -= 1
    partition_tree_new = np.delete(partition_tree_new, ptind, axis=0)

    # update the counts to be correct
    fill_counts(partition_tree_new)

    return partition_tree_new, ind1, ind2


def dendrogram_coords(leaf_positions, partition_tree):
    """Returns the x and y coords of the lines of a dendrogram where the leaf order is given.

    Note that scipy can compute these coords as well, but it does not allow you to easily specify
    a specific leaf order, hence this reimplementation.
    """
    xout: list[list[float]] = []
    yout: list[list[float]] = []
    _dendrogram_coords_rec(partition_tree.shape[0] - 1, leaf_positions, partition_tree, xout, yout)

    return np.array(xout), np.array(yout)


def _dendrogram_coords_rec(pos, leaf_positions, partition_tree, xout, yout):
    M = partition_tree.shape[0] + 1

    if pos < 0:
        return leaf_positions[pos + M], 0

    left = int(partition_tree[pos, 0]) - M
    right = int(partition_tree[pos, 1]) - M

    x_left, y_left = _dendrogram_coords_rec(left, leaf_positions, partition_tree, xout, yout)
    x_right, y_right = _dendrogram_coords_rec(right, leaf_positions, partition_tree, xout, yout)

    y_curr = partition_tree[pos, 2]

    xout.append([x_left, x_left, x_right, x_right])
    yout.append([y_left, y_curr, y_curr, y_right])

    return (x_left + x_right) / 2, y_curr


def fill_internal_max_values(partition_tree, leaf_values):
    """This fills the forth column of the partition tree matrix with the max leaf value in that cluster."""
    M = partition_tree.shape[0] + 1
    new_tree = partition_tree.copy()
    for i in range(new_tree.shape[0]):
        val = 0
        if new_tree[i, 0] < M:
            ind = int(new_tree[i, 0])
            val = max(val, np.abs(leaf_values[ind]))
        else:
            ind = int(new_tree[i, 0]) - M
            val = max(val, np.abs(new_tree[ind, 3]))  # / partition_tree[ind,2])
        if new_tree[i, 1] < M:
            ind = int(new_tree[i, 1])
            val = max(val, np.abs(leaf_values[ind]))
        else:
            ind = int(new_tree[i, 1]) - M
            val = max(val, np.abs(new_tree[ind, 3]))  # / partition_tree[ind,2])
        new_tree[i, 3] = val
    return new_tree


def fill_counts(partition_tree):
    """This updates the"""
    M = partition_tree.shape[0] + 1
    for i in range(partition_tree.shape[0]):
        val = 0
        if partition_tree[i, 0] < M:
            ind = int(partition_tree[i, 0])
            val += 1
        else:
            ind = int(partition_tree[i, 0]) - M
            val += partition_tree[ind, 3]
        if partition_tree[i, 1] < M:
            ind = int(partition_tree[i, 1])
            val += 1
        else:
            ind = int(partition_tree[i, 1]) - M
            val += partition_tree[ind, 3]
        partition_tree[i, 3] = val


def sort_inds(partition_tree, leaf_values, pos=None, inds=None):
    if inds is None:
        inds = []

    if pos is None:
        partition_tree = fill_internal_max_values(partition_tree, leaf_values)
        pos = partition_tree.shape[0] - 1

    M = partition_tree.shape[0] + 1

    if pos < 0:
        inds.append(pos + M)
        return

    left = int(partition_tree[pos, 0]) - M
    right = int(partition_tree[pos, 1]) - M

    left_val = partition_tree[left, 3] if left >= 0 else leaf_values[left + M]
    right_val = partition_tree[right, 3] if right >= 0 else leaf_values[right + M]

    if left_val < right_val:
        tmp = right
        right = left
        left = tmp

    sort_inds(partition_tree, leaf_values, left, inds)
    sort_inds(partition_tree, leaf_values, right, inds)

    return inds


# Various ways to specify a desired axis limit in plots
AxisLimitSpec: TypeAlias = Union[Explanation, str, float, None]


def parse_axis_limit(ax_limit: AxisLimitSpec, ax_values: np.ndarray, *, is_shap_axis: bool) -> float | None:
    """Handle axis limits in "percentile(float)" format or from Explanation objects.

    Parameters
    ----------
    ax_limit : LimitSpec
        Represents one of the lower or upper bounds of an xlim / ylim of a plot.
        Can be in "percentile(float)" format, a float or an :class:`.Explanation` object that has been aggregated
        into a single value, e.g. ``explanation[:, "feature_name"].percentile(20)``.

    ax_values : np.ndarray
        The values represented by the axis in question. Usually the SHAP values or the feature values. Only used if
        ``ax_limit`` is a string of the "percentile(float)" form.

    is_shap_axis : bool
        Whether the ``ax_limit`` is describing the axis representing SHAP values, or not. Only relevant when
        ``ax_limit`` is an :class:`.Explanation` object. It is assumed that when False, the axis is representing
        the feature values instead of the SHAP values.

    """
    if isinstance(ax_limit, str):
        try:
            percentage = float(ax_limit.removeprefix("percentile(").removesuffix(")"))
        except ValueError as e:
            raise ValueError("Only strings of the format `percentile(x)` are supported.") from e
        return np.nanpercentile(ax_values, percentage)
    if isinstance(ax_limit, Explanation):
        # Extract relevant attributes, depending on whether the axis is a SHAP-value axis or not
        return float(ax_limit.values) if is_shap_axis else float(ax_limit.data)
    # Else, should be float or None
    return ax_limit
