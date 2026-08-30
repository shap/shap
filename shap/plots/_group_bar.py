from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .. import Explanation
from ..utils import format_value
from ._style import get_style


def group_bar(
    shap_values: Explanation,
    partition_tree: dict[str, Any],
    max_display: int = 10,
    show_individual: bool = True,
    show: bool = True,
):
    """Bar plot for CoalitionExplainer output that respects the group hierarchy.

    when you use CoalitionExplainer with a partition_tree, the resulting shap values
    have a natural two-level structure -- groups and individual features within those groups.
    the regular shap.plots.bar() flattens everything and loses that structure entirely.

    this plot renders both levels: a bar for each group showing its total contribution,
    and indented bars for each feature inside that group.

    Parameters
    ----------
    shap_values : Explanation
        output from CoalitionExplainer.__call__(). if multiple rows are passed,
        the mean absolute shap value is used (same behaviour as shap.plots.bar).

    partition_tree : dict
        the same partition_tree dict you passed to CoalitionExplainer. defines
        which features belong to which group.

    max_display : int
        max number of groups to show. individual features within a shown group
        are always displayed. default is 10.

    show_individual : bool
        if True (default), individual feature bars are drawn indented under each group.
        set to False to only show group-level bars.

    show : bool
        whether to call plt.show(). set to False if you want to customise the plot further.

    Returns
    -------
    ax : matplotlib Axes
        only returned when show=False.

    Examples
    --------
    >>> import shap
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>>
    >>> X, y = load_iris(return_X_y=True)
    >>> model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    >>>
    >>> feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    >>> masker = shap.maskers.Partition(X)
    >>> masker.feature_names = feature_names
    >>>
    >>> tree = {
    ...     "Sepal": ["sepal length", "sepal width"],
    ...     "Petal": ["petal length", "petal width"],
    ... }
    >>>
    >>> explainer = shap.CoalitionExplainer(
    ...     model.predict, masker, partition_tree=tree, feature_names=feature_names
    ... )
    >>> sv = explainer(X[:10])
    >>> shap.plots.group_bar(sv, partition_tree=tree)
    """
    style = get_style()

    if show is False:
        plt.ioff()

    if not isinstance(shap_values, Explanation):
        raise TypeError(
            "group_bar requires an Explanation object as shap_values. "
            "pass the output of CoalitionExplainer(X) directly."
        )

    values = shap_values.values
    if values.ndim == 2:
        display_values = values.mean(axis=0)
    elif values.ndim == 1:
        display_values = values
    else:
        raise ValueError(f"shap_values.values must be 1d or 2d, got shape {values.shape}")

    feature_names = shap_values.feature_names
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(display_values))]

    name_to_idx: dict[str, int] = {}
    for idx, name in enumerate(feature_names):
        name_to_idx[name] = idx

    groups = _parse_partition_tree(partition_tree)

    all_tree_features: list[str] = []
    for _, feats in groups:
        all_tree_features.extend(feats)
    missing = [f for f in all_tree_features if f not in name_to_idx]
    if missing:
        raise ValueError(f"these features appear in partition_tree but not in shap_values.feature_names: {missing}")

    group_totals: list[tuple[str, float, list[tuple[str, float]]]] = []
    for group_name, feat_list in groups:
        feat_vals = [(f, float(display_values[name_to_idx[f]])) for f in feat_list]
        group_total = sum(v for _, v in feat_vals)
        group_totals.append((group_name, group_total, feat_vals))

    group_totals.sort(key=lambda x: abs(x[1]), reverse=True)
    group_totals = group_totals[:max_display]

    row_labels: list[str] = []
    row_values: list[float] = []
    row_is_group: list[bool] = []

    for group_name, group_total, feat_vals in group_totals:
        row_labels.append(group_name)
        row_values.append(group_total)
        row_is_group.append(True)

        if show_individual:
            feat_vals_sorted = sorted(feat_vals, key=lambda x: abs(x[1]), reverse=True)
            for feat_name, feat_val in feat_vals_sorted:
                row_labels.append(f"  {feat_name}")
                row_values.append(feat_val)
                row_is_group.append(False)

    n_rows = len(row_labels)
    row_values_arr = np.array(row_values)

    fig, ax = plt.subplots()
    row_height = 0.55
    fig.set_size_inches(9, max(3, n_rows * row_height + 1.8))

    y_pos = np.arange(n_rows - 1, -1, -1, dtype=float)

    if np.any(row_values_arr < 0):
        ax.axvline(0, color="#000000", linewidth=0.9, zorder=1)

    for i, (yp, val, is_group) in enumerate(zip(y_pos, row_values, row_is_group)):
        alpha = 1.0 if is_group else 0.55
        bar_h = 0.55 if is_group else 0.38
        color = style.primary_color_positive if val >= 0 else style.primary_color_negative
        ax.barh(
            yp,
            val,
            bar_h,
            align="center",
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )

    xmin, xmax = ax.get_xlim()
    xlen = xmax - xmin
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox_to_xscale = xlen / bbox.width if bbox.width > 0 else 1.0
    offset = (5 / 72) * bbox_to_xscale

    for i, (yp, val, is_group) in enumerate(zip(y_pos, row_values, row_is_group)):
        txt = format_value(val, "%+0.02f")
        if val >= 0:
            ax.text(
                val + offset,
                yp,
                txt,
                ha="left",
                va="center",
                fontsize=11 if is_group else 9.5,
                fontweight="bold" if is_group else "normal",
                color=style.primary_color_positive,
            )
        else:
            ax.text(
                val - offset,
                yp,
                txt,
                ha="right",
                va="center",
                fontsize=11 if is_group else 9.5,
                fontweight="bold" if is_group else "normal",
                color=style.primary_color_negative,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=12)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i, is_group in enumerate(row_is_group):
        if is_group:
            tick_labels[i].set_fontweight("bold")
            tick_labels[i].set_color("#222222")
        else:
            tick_labels[i].set_color(style.tick_labels_color)
            tick_labels[i].set_fontsize(10)

    for i, (yp, is_group) in enumerate(zip(y_pos, row_is_group)):
        if is_group and i > 0:
            ax.axhline(yp + 0.5, color="#dddddd", linewidth=0.8, zorder=0)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)

    xmin2, xmax2 = ax.get_xlim()
    ax.set_xlim(xmin2, xmax2 + (xmax2 - xmin2) * 0.12)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        return ax


def _parse_partition_tree(
    tree: dict[str, Any],
    _parent_prefix: str = "",
) -> list[tuple[str, list[str]]]:
    """Recursively flatten a nested partition_tree dict into (group_name, [leaf_features]) pairs."""
    result: list[tuple[str, list[str]]] = []
    for key, value in tree.items():
        group_name = f"{_parent_prefix} > {key}" if _parent_prefix else key
        if isinstance(value, dict):
            sub = _parse_partition_tree(value, _parent_prefix=group_name)
            result.extend(sub)
        elif isinstance(value, list):
            leaves: list[str] = []
            for item in value:
                if isinstance(item, str):
                    leaves.append(item)
                elif isinstance(item, dict):
                    sub = _parse_partition_tree(item, _parent_prefix=group_name)
                    result.extend(sub)
            if leaves:
                result.append((group_name, leaves))
        else:
            result.append((group_name, [str(value)]))
    return result
