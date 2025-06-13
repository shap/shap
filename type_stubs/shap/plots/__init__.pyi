# Type stubs for shap.plots
from typing import Any

# Core plotting functions
def bar(
    shap_values: Any,
    max_display: int = 10,
    order: Any = ...,
    clustering: Any = None,
    clustering_cutoff: float = 0.5,
    merge_cohorts: bool = False,
    show_data: bool = True,
    show: bool = True,
    **kwargs: Any,
) -> Any: ...
def waterfall(
    shap_values: Any,
    max_display: int = 10,
    show: bool = True,
    **kwargs: Any,
) -> Any: ...
def scatter(
    shap_values: Any,
    color: Any = None,
    x_jitter: float = 0,
    alpha: float = 1,
    dot_size: int = 16,
    show: bool = True,
    **kwargs: Any,
) -> Any: ...
def heatmap(
    shap_values: Any,
    instance_order: Any = ...,
    feature_order: Any = ...,
    max_display: int = 10,
    show: bool = True,
    **kwargs: Any,
) -> Any: ...
def force(
    base_value: float | Any,
    shap_values: Any = None,
    features: Any = None,
    feature_names: list[str] | None = None,
    out_names: str | None = None,
    link: str = "identity",
    plot_cmap: str = "RdBu_r",
    matplotlib: bool = False,
    show: bool = True,
    figsize: tuple[int, int] = (20, 3),
    ordering_keys: Any = None,
    ordering_keys_time_format: str | None = None,
    text_rotation: float = 0,
    contribution_threshold: float = 0.05,
    **kwargs: Any,
) -> Any: ...
def text(
    shap_values: Any,
    grouping_threshold: float = 0.01,
    separator: str = "",
    xmin: float | None = None,
    xmax: float | None = None,
    cmax: float | None = None,
    display_mode: str = "auto",
    **kwargs: Any,
) -> Any: ...
def image(
    shap_values: Any,
    pixel_values: Any = None,
    labels: Any = None,
    true_labels: Any = None,
    width: int = 20,
    aspect: float = 0.2,
    hspace: float = 0.2,
    labelpad: int | None = None,
    cmap: Any = ...,
    show: bool = True,
    **kwargs: Any,
) -> Any: ...
def partial_dependence(
    ind: int | str,
    model: Any,
    data: Any,
    ice: bool = True,
    model_expected_value: bool = False,
    feature_expected_value: bool = False,
    feature_names: list[str] | None = None,
    npoints: int = 100,
    hist: bool = True,
    xmin: float | None = None,
    xmax: float | None = None,
    show: bool = True,
    **kwargs: Any,
) -> Any: ...
def decision(
    expected_value: float | Any,
    shap_values: Any,
    features: Any = None,
    feature_names: list[str] | None = None,
    feature_order: str | list[int] = "importance",
    feature_display_range: slice | None = None,
    highlight: int | None = None,
    link: str = "identity",
    plot_color: str | None = None,
    axis_color: str = "#333333",
    y_demarc_color: str = "#333333",
    alpha: float = 1,
    color_bar: bool = True,
    auto_size_plot: bool = True,
    title: str | None = None,
    show: bool = True,
    return_objects: bool = False,
    **kwargs: Any,
) -> Any: ...
def embedding(
    shap_values: Any,
    feature_names: list[str] | None = None,
    max_points: int = 500,
    alpha: float = 1,
    x_jitter: float = 0,
    y_jitter: float = 0,
    **kwargs: Any,
) -> Any: ...
def beeswarm(
    shap_values: Any,
    max_display: int = 10,
    order: Any = ...,
    clustering: Any = None,
    cluster_threshold: float = 0.5,
    color: Any = None,
    axis_color: str = "#333333",
    alpha: float = 1,
    show: bool = True,
    log_scale: bool = False,
    color_bar: bool = True,
    plot_size: str = "auto",
    color_bar_label: str = "Feature value",
    **kwargs: Any,
) -> Any: ...
def violin(
    shap_values: Any,
    features: Any = None,
    feature_names: list[str] | None = None,
    max_display: int | None = None,
    plot_type: str = "violin",
    color: str | None = None,
    axis_color: str = "#333333",
    title: str | None = None,
    alpha: float = 1,
    show: bool = True,
    sort: bool = True,
    color_bar: bool = True,
    plot_size: str = "auto",
    layered_violin_max_num_bins: int = 20,
    x_jitter: float = 0,
    color_bar_label: str = "Feature value",
    **kwargs: Any,
) -> Any: ...
def group_difference(
    shap_values: Any,
    group_mask: Any,
    feature_names: list[str] | None = None,
    max_display: int = 30,
    show: bool = True,
    **kwargs: Any,
) -> Any: ...
def monitoring(
    indices: Any,
    shap_values: Any,
    features: Any = None,
    feature_names: list[str] | None = None,
    dates: Any = None,
    **kwargs: Any,
) -> Any: ...

# Utility functions
def initjs() -> None: ...
def getjs() -> str: ...
def save_html(
    out_file: str,
    plot: Any,
    full_html: bool = True,
) -> None: ...

# Legacy aliases for backward compatibility
summary_plot = beeswarm
dependence_plot = scatter
bar_plot = bar
