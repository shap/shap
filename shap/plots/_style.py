"""Configuration of customisable style options for SHAP plots.

NOTE: This is experimental and subject to change!
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import TypedDict, Unpack

import numpy as np

from ..utils._exceptions import InvalidStyleOptionError
from . import colors

# Type hints, adapted from matplotlib.typing
RGBColorType = tuple[float, float, float] | str
type RGBAColorType = (
    str  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    | tuple[float, float, float, float]
    |
    # 2 tuple (color, alpha) representations, not infinitely recursive
    # RGBColorType includes the (str, float) tuple, even for RGBA strings
    tuple[RGBColorType, float]
    |
    # (4-tuple, float) is odd, but accepted as the outer float overriding A of 4-tuple
    tuple[tuple[float, float, float, float], float]
)
type ColorType = RGBColorType | RGBAColorType | np.ndarray

# Matches matplotlib's accepted fontsize values (number or named size string)
FontSizeType = int | float | str


# TODO: Use dataclass(kw_only=True) when we drop Python 3.9
@dataclasses.dataclass(frozen=True)
class StyleConfig:
    """A complete set of configuration options for matplotlib-based shap plots."""

    primary_color_positive: ColorType
    primary_color_negative: ColorType
    secondary_color_positive: ColorType
    secondary_color_negative: ColorType
    hlines_color: ColorType
    vlines_color: ColorType
    text_color: ColorType
    tick_labels_color: ColorType
    # Typography and line properties
    font_size: FontSizeType  # inline annotation text and colorbar labels
    label_size: FontSizeType  # axis labels and feature names
    tick_label_size: FontSizeType  # axis tick labels
    line_width: float  # thin separator and grid lines

    def asdict(self):
        return dataclasses.asdict(self)


class StyleOptions(TypedDict, total=False):
    """A TypedDict of partial updates to a style configuration"""

    # Nb. There is some duplication here with the StyleConfig dataclass, but
    # it's necessary to provide helpful type hints with `typing.Unpack`.
    # See https://github.com/python/typing/issues/1495
    primary_color_positive: ColorType
    primary_color_negative: ColorType
    secondary_color_positive: ColorType
    secondary_color_negative: ColorType
    hlines_color: ColorType
    vlines_color: ColorType
    text_color: ColorType
    tick_labels_color: ColorType
    font_size: FontSizeType
    label_size: FontSizeType
    tick_label_size: FontSizeType
    line_width: float


_shap_defaults = StyleConfig(
    primary_color_positive=colors.red_rgb,
    primary_color_negative=colors.blue_rgb,
    secondary_color_positive=colors.light_red_rgb,
    secondary_color_negative=colors.light_blue_rgb,
    hlines_color="#cccccc",
    vlines_color="#bbbbbb",
    text_color="white",
    tick_labels_color="#999999",
    font_size=12,
    label_size=13,
    tick_label_size=11,
    line_width=0.5,
)


def load_default_style() -> StyleConfig:
    """Load the default style configuration."""
    # In future, this could allow reading from a persistent config file, like matplotlib rcParams.
    return _shap_defaults


def load_matplotlib_style(style: str | None = None) -> StyleConfig:
    """Load style configuration from the active matplotlib rcParams or a matplotlib style sheet."""
    import matplotlib.pyplot as plt

    if style is not None:
        with plt.style.context(style):
            return _style_from_rc_params()
    return _style_from_rc_params()


def _style_from_rc_params() -> StyleConfig:
    import matplotlib.pyplot as plt

    rc_params = plt.rcParams
    prop_cycle = rc_params["axes.prop_cycle"].by_key().get("color", [])
    primary_color_positive = prop_cycle[0] if len(prop_cycle) > 0 else _shap_defaults.primary_color_positive
    primary_color_negative = prop_cycle[1] if len(prop_cycle) > 1 else _shap_defaults.primary_color_negative

    return StyleConfig(
        primary_color_positive=primary_color_positive,
        primary_color_negative=primary_color_negative,
        secondary_color_positive=primary_color_positive,
        secondary_color_negative=primary_color_negative,
        hlines_color=rc_params["grid.color"],
        vlines_color=rc_params["axes.edgecolor"],
        text_color=rc_params["text.color"],
        tick_labels_color=rc_params["xtick.color"],
        font_size=rc_params["font.size"],
        label_size=rc_params["axes.labelsize"],
        tick_label_size=rc_params["xtick.labelsize"],
        line_width=rc_params["grid.linewidth"],
    )


# Singleton instance that determines the current style.
# CAREFUL! To ensure the correct object is picked up, do not import this directly,
# but instead access this at runtime with get_style().
_STYLE = load_default_style()


def get_style() -> StyleConfig:
    """Return all currently active global style configuration options."""
    return _STYLE


def set_style(_style: StyleConfig | str | None = None, /, **options: Unpack[StyleOptions]) -> None:
    """Set options in the currently active global style configuration.

    Pass keyword arguments to set individual options, pass a StyleConfig dataclass
    to replace all options, or pass ``"shap"``/``"matplotlib"`` to load a base style.
    """
    if _style is not None:
        if isinstance(_style, str):
            if _style == "shap":
                _style = load_default_style()
            elif _style == "matplotlib":
                _style = load_matplotlib_style()
            else:
                try:
                    _style = load_matplotlib_style(_style)
                except OSError as exc:
                    raise InvalidStyleOptionError(f"Invalid style config option: {_style}") from exc
        # Unpack the dataclass; any keyword arguments take precendence.
        options = _style.asdict() | options
    global _STYLE
    _STYLE = _apply_options(_STYLE, options)


@contextmanager
def style_context(_style: StyleConfig | str | None = None, /, **options: Unpack[StyleOptions]):
    """Context manager to temporarily change style options.

    NOTE: This is experimental and subject to change!

    Examples
    --------
    To temporarily use black text color instead of the default (white)::

        with shap.plots.style_context(text_color="black"):
            shap.plots.waterfall(...)
    """
    old_style = get_style()
    set_style(_style, **options)
    try:
        yield
    finally:
        set_style(old_style)


def _apply_options(style: StyleConfig, changes: StyleOptions) -> StyleConfig:
    """Return a new StyleConfig with any changes applied, handling any invalid options."""
    valid_keys = set(f.name for f in dataclasses.fields(StyleConfig))
    for key in changes.keys():
        if key not in valid_keys:
            raise InvalidStyleOptionError(f"Invalid style config option: {key}")
    return dataclasses.replace(style, **changes)
