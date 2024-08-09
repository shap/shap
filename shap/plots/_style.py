"""Configuration of customisable style options for SHAP plots.

NOTE: This is experimental and subject to change!
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TypeAlias, Union

from . import colors

# Type hints, adapted from matplotlib.typing
RGBColorType: TypeAlias = Union[tuple[float, float, float], str]
RGBAColorType: TypeAlias = Union[
    str,  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    tuple[float, float, float, float],
    # 2 tuple (color, alpha) representations, not infinitely recursive
    # RGBColorType includes the (str, float) tuple, even for RGBA strings
    tuple[RGBColorType, float],
    # (4-tuple, float) is odd, but accepted as the outer float overriding A of 4-tuple
    tuple[tuple[float, float, float, float], float],
]
ColorType: TypeAlias = Union[RGBColorType, RGBAColorType]


@dataclass
class StyleConfig:
    """Configuration of colors for shap plots."""

    # Waterfall plot config
    positive_arrow: ColorType = field(default_factory=lambda: colors.red_rgb)
    negative_arrow: ColorType = field(default_factory=lambda: colors.blue_rgb)
    default_positive_color: ColorType = field(default_factory=lambda: colors.light_red_rgb)
    default_negative_color: ColorType = field(default_factory=lambda: colors.light_blue_rgb)
    hlines: ColorType = "#cccccc"
    vlines: ColorType = "#bbbbbb"
    text: ColorType = "white"
    tick_labels: ColorType = "#999999"


def _load_default_style() -> StyleConfig:
    # In future, this could allow reading from a persistent config file, like matplotlib rcParams
    return StyleConfig()


# Singleton instance that determines the current style
STYLE = _load_default_style()


@contextmanager
def use_style(style: StyleConfig):
    """Context manager to temporarily change the style.

    NOTE: This is experimental and subject to change!

    Example
    -------
    with shap.plots.style.use_style(new_style):
        shap.plots.waterfall(...)
    """
    global STYLE
    old_style = STYLE
    STYLE = style
    yield
    STYLE = old_style
