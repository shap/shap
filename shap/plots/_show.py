"""Utilities for handling deprecated plot display arguments."""

from __future__ import annotations

import warnings


def resolve_show(show: bool | None, *, plot_name: str) -> bool:
    """Resolve deprecated ``show`` values to runtime behavior.

    Parameters
    ----------
    show : bool or None
        Deprecated display argument. ``None`` means the argument was not explicitly
        provided by the caller.
    plot_name : str
        Public plotting function name used in the warning message.

    Returns
    -------
    bool
        Whether ``matplotlib.pyplot.show()`` should be called.
    """
    if show is not None:
        warnings.warn(
            "The `show` argument to `shap.plots."
            f"{plot_name}` is deprecated and will be removed in a future release. "
            "Plots no longer call `matplotlib.pyplot.show()` by default. "
            "Call `matplotlib.pyplot.show()` explicitly if needed.",
            DeprecationWarning,
            stacklevel=3,
        )
    return bool(show)
