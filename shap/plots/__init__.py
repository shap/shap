import lazy_loader as lazy  # type: ignore[import-untyped]

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this."
    ) from exc

_lazy_getattr, _, _ = lazy.attach_stub(__name__, __file__)


def __getattr__(name: str):
    value = _lazy_getattr(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__.copy()


__all__ = [
    "bar",
    "beeswarm",
    "benchmark",
    "decision",
    "embedding",
    "force",
    "initjs",
    "group_difference",
    "heatmap",
    "image",
    "image_to_text",
    "monitoring",
    "partial_dependence",
    "scatter",
    "text",
    "violin",
    "waterfall",
]
