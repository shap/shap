import lazy_loader as lazy  # type: ignore[import-untyped]

_lazy_getattr, _, _ = lazy.attach_stub(__name__, __file__)
_EXTRA_NAMES = {"other", "Additive", "Deep", "Exact", "GPUTree", "Gradient", "Kernel", "Linear", "Partition", "Coalition", "Permutation", "Sampling", "Tree"}


def __getattr__(name: str):
    value = _lazy_getattr(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | _EXTRA_NAMES)

__all__ = [
    "AdditiveExplainer",
    "DeepExplainer",
    "ExactExplainer",
    "GPUTreeExplainer",
    "GradientExplainer",
    "KernelExplainer",
    "LinearExplainer",
    "PartitionExplainer",
    "CoalitionExplainer",
    "PermutationExplainer",
    "SamplingExplainer",
    "TreeExplainer",
]
