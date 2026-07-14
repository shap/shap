from importlib import import_module
from typing import Any

_EXPLAINER_MAP = {
    "AdditiveExplainer": ("shap.explainers._additive", "AdditiveExplainer"),
    "CoalitionExplainer": ("shap.explainers._coalition", "CoalitionExplainer"),
    "DeepExplainer": ("shap.explainers._deep", "DeepExplainer"),
    "ExactExplainer": ("shap.explainers._exact", "ExactExplainer"),
    "GPUTreeExplainer": ("shap.explainers._gpu_tree", "GPUTreeExplainer"),
    "GradientExplainer": ("shap.explainers._gradient", "GradientExplainer"),
    "KernelExplainer": ("shap.explainers._kernel", "KernelExplainer"),
    "LinearExplainer": ("shap.explainers._linear", "LinearExplainer"),
    "PartitionExplainer": ("shap.explainers._partition", "PartitionExplainer"),
    "PermutationExplainer": ("shap.explainers._permutation", "PermutationExplainer"),
    "SamplingExplainer": ("shap.explainers._sampling", "SamplingExplainer"),
    "TreeExplainer": ("shap.explainers._tree", "TreeExplainer"),
}

_ALIAS_MAP = {
    "Additive": "AdditiveExplainer",
    "Deep": "DeepExplainer",
    "Exact": "ExactExplainer",
    "GPUTree": "GPUTreeExplainer",
    "Gradient": "GradientExplainer",
    "Kernel": "KernelExplainer",
    "Linear": "LinearExplainer",
    "Partition": "PartitionExplainer",
    "Coalition": "CoalitionExplainer",
    "Permutation": "PermutationExplainer",
    "Sampling": "SamplingExplainer",
    "Tree": "TreeExplainer",
}


def __getattr__(name: str) -> Any:
    if name in _EXPLAINER_MAP:
        module_name, attr_name = _EXPLAINER_MAP[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    if name in _ALIAS_MAP:
        value = __getattr__(_ALIAS_MAP[name])
        globals()[name] = value
        return value
    raise AttributeError(f"No {__name__} attribute {name}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPLAINER_MAP.keys()) + list(_ALIAS_MAP.keys()))


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
