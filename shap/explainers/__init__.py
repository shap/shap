# from ._additive import AdditiveExplainer
# from ._coalition import CoalitionExplainer
# from ._deep import DeepExplainer
# from ._exact import ExactExplainer
# from ._gpu_tree import GPUTreeExplainer
# from ._gradient import GradientExplainer
# from ._kernel import KernelExplainer
# from ._linear import LinearExplainer
# from ._partition import PartitionExplainer
# from ._permutation import PermutationExplainer
# from ._sampling import SamplingExplainer

# Lazy import for TreeExplainer to improve import performance
import lazy_loader as lazy

_lazy_getattr, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "_tree": ["TreeExplainer"],
        "_additive": ["AdditiveExplainer"],
        "_coalition": ["CoalitionExplainer"],
        "_deep": ["DeepExplainer"],
        "_exact": ["ExactExplainer"],
        "_gpu_tree": ["GPUTreeExplainer"],
        "_gradient": ["GradientExplainer"],
        "_kernel": ["KernelExplainer"],
        "_linear": ["LinearExplainer"],
        "_partition": ["PartitionExplainer"],
        "_permutation": ["PermutationExplainer"],
        "_sampling": ["SamplingExplainer"],
        "_explainer": ["Explainer"],
        "_tabular": ["Partition"],
    },
)


# Create custom __getattr__ to handle both lazy loading and legacy aliases
def __getattr__(name):
    # Handle legacy aliases first
    if name == "Tree":
        return _lazy_getattr("TreeExplainer")
    elif name == "Additive":
        return _lazy_getattr("AdditiveExplainer")
    elif name == "Coalition":
        return _lazy_getattr("CoalitionExplainer")
    elif name == "Deep":
        return _lazy_getattr("DeepExplainer")
    elif name == "Exact":
        return _lazy_getattr("ExactExplainer")
    elif name == "GPUTree":
        return _lazy_getattr("GPUTreeExplainer")
    elif name == "Gradient":
        return _lazy_getattr("GradientExplainer")
    elif name == "Kernel":
        return _lazy_getattr("KernelExplainer")
    elif name == "Linear":
        return _lazy_getattr("LinearExplainer")
    elif name == "Partition":
        return _lazy_getattr("PartitionExplainer")
    elif name == "Permutation":
        return _lazy_getattr("PermutationExplainer")
    elif name == "Sampling":
        return _lazy_getattr("SamplingExplainer")
    elif name == "Explainer":
        return _lazy_getattr("Explainer")
    elif name == "Partition":
        return _lazy_getattr("PartitionExplainer")
    # Fall back to lazy loader
    return _lazy_getattr(name)


def __dir__():
    return _lazy_dir() + ["Tree"]


__all__ = _lazy_all + ["Tree"]

# Alternative legacy "short-form" aliases, which are kept here for backwards-compatibility
# Additive = AdditiveExplainer
# Deep = DeepExplainer
# Exact = ExactExplainer
# GPUTree = GPUTreeExplainer
# Gradient = GradientExplainer
# Kernel = KernelExplainer
# Linear = LinearExplainer
# Partition = PartitionExplainer
# Coalition = CoalitionExplainer
# Permutation = PermutationExplainer
# Sampling = SamplingExplainer
# Tree = TreeExplainer  # This will be handled by lazy loading

# Note: __all__ is now managed by lazy_loader.attach()
