# Lazy import for explainers to improve import performance
import lazy_loader as lazy

# Use lazy.attach_stub for all explainers to enable proper type checking
# attach_stub expects __file__ and will look for adjacent .pyi file
_stub_getattr, _stub_dir, _stub_all = lazy.attach_stub(__name__, __file__)


# Legacy aliases for backwards compatibility
def __getattr__(name):
    # Handle legacy aliases first
    if name == "Tree":
        return _stub_getattr("TreeExplainer")
    elif name == "Additive":
        return _stub_getattr("AdditiveExplainer")
    elif name == "Coalition":
        return _stub_getattr("CoalitionExplainer")
    elif name == "Deep":
        return _stub_getattr("DeepExplainer")
    elif name == "Exact":
        return _stub_getattr("ExactExplainer")
    elif name == "GPUTree":
        return _stub_getattr("GPUTreeExplainer")
    elif name == "Gradient":
        return _stub_getattr("GradientExplainer")
    elif name == "Kernel":
        return _stub_getattr("KernelExplainer")
    elif name == "Linear":
        return _stub_getattr("LinearExplainer")
    elif name == "Partition":
        return _stub_getattr("PartitionExplainer")
    elif name == "Permutation":
        return _stub_getattr("PermutationExplainer")
    elif name == "Sampling":
        return _stub_getattr("SamplingExplainer")
    else:
        # Fall back to stub-based lazy loader
        return _stub_getattr(name)


def __dir__():
    return list(set(_stub_dir() + ["Tree"]))


__all__ = list(set(_stub_all + ["Tree"]))

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
