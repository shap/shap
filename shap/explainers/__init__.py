from ._additive import AdditiveExplainer
from ._deep import DeepExplainer
from ._exact import ExactExplainer
from ._gpu_tree import GPUTreeExplainer
from ._gradient import GradientExplainer
from ._kernel import KernelExplainer
from ._linear import LinearExplainer
from ._partition import PartitionExplainer
from ._permutation import PermutationExplainer
from ._sampling import SamplingExplainer
from ._tree import TreeExplainer

# Alternative legacy "short-form" aliases, which are kept here for backwards-compatibility
Additive = AdditiveExplainer
Deep = DeepExplainer
Exact = ExactExplainer
GPUTree = GPUTreeExplainer
Gradient = GradientExplainer
Kernel = KernelExplainer
Linear = LinearExplainer
Partition = PartitionExplainer
Permutation = PermutationExplainer
Sampling = SamplingExplainer
Tree = TreeExplainer

__all__ = [
    "AdditiveExplainer",
    "DeepExplainer",
    "ExactExplainer",
    "GPUTreeExplainer",
    "GradientExplainer",
    "KernelExplainer",
    "LinearExplainer",
    "PartitionExplainer",
    "PermutationExplainer",
    "SamplingExplainer",
    "TreeExplainer",
]
