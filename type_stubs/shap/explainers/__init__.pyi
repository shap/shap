# Type stubs for shap.explainers
from ._additive import AdditiveExplainer as AdditiveExplainer
from ._coalition import CoalitionExplainer as CoalitionExplainer
from ._deep import DeepExplainer as DeepExplainer
from ._exact import ExactExplainer as ExactExplainer
from ._explainer import Explainer as Explainer
from ._gpu_tree import GPUTreeExplainer as GPUTreeExplainer
from ._gradient import GradientExplainer as GradientExplainer
from ._kernel import KernelExplainer as KernelExplainer
from ._linear import LinearExplainer as LinearExplainer
from ._partition import PartitionExplainer as PartitionExplainer
from ._permutation import PermutationExplainer as PermutationExplainer
from ._sampling import SamplingExplainer as SamplingExplainer
from ._tree import TreeExplainer as TreeExplainer

# Legacy aliases
Additive = AdditiveExplainer
Deep = DeepExplainer
Exact = ExactExplainer
GPUTree = GPUTreeExplainer
Gradient = GradientExplainer
Kernel = KernelExplainer
Linear = LinearExplainer
Partition = PartitionExplainer
Coalition = CoalitionExplainer
Permutation = PermutationExplainer
Sampling = SamplingExplainer
Tree = TreeExplainer
