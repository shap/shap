# Import-based type stub for lazy loading with attach_stub()
# This file tells lazy_loader what to import from which modules

# TreeExplainer and related functions
# Other explainers
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
from ._tree import _check_xgboost_version as _check_xgboost_version
from ._tree import _safe_check_tree_instance_experimental as _safe_check_tree_instance_experimental
from ._tree import _xgboost_cat_unsupported as _xgboost_cat_unsupported
from ._tree import _xgboost_n_iterations as _xgboost_n_iterations
from ._tree import feature_perturbation_codes as feature_perturbation_codes
from ._tree import output_transform_codes as output_transform_codes

# Legacy aliases for backwards compatibility
Tree = TreeExplainer
Additive = AdditiveExplainer
Coalition = CoalitionExplainer
Deep = DeepExplainer
Exact = ExactExplainer
GPUTree = GPUTreeExplainer
Gradient = GradientExplainer
Kernel = KernelExplainer
Linear = LinearExplainer
Partition = PartitionExplainer
Permutation = PermutationExplainer
Sampling = SamplingExplainer
