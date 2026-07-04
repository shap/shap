# Explainers that use Numba-backed functions in the active source package:

# - [ExactExplainer](shap/explainers/_exact.py:24)
#   Uses `MaskedModel`, `_compute_grey_code_row_values_st`, `make_masks`, and `delta_minimization_order`.

# - [PartitionExplainer](shap/explainers/_partition.py:17)
#   Uses `MaskedModel`, `make_masks`, and the local Numba function `lower_credit`.

# - [PermutationExplainer](shap/explainers/_permutation.py:16)
#   Uses `MaskedModel`; also uses `partition_tree_shuffle`, which calls the Numba helper `_pt_shuffle_rec` when clustering is present.

# - [AdditiveExplainer](shap/explainers/_additive.py:10)
#   Uses `MaskedModel`, which calls the Numba output aggregation helpers `_build_fixed_single_output` / `_build_fixed_multi_output`.

# - [CoalitionExplainer](shap/explainers/_coalition.py:20)
#   Uses `MaskedModel` and `make_masks`.

# - [Random](shap/explainers/other/_random.py:19)
#   Uses `MaskedModel`.

# I did not count `KernelExplainer`: although it has `"identity"` / `"logit"` links, it uses legacy `IdentityLink` / `LogitLink` objects, not the Numba-decorated `shap.links.identity` / `shap.links.logit`.

# I also did not count `LinearExplainer` on algorithm-helper usage. It has a default `links.identity` constructor argument, but it does not call `MaskedModel` or the Numba helper paths above.
