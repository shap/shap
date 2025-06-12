import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "_clustering": [
            "delta_minimization_order",
            "hclust",
            "hclust_ordering",
            "partition_tree",
            "partition_tree_shuffle",
        ],
        "_general": [
            "OpChain",
            "approximate_interactions",
            "assert_import",
            "convert_name",
            "format_value",
            "ordinal_str",
            "potential_interactions",
            "record_import_error",
            "safe_isinstance",
            "sample",
            "shapley_coefficients",
            "suppress_stderr",
        ],
        "_masked_model": ["MaskedModel", "make_masks"],
        "_show_progress": ["show_progress"],
    },
)
# from ._clustering import (
#     delta_minimization_order,
#     hclust,
#     hclust_ordering,
#     partition_tree,
#     partition_tree_shuffle,
# )
# from ._general import (
#     OpChain,
#     approximate_interactions,
#     assert_import,
#     convert_name,
#     format_value,
#     ordinal_str,
#     potential_interactions,
#     record_import_error,
#     safe_isinstance,
#     sample,
#     shapley_coefficients,
#     suppress_stderr,
# )
# from ._masked_model import MaskedModel, make_masks
# from ._show_progress import show_progress
#
# __all__ = [
#     "delta_minimization_order",
#     "hclust",
#     "hclust_ordering",
#     "partition_tree",
#     "partition_tree_shuffle",
#     "OpChain",
#     "approximate_interactions",
#     "assert_import",
#     "convert_name",
#     "format_value",
#     "ordinal_str",
#     "potential_interactions",
#     "record_import_error",
#     "safe_isinstance",
#     "sample",
#     "shapley_coefficients",
#     "suppress_stderr",
#     "MaskedModel",
#     "make_masks",
#     "show_progress",
# ]
