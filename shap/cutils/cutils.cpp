// see https://nanobind.readthedocs.io/en/latest/basics.html#basics and following docs
#include <nanobind/nanobind.h>
#include "grey_code_utils.h"
#include "kernel_explainer_utils.h"
#include "clustering_utils.h"

namespace nb = nanobind;

NB_MODULE(_cutils, m)
{
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_1d, "row_values"_a, "mask"_a, "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 1D");
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_2d, "row_values"_a, "mask"_a, "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 2D");
    m.def("compute_exp_val", &compute_exp_val, "nsamples_run"_a, "nsamples_added"_a, "D"_a, "N"_a, "weights"_a, "y"_a, "ey"_a, "Compute the expected value for the kernel explainer algorithm");
    // `indexes` is written by the C++ and so must be .noconvert(): without it an
    // int32/uint64/float64 array is accepted, written into a temporary cast copy,
    // and silently discarded.
    m.def("_pt_shuffle_rec", &clustering::pt_shuffle_rec, "i"_a, "indexes"_a.noconvert(), "index_mask"_a, "partition_tree"_a, "num_features"_a, "pos"_a, "switches"_a, "Shuffle indexes recursively according to a partition tree");
    m.def("delta_minimization_order", &clustering::delta_minimization_order, "all_masks"_a, "max_swap_size"_a = 100, "num_passes"_a = 2, "Compute the order of elements that minimizes the delta score");
}
