// see https://nanobind.readthedocs.io/en/latest/basics.html#basics and following docs
#include <nanobind/nanobind.h>
#include "grey_code_utils.h"
#include "kernel_explainer_utils.h"
#include "tabular_utils.h"
#include "clustering_utils.h"

namespace nb = nanobind;

NB_MODULE(_cutils, m)
{
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_1d, "row_values"_a.noconvert(), "mask"_a.noconvert(), "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 1D");
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_2d, "row_values"_a.noconvert(), "mask"_a.noconvert(), "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 2D");
    m.def("compute_grey_code_row_values_st", &compute_grey_code_row_values_st_1d, "row_values"_a.noconvert(), "mask"_a.noconvert(), "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute Shapley-Taylor row values for the grey code algorithm in 1D");
    m.def("compute_grey_code_row_values_st", &compute_grey_code_row_values_st_2d, "row_values"_a.noconvert(), "mask"_a.noconvert(), "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute Shapley-Taylor row values for the grey code algorithm in 2D");
    m.def("compute_exp_val", &compute_exp_val, "nsamples_run"_a, "nsamples_added"_a, "D"_a, "N"_a, "weights"_a, "y"_a, "ey"_a, "Compute the expected value for the kernel explainer algorithm");
    m.def("_delta_masking", &tabular::delta_masking<double>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    m.def("_delta_masking", &tabular::delta_masking<float>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    m.def("_delta_masking", &tabular::delta_masking<int64_t>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    m.def("_delta_masking", &tabular::delta_masking<int32_t>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    m.def("_delta_masking", &tabular::delta_masking<int16_t>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    m.def("_delta_masking", &tabular::delta_masking<int8_t>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    m.def("_delta_masking", &tabular::delta_masking<uint64_t>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    m.def("_delta_masking", &tabular::delta_masking<uint32_t>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    m.def("_delta_masking", &tabular::delta_masking<uint16_t>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    m.def("_delta_masking", &tabular::delta_masking<uint8_t>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    m.def("_delta_masking", &tabular::delta_masking<bool>, "masks"_a.noconvert(), "x"_a.noconvert(), "curr_delta_inds"_a.noconvert(), "varying_rows_out"_a.noconvert(), "masked_inputs_tmp"_a.noconvert(), "last_mask"_a.noconvert(), "data"_a.noconvert(), "variants"_a.noconvert(), "masked_inputs_out"_a.noconvert(), "noop_code"_a, "Apply delta masks to tabular data");
    // `indexes` is written by the C++ and so must be .noconvert(): without it an
    // int32/uint64/float64 array is accepted, written into a temporary cast copy,
    // and silently discarded.
    m.def("_pt_shuffle_rec", &clustering::pt_shuffle_rec, "i"_a, "indexes"_a.noconvert(), "index_mask"_a, "partition_tree"_a, "num_features"_a, "pos"_a, "switches"_a, "Shuffle indexes recursively according to a partition tree");
    m.def("delta_minimization_order", &clustering::delta_minimization_order, "all_masks"_a, "max_swap_size"_a = 100, "num_passes"_a = 2, "Compute the order of elements that minimizes the delta score");
}
