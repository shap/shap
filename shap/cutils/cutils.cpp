// see https://nanobind.readthedocs.io/en/latest/basics.html#basics and following docs
#include <nanobind/nanobind.h>
#include "grey_code_utils.h"
#include "kernel_explainer_utils.h"
#include "tabular_utils.h"

namespace nb = nanobind;

NB_MODULE(_cutils, m)
{
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_1d, "row_values"_a, "mask"_a, "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 1D");
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_2d, "row_values"_a, "mask"_a, "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 2D");
    m.def("compute_exp_val", &compute_exp_val, "nsamples_run"_a, "nsamples_added"_a, "D"_a, "N"_a, "weights"_a, "y"_a, "ey"_a, "Compute the expected value for the kernel explainer algorithm");
    m.def("delta_masking", &tabular::delta_masking<double>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
    m.def("delta_masking", &tabular::delta_masking<float>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
    m.def("delta_masking", &tabular::delta_masking<int64_t>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
    m.def("delta_masking", &tabular::delta_masking<int32_t>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
    m.def("delta_masking", &tabular::delta_masking<int16_t>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
    m.def("delta_masking", &tabular::delta_masking<int8_t>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
    m.def("delta_masking", &tabular::delta_masking<uint64_t>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
    m.def("delta_masking", &tabular::delta_masking<uint32_t>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
    m.def("delta_masking", &tabular::delta_masking<uint16_t>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
    m.def("delta_masking", &tabular::delta_masking<uint8_t>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
    m.def("delta_masking", &tabular::delta_masking<bool>, "masks"_a, "x"_a, "curr_delta_inds"_a, "varying_rows_out"_a, "masked_inputs_tmp"_a, "last_mask"_a, "data"_a, "variants"_a, "masked_inputs_out"_a, "noop_code"_a, "Apply delta masks to tabular data");
}
