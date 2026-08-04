// see https://nanobind.readthedocs.io/en/latest/basics.html#basics and following docs
#include <nanobind/nanobind.h>
#include "grey_code_utils.h"
#include "kernel_explainer_utils.h"
#include "masked_model_utils.h"

namespace nb = nanobind;

NB_MODULE(_cutils, m)
{
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_1d, "row_values"_a, "mask"_a, "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 1D");
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_2d, "row_values"_a, "mask"_a, "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 2D");
    m.def("compute_exp_val", &compute_exp_val, "nsamples_run"_a, "nsamples_added"_a, "D"_a, "N"_a, "weights"_a, "y"_a, "ey"_a, "Compute the expected value for the kernel explainer algorithm");
    m.def("build_fixed_single_output", &masked_model::build_fixed_single_output<double>, "averaged_outs"_a, "last_outs"_a, "outputs"_a, "batch_positions"_a, "varying_rows"_a, "num_varying_rows"_a);
    m.def("build_fixed_single_output", &masked_model::build_fixed_single_output<float>, "averaged_outs"_a, "last_outs"_a, "outputs"_a, "batch_positions"_a, "varying_rows"_a, "num_varying_rows"_a);
    m.def("build_fixed_single_output", &masked_model::build_fixed_single_output_weighted<double>, "averaged_outs"_a, "last_outs"_a, "outputs"_a, "batch_positions"_a, "varying_rows"_a, "num_varying_rows"_a, "linearizing_weights"_a);
    m.def("build_fixed_single_output", &masked_model::build_fixed_single_output_weighted<float>, "averaged_outs"_a, "last_outs"_a, "outputs"_a, "batch_positions"_a, "varying_rows"_a, "num_varying_rows"_a, "linearizing_weights"_a);
    m.def("build_fixed_multi_output", &masked_model::build_fixed_multi_output<double>, "averaged_outs"_a, "last_outs"_a, "outputs"_a, "batch_positions"_a, "varying_rows"_a, "num_varying_rows"_a);
    m.def("build_fixed_multi_output", &masked_model::build_fixed_multi_output<float>, "averaged_outs"_a, "last_outs"_a, "outputs"_a, "batch_positions"_a, "varying_rows"_a, "num_varying_rows"_a);
    m.def("build_fixed_multi_output", &masked_model::build_fixed_multi_output_weighted<double>, "averaged_outs"_a, "last_outs"_a, "outputs"_a, "batch_positions"_a, "varying_rows"_a, "num_varying_rows"_a, "linearizing_weights"_a);
    m.def("build_fixed_multi_output", &masked_model::build_fixed_multi_output_weighted<float>, "averaged_outs"_a, "last_outs"_a, "outputs"_a, "batch_positions"_a, "varying_rows"_a, "num_varying_rows"_a, "linearizing_weights"_a);
}
