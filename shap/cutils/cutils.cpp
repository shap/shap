#include <nanobind/nanobind.h>
#include "grey_code_utils.h"
#include "kernel_explainer_utils.h"
#include "masked_model_utils.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_cutils, m)
{
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_1d, "row_values"_a, "mask"_a, "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 1D");
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_2d, "row_values"_a, "mask"_a, "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 2D");
    m.def("compute_exp_val", &compute_exp_val, "nsamples_run"_a, "nsamples_added"_a, "D"_a, "N"_a, "weights"_a, "y"_a, "ey"_a, "Compute the expected value for the kernel explainer algorithm");

    // MaskedModel functions
    m.def("init_masks", &detail::init_masks, "cluster_matrix"_a, "M"_a, "indices_row_pos"_a, "indptr"_a, "Initialize masks from a clustering matrix");
    m.def("rec_fill_masks", &detail::rec_fill_masks, "cluster_matrix"_a, "indices_row_pos"_a, "indptr"_a, "indices"_a, "M"_a, "ind"_a, "Recursively fill masks from a clustering matrix");

    // Variadic approach to bypass any dispatcher matching issues
    m.def("build_fixed_single_output", [](nb::args args) {
        if (args.size() != 8) throw std::runtime_error("build_fixed_single_output expected 8 arguments");
        detail::build_fixed_single_output(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
    });

    m.def("build_fixed_multi_output", [](nb::args args) {
        if (args.size() != 8) throw std::runtime_error("build_fixed_multi_output expected 8 arguments");
        detail::build_fixed_multi_output(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
    });
}
