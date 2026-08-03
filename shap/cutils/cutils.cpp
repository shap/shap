// see https://nanobind.readthedocs.io/en/latest/basics.html#basics and following docs
#include <nanobind/nanobind.h>
#include "grey_code_utils.h"
#include "kernel_explainer_utils.h"
#include "make_masks_utils.h"

namespace nb = nanobind;

NB_MODULE(_cutils, m)
{
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_1d, "row_values"_a, "mask"_a, "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 1D");
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_2d, "row_values"_a, "mask"_a, "inds"_a, "outputs"_a, "shapley_coeff"_a, "extended_delta_indexes"_a, "noop_code"_a, "Compute the row values for the grey code algorithm in 2D");
    m.def("compute_exp_val", &compute_exp_val, "nsamples_run"_a, "nsamples_added"_a, "D"_a, "N"_a, "weights"_a, "y"_a, "ey"_a, "Compute the expected value for the kernel explainer algorithm");
    m.def("init_masks", &masks::init_masks, "cluster_matrix"_a, "M"_a, "indices_row_pos"_a, "indptr"_a, "Initialize the masks for the kernel explainer algorithm");
    m.def("rec_fill_masks", &masks::rec_fill_masks, "cluster_matrix"_a, "indices_row_pos"_a, "indices"_a, "M"_a, "ind"_a, "Fill the masks for the partition explainer algorithm");
}
