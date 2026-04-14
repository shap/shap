// see https://nanobind.readthedocs.io/en/latest/basics.html#basics and following docs
#include <nanobind/nanobind.h>
#include "grey_code_utils.h"
#include "kernel_explainer_utils.h"
#include "clustering_utils.h"

namespace nb = nanobind;

NB_MODULE(_cutils, m)
{
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_1d);
    m.def("compute_grey_code_row_values", &compute_grey_code_row_values_2d);
    m.def("compute_exp_val", &compute_exp_val);
    m.def("reverse_window", &reverse_window, "order"_a, "start"_a, "length"_a);
}
