#include <nanobind/nanobind.h>
#include "partition.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(_ext, m)
{
    m.doc() = "Internal C++ implementation of SHAP algorithms";
    m.def("lower_credit", &partition::lower_credit, "i"_a, "M"_a, "values"_a, "clustering"_a);
}
