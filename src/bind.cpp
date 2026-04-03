#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_ext, m)
{
    m.doc() = "Internal C++ implementation of SHAP algorithms";
    // partition::bind(m);
}
