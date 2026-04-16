#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace partition
{
    void lower_credit(
        int i,
        int M,
        nb::ndarray<nb::numpy, double> values,
        nb::ndarray<nb::numpy, double, nb::ndim<2>> clustering);

} // namespace partition
