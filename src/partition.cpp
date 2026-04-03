#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace partition
{
    void lower_credit_1d(
        int i,
        double value,
        int M,
        nb::ndarray<nb::numpy, double> values,
        nb::ndarray<nb::numpy, double, nb::ndim<2>> clustering)
    {
        auto v = values.view<double, nb::ndim<1>>();
        if (i < M)
        {
            v(i) += value;
            return;
        }

        auto c = clustering.view<double, nb::ndim<2>>();

        const int li = static_cast<int>(c(i - M, 0));
        const int ri = static_cast<int>(c(i - M, 1));
        const int group_size = static_cast<int>(c(i - M, 3));
        const int lsize = li >= M ? static_cast<int>(c(li - M, 3)) : 1;
        const int rsize = ri >= M ? static_cast<int>(c(ri - M, 3)) : 1;

        if (lsize + rsize != group_size)
        {
            throw nb::value_error("left and right cluster sizes do not match parent group size");
        }

        v(i) += value;

        const double left_value = v(i) * static_cast<double>(lsize) / static_cast<double>(group_size);
        const double right_value = v(i) * static_cast<double>(rsize) / static_cast<double>(group_size);

        lower_credit_1d(li, left_value, M, values, clustering);
        lower_credit_1d(ri, right_value, M, values, clustering);
    }

    void lower_credit(
        int i,
        int M,
        nb::ndarray<nb::numpy, double> values,
        nb::ndarray<nb::numpy, double, nb::ndim<2>> clustering)
    {
        if (values.ndim() == 1)
        {
            lower_credit_1d(i, 0, M, values, clustering);
        }
        else
        {
            throw nb::value_error("values array must be 1-dimensional");
        }
    }

} // namespace partition
