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
        v(i) += value;

        if (i < M)
        {
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

        const double left_value = v(i) * static_cast<double>(lsize) / static_cast<double>(group_size);
        const double right_value = v(i) * static_cast<double>(rsize) / static_cast<double>(group_size);

        lower_credit_1d(li, left_value, M, values, clustering);
        lower_credit_1d(ri, right_value, M, values, clustering);
    }

    void lower_credit_2d(
        int i,
        int M,
        int prev_i,
        double factor,
        nb::ndarray<nb::numpy, double> values,
        nb::ndarray<nb::numpy, double, nb::ndim<2>> clustering)
    {
        auto v = values.view<double, nb::ndim<2>>();
        for (int j = 0; j < v.shape(1); ++j)
        {
            v(i, j) += factor * v(prev_i, j);
        }
        if (i < M)
        {
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

        const double left_factor = static_cast<double>(lsize) / static_cast<double>(group_size);
        const double right_factor = static_cast<double>(rsize) / static_cast<double>(group_size);

        lower_credit_2d(li, M, i, left_factor, values, clustering);
        lower_credit_2d(ri, M, i, right_factor, values, clustering);
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
        else if (values.ndim() == 2)
        {
            lower_credit_2d(i, M, i, 0.0, values, clustering);
        }
        else
        {
            throw nb::value_error("values array must be 1-dimensional or 2-dimensional");
        }
    }

} // namespace partition
