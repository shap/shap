#ifndef PARTITION_EXPLAINER_UTILS_H
#define PARTITION_EXPLAINER_UTILS_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <vector>

namespace nb = nanobind;

namespace partition
{
    // The child's additive value is computed at the parent as
    // values[i] * lsize / group_size (multiply, then divide), exactly like the
    // numba implementation this replaces: reassociating it changes the rounding
    // and breaks bitwise parity with previously computed SHAP values.

    inline void check_node(int i, int M, int n_values, int n_clusters, int depth_left)
    {
        if (i < 0 || i >= n_values)
        {
            throw nb::index_error("clustering refers to a node outside the values array");
        }
        if (i >= M && i - M >= n_clusters)
        {
            throw nb::index_error("clustering refers to a row outside the clustering array");
        }
        if (depth_left <= 0)
        {
            throw nb::value_error("clustering is not a tree (recursion exceeded the node count)");
        }
    }

    // Reads the child indices and validates the subtree sizes; returns them via out-params.
    template <typename ClusteringView>
    void read_children(const ClusteringView &c, int i, int M, int n_values, int n_clusters,
                       int &li, int &ri, int &lsize, int &rsize, int &group_size)
    {
        li = static_cast<int>(c(i - M, 0));
        ri = static_cast<int>(c(i - M, 1));
        group_size = static_cast<int>(c(i - M, 3));

        if (li < 0 || li >= n_values || ri < 0 || ri >= n_values ||
            (li >= M && li - M >= n_clusters) || (ri >= M && ri - M >= n_clusters))
        {
            throw nb::index_error("clustering refers to a node outside the values array");
        }

        lsize = li >= M ? static_cast<int>(c(li - M, 3)) : 1;
        rsize = ri >= M ? static_cast<int>(c(ri - M, 3)) : 1;

        if (lsize + rsize != group_size)
        {
            throw nb::value_error("left and right cluster sizes do not match parent group size");
        }
    }

    template <typename ValuesView, typename ClusteringView>
    void lower_credit_1d_rec(int i, double value, int M, ValuesView &v, ClusteringView &c,
                             int n_values, int n_clusters, int depth_left)
    {
        check_node(i, M, n_values, n_clusters, depth_left);
        if (i < M)
        {
            v(i) += value;
            return;
        }

        int li, ri, lsize, rsize, group_size;
        read_children(c, i, M, n_values, n_clusters, li, ri, lsize, rsize, group_size);

        v(i) += value;
        lower_credit_1d_rec(li, v(i) * lsize / group_size, M, v, c, n_values, n_clusters, depth_left - 1);
        lower_credit_1d_rec(ri, v(i) * rsize / group_size, M, v, c, n_values, n_clusters, depth_left - 1);
    }

    void lower_credit_1d(
        int i,
        double value,
        int M,
        nb::ndarray<double, nb::ndim<1>, nb::device::cpu> values,
        nb::ndarray<double, nb::shape<-1, 4>, nb::device::cpu> clustering)
    {
        auto v = values.view();
        auto c = clustering.view();
        const int n_values = static_cast<int>(values.shape(0));
        const int n_clusters = static_cast<int>(clustering.shape(0));
        if (M < 0 || M > n_values)
        {
            throw nb::value_error("M must lie within the values array");
        }
        lower_credit_1d_rec(i, value, M, v, c, n_values, n_clusters, n_values);
    }

    template <typename ValuesView, typename ClusteringView>
    void lower_credit_2d_rec(int i, const std::vector<double> &value, int M, ValuesView &v, ClusteringView &c,
                             int n_values, int n_clusters, int depth_left)
    {
        check_node(i, M, n_values, n_clusters, depth_left);
        const size_t cols = static_cast<size_t>(v.shape(1));
        if (i < M)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                v(i, j) += value[j];
            }
            return;
        }

        int li, ri, lsize, rsize, group_size;
        read_children(c, i, M, n_values, n_clusters, li, ri, lsize, rsize, group_size);

        for (size_t j = 0; j < cols; ++j)
        {
            v(i, j) += value[j];
        }

        std::vector<double> child(cols);
        for (size_t j = 0; j < cols; ++j)
        {
            child[j] = v(i, j) * lsize / group_size;
        }
        lower_credit_2d_rec(li, child, M, v, c, n_values, n_clusters, depth_left - 1);
        for (size_t j = 0; j < cols; ++j)
        {
            child[j] = v(i, j) * rsize / group_size;
        }
        lower_credit_2d_rec(ri, child, M, v, c, n_values, n_clusters, depth_left - 1);
    }

    void lower_credit_2d(
        int i,
        double value,
        int M,
        nb::ndarray<double, nb::ndim<2>, nb::device::cpu> values,
        nb::ndarray<double, nb::shape<-1, 4>, nb::device::cpu> clustering)
    {
        auto v = values.view();
        auto c = clustering.view();
        const int n_values = static_cast<int>(values.shape(0));
        const int n_clusters = static_cast<int>(clustering.shape(0));
        if (M < 0 || M > n_values)
        {
            throw nb::value_error("M must lie within the values array");
        }
        const std::vector<double> value_per_output(static_cast<size_t>(values.shape(1)), value);
        lower_credit_2d_rec(i, value_per_output, M, v, c, n_values, n_clusters, n_values);
    }

} // namespace partition

#endif // PARTITION_EXPLAINER_UTILS_H
