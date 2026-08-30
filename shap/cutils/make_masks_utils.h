#ifndef MAKE_MASKS_UTILS_H
#define MAKE_MASKS_UTILS_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace masks {
    void init_masks(
        const nb::ndarray<double, nb::shape<-1, 4>, nb::device::cpu>& cluster_matrix,
        const int64_t M,
        nb::ndarray<int64_t, nb::ndim<1>, nb::device::cpu>& indices_row_pos,
        nb::ndarray<int64_t, nb::ndim<1>, nb::device::cpu>& indptr
    ) {
        if (M < 1 || cluster_matrix.shape(0) != static_cast<size_t>(M - 1)) {
            throw nb::value_error("M must equal cluster_matrix.shape[0] + 1");
        }
        if (indices_row_pos.shape(0) < static_cast<size_t>(2 * M - 1) || indptr.shape(0) < static_cast<size_t>(2 * M)) {
            throw nb::value_error("indices_row_pos or indptr is too small for the given M");
        }

        int64_t pos = 0;
        auto c = cluster_matrix.view();
        auto irp = indices_row_pos.view();
        auto ip = indptr.view();

        for (int64_t i = 0; i < 2 * M - 1; i++) {
            if (i < M) {
                pos++;
            } else {
                pos += static_cast<int64_t>(c(i - M, 3));
            }
            ip(i+1) = pos;
            irp(i) = ip(i);
        }
    }

    namespace detail {
        // a degenerate chain clustering ~100k leaves deep overflows the native
        // stack (segfault); realistic linkage trees are orders of magnitude
        // shallower, so fail cleanly long before that point
        constexpr int64_t max_fill_depth = 30000;

        template <class Cluster, class RowPos, class Indices>
        void fill_node(const Cluster& c, const RowPos& irp, Indices& idx, int64_t M, int64_t ind, int64_t depth);

        // phase 1: fill the left subtree, then copy its finished block to the
        // start of this node's block
        template <class Cluster, class RowPos, class Indices>
        void fill_left_block(const Cluster& c, const RowPos& irp, Indices& idx, const int64_t M, const int64_t ind, const int64_t depth) {
            const int64_t lind = static_cast<int64_t>(c(ind - M, 0));
            const int64_t lind_size = lind >= M ? static_cast<int64_t>(c(lind - M, 3)) : 1;

            fill_node(c, irp, idx, M, lind, depth);

            const int64_t pos = irp(ind);
            const int64_t lpos = irp(lind);
            for (int64_t k = 0; k < lind_size; k++) {
                idx(pos + k) = idx(lpos + k);
            }
        }

        // phase 2: fill the right subtree, then copy its finished block just
        // after the left child's block
        template <class Cluster, class RowPos, class Indices>
        void fill_right_block(const Cluster& c, const RowPos& irp, Indices& idx, const int64_t M, const int64_t ind, const int64_t depth) {
            const int64_t lind = static_cast<int64_t>(c(ind - M, 0));
            const int64_t rind = static_cast<int64_t>(c(ind - M, 1));
            const int64_t lind_size = lind >= M ? static_cast<int64_t>(c(lind - M, 3)) : 1;
            const int64_t rind_size = rind >= M ? static_cast<int64_t>(c(rind - M, 3)) : 1;

            fill_node(c, irp, idx, M, rind, depth);

            const int64_t pos = irp(ind);
            const int64_t rpos = irp(rind);
            for (int64_t k = 0; k < rind_size; k++) {
                idx(pos + lind_size + k) = idx(rpos + k);
            }
        }

        template <class Cluster, class RowPos, class Indices>
        void fill_node(const Cluster& c, const RowPos& irp, Indices& idx, const int64_t M, const int64_t ind, const int64_t depth) {
            if (depth > max_fill_depth) {
                throw nb::value_error("clustering is too deep to fill recursively");
            }
            if (ind < M) {
                idx(irp(ind)) = ind;
                return;
            }
            fill_left_block(c, irp, idx, M, ind, depth + 1);
            fill_right_block(c, irp, idx, M, ind, depth + 1);
        }
    }

    void rec_fill_masks(
        const nb::ndarray<double, nb::shape<-1, 4>, nb::device::cpu>& cluster_matrix,
        const nb::ndarray<int64_t, nb::ndim<1>, nb::device::cpu>& indices_row_pos,
        nb::ndarray<int64_t, nb::ndim<1>, nb::device::cpu>& indices,
        const int64_t M,
        const int64_t ind
    ) {
        if (M < 1 || cluster_matrix.shape(0) != static_cast<size_t>(M - 1)) {
            throw nb::value_error("M must equal cluster_matrix.shape[0] + 1");
        }
        if (ind < 0 || ind >= 2 * M - 1 || indices_row_pos.shape(0) < static_cast<size_t>(2 * M - 1)) {
            throw nb::value_error("ind or indices_row_pos does not match the given M");
        }

        auto c = cluster_matrix.view();
        auto irp = indices_row_pos.view();
        auto idx = indices.view();

        // validate every node once up front so the recursion itself can index
        // unchecked: each node's block must fit in `indices`, and each internal
        // node's children must be real node ids
        const int64_t n_nodes = 2 * M - 1;
        const int64_t n_indices = static_cast<int64_t>(indices.shape(0));
        for (int64_t i = 0; i < n_nodes; i++) {
            const int64_t size = i >= M ? static_cast<int64_t>(c(i - M, 3)) : 1;
            const int64_t pos = irp(i);
            if (pos < 0 || size < 1 || pos + size > n_indices) {
                throw nb::index_error("mask block lies outside the indices array");
            }
            if (i >= M) {
                const int64_t lind = static_cast<int64_t>(c(i - M, 0));
                const int64_t rind = static_cast<int64_t>(c(i - M, 1));
                if (lind < 0 || lind >= n_nodes || rind < 0 || rind >= n_nodes) {
                    throw nb::index_error("clustering refers to a node outside the mask arrays");
                }
            }
        }

        detail::fill_node(c, irp, idx, M, ind, 0);
    }
}

#endif // MAKE_MASKS_UTILS_H
