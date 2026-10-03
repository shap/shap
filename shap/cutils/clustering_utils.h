#ifndef CLUSTERING_UTILS_H
#define CLUSTERING_UTILS_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <stdexcept>

namespace nb = nanobind;

namespace clustering {
    // Argument types follow the conventions already used by grey_code_utils.h
    // and kernel_explainer_utils.h: explicit extents plus nb::device::cpu. The
    // numba implementations these replaced were host-only, so requiring CPU
    // memory loses nothing; without the tag a CUDA tensor is accepted and its
    // device pointer dereferenced on the host.
    using IndexArray = nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>;
    using BoolArray = nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>;
    using TreeArray = nb::ndarray<double, nb::shape<-1, 4>, nb::device::cpu>;
    using MasksArray = nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cpu>;
    // Allocated and returned by delta_minimization_order, so it carries the
    // nb::numpy tag that turns it into an ndarray on the way back to Python.
    using OrderArray = nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>;

    namespace detail {
        // Unchecked. pt_shuffle_rec validates the whole tree before recursing,
        // which bounds every index used below. The exception is `pos`, whose
        // bound also depends on no leaf being named twice, so it is checked at
        // the single write site.
        int64_t pt_shuffle_rec(
            const int64_t i,
            IndexArray& indexes,
            const BoolArray& index_mask,
            const TreeArray& partition_tree,
            const int64_t num_features,
            const int64_t pos,
            const BoolArray& switches
        ) {
            auto inds = indexes.view();
            auto mask = index_mask.view();
            auto tree = partition_tree.view();
            auto branch_switches = switches.view();

            if (i < 0) {
                const int64_t feature_index = i + num_features;
                if (mask(feature_index)) {
                    if (pos >= static_cast<int64_t>(indexes.shape(0))) {
                        throw std::invalid_argument(
                            "pt_shuffle_rec: indexes is too small to hold the shuffled result"
                        );
                    }
                    inds(pos) = feature_index;
                    return pos + 1;
                }
                return pos;
            }

            const int64_t left = static_cast<int64_t>(tree(i, 0)) - num_features;
            const int64_t right = static_cast<int64_t>(tree(i, 1)) - num_features;
            int64_t next_pos = pos;
            if (branch_switches(i)) {
                next_pos = pt_shuffle_rec(left, indexes, index_mask, partition_tree, num_features, next_pos, switches);
                next_pos = pt_shuffle_rec(right, indexes, index_mask, partition_tree, num_features, next_pos, switches);
            } else {
                next_pos = pt_shuffle_rec(right, indexes, index_mask, partition_tree, num_features, next_pos, switches);
                next_pos = pt_shuffle_rec(left, indexes, index_mask, partition_tree, num_features, next_pos, switches);
            }
            return next_pos;
        }
    }

    int64_t pt_shuffle_rec(
        const int64_t i,
        IndexArray& indexes,
        const BoolArray& index_mask,
        const TreeArray& partition_tree,
        const int64_t num_features,
        const int64_t pos,
        const BoolArray& switches
    ) {
        const int64_t n_internal = static_cast<int64_t>(partition_tree.shape(0));

        if (num_features != static_cast<int64_t>(index_mask.shape(0))) {
            throw std::invalid_argument("pt_shuffle_rec: num_features must equal len(index_mask)");
        }
        if (static_cast<int64_t>(switches.shape(0)) < n_internal) {
            throw std::invalid_argument("pt_shuffle_rec: switches needs one entry per partition_tree row");
        }
        if (pos < 0) {
            throw std::invalid_argument("pt_shuffle_rec: pos must be non-negative");
        }
        // `i` names a partition_tree row when non-negative and a leaf when
        // negative. An empty tree legitimately arrives as i == -1: that is the
        // single-feature case, where the root *is* the only leaf.
        if (i < -num_features || i >= n_internal) {
            throw std::invalid_argument("pt_shuffle_rec: i is out of range for partition_tree");
        }

        // Every child id must be a leaf (< num_features) or a cluster formed by
        // an earlier row of the linkage matrix. That single bound is what makes
        // the recursion safe without per-node checks: it keeps feature_index
        // inside index_mask, keeps row indices inside partition_tree, and rules
        // out cycles, so the recursion depth cannot exceed n_internal.
        auto tree = partition_tree.view();
        for (int64_t r = 0; r < n_internal; r++) {
            for (int64_t c = 0; c < 2; c++) {
                const int64_t id = static_cast<int64_t>(tree(r, c));
                if (id < 0 || id >= num_features + r) {
                    throw std::invalid_argument(
                        "pt_shuffle_rec: partition_tree names a child that is negative, out of "
                        "range, or not formed strictly before the row that uses it"
                    );
                }
            }
        }

        return detail::pt_shuffle_rec(i, indexes, index_mask, partition_tree, num_features, pos, switches);
    }

    // The three helpers below are internal to delta_minimization_order and are
    // deliberately not bound in cutils.cpp: they index `order` relative to
    // `start` (reverse_window_score_gain reads order[start - 1]), so exposing
    // them to Python would hand out unchecked out-of-bounds access. The loop in
    // delta_minimization_order starts at 1, which keeps every access in range.
    void reverse_window(
        OrderArray& order,
        const int64_t start,
        const int64_t length
    ) {
        auto o = order.view();
        int64_t tmp;

        for (int64_t i = 0; i < length / 2; i++) {
            tmp = o(start + i);
            o(start + i) = o(start + length - 1 - i);
            o(start + length - 1 - i) = tmp;
        }
    }

    int64_t mask_delta_score(
        const MasksArray& masks,
        const int64_t row1,
        const int64_t row2
    ) {
        auto m = masks.view();
        int64_t score = 0;
        for (int64_t j = 0; j < static_cast<int64_t>(masks.shape(1)); j++) {
            score += (m(row1, j) ^ m(row2, j));
        }
        return score;
    }

    int64_t reverse_window_score_gain(
        const MasksArray& masks,
        const OrderArray& order,
        const int64_t start,
        const int64_t length
    ) {
        auto o = order.view();
        int64_t forward_score = mask_delta_score(masks, o(start - 1), o(start)) + mask_delta_score(masks, o(start + length - 1), o(start + length));
        int64_t reverse_score = mask_delta_score(masks, o(start - 1), o(start + length - 1)) + mask_delta_score(masks, o(start), o(start + length));
        return forward_score - reverse_score;
    }

    OrderArray delta_minimization_order(
        const MasksArray& all_masks,
        const int64_t max_swap_size,
        const int64_t num_passes
    ) {
        size_t num_rows = all_masks.shape(0);
        // Allocate a memory region for the order array
        int64_t* order_data = new int64_t[num_rows];
        // Initialize the order array with sequential indices
        for (size_t i = 0; i < num_rows; ++i) {
            order_data[i] = static_cast<int64_t>(i);
        }
        // Delete 'order_data' when the 'owner' capsule expires
        nb::capsule owner(order_data, [](void *p) noexcept {
            delete[] static_cast<int64_t*>(p);
        });
        // Create a 1D ndarray that uses the allocated memory
        OrderArray order(order_data, {num_rows}, owner);

        for (int64_t pass = 0; pass < num_passes; ++pass) {
            for (int64_t length = 2; length < max_swap_size; ++length) {
                for (int64_t i = 1; i < static_cast<int64_t>(num_rows) - length; ++i) {
                    if (reverse_window_score_gain(all_masks, order, i, length) > 0) {
                        reverse_window(order, i, length);
                    }
                }
            }
        }

        return order;
    }

}

#endif // CLUSTERING_UTILS_H
