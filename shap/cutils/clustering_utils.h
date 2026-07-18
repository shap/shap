#ifndef CLUSTERING_UTILS_H
#define CLUSTERING_UTILS_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace clustering {
    int64_t pt_shuffle_rec(
        const int64_t i,
        nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>& indexes,
        const nb::ndarray<nb::numpy, bool, nb::ndim<1>>& index_mask,
        const nb::ndarray<nb::numpy, double, nb::ndim<2>>& partition_tree,
        const int64_t num_features,
        const int64_t pos,
        const nb::ndarray<nb::numpy, bool, nb::ndim<1>>& switches
    ) {
        auto inds = indexes.view();
        auto mask = index_mask.view();
        auto tree = partition_tree.view();
        auto branch_switches = switches.view();

        if (i < 0) {
            const int64_t feature_index = i + num_features;
            if (mask(feature_index)) {
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

    void reverse_window(
        nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>& order,
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
        const nb::ndarray<bool, nb::ndim<2>>& masks,
        const int64_t row1,
        const int64_t row2
    ) {
        auto m = masks.view();
        int64_t score = 0;
        for (int64_t j = 0; j < masks.shape(1); j++) {
            score += (m(row1, j) ^ m(row2, j));
        }
        return score;
    }

    int64_t reverse_window_score_gain(
        const nb::ndarray<bool, nb::ndim<2>>& masks,
        const nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>& order,
        const int64_t start,
        const int64_t length
    ) {
        auto o = order.view();
        int64_t forward_score = mask_delta_score(masks, o(start - 1), o(start)) + mask_delta_score(masks, o(start + length - 1), o(start + length));
        int64_t reverse_score = mask_delta_score(masks, o(start - 1), o(start + length - 1)) + mask_delta_score(masks, o(start), o(start + length));
        return forward_score - reverse_score;
    }

    nb::ndarray<nb::numpy, int64_t, nb::ndim<1>> delta_minimization_order(
        const nb::ndarray<bool, nb::ndim<2>>& all_masks,
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
        nb::ndarray<nb::numpy, int64_t, nb::ndim<1>> order(order_data, {num_rows}, owner);

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
