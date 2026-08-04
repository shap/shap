#ifndef CLUSTERING_UTILS_H
#define CLUSTERING_UTILS_H

#include <cassert>
#include <algorithm>
#include <random>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace detail {

inline int row_delta(
    const nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cpu>& masks,
    int64_t row_a,
    int64_t row_b
) {
    auto m = masks.view();
    int score = 0;
    for (size_t k = 0; k < masks.shape(1); k++) {
        score += m(row_a, k) ^ m(row_b, k);
    }
    return score;
}

int pt_shuffle_rec(
    int i,
    nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& indexes,
    const nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& index_mask,
    const nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& partition_tree,
    int M,
    int pos,
    std::mt19937& rng
) {
    if (i < 0) {
        if (index_mask(i + M)) {
            indexes(pos) = i + M;
            return pos + 1;
        }
        return pos;
    }
    int left  = static_cast<int>(partition_tree(i, 0)) - M;
    int right = static_cast<int>(partition_tree(i, 1)) - M;
    if (std::bernoulli_distribution(0.5)(rng)) {
        pos = pt_shuffle_rec(left,  indexes, index_mask, partition_tree, M, pos, rng);
        pos = pt_shuffle_rec(right, indexes, index_mask, partition_tree, M, pos, rng);
    } else {
        pos = pt_shuffle_rec(right, indexes, index_mask, partition_tree, M, pos, rng);
        pos = pt_shuffle_rec(left,  indexes, index_mask, partition_tree, M, pos, rng);
    }
    return pos;
}

} // namespace detail


void reverse_window(
    nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& order,
    const int start,
    const int length
) {
    assert(length >= 0);
    assert(start >= 0);
    assert((size_t)start + (size_t)length <= order.shape(0));
    for (size_t i = 0; i < (size_t)length / 2; i++) {
        int64_t tmp = order(start + i);
        order(start + i) = order(start + length - i - 1);
        order(start + length - i - 1) = tmp;
    }
}

int mask_delta_score(
    const nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& m1,
    const nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& m2
) {
    assert(m1.shape(0) == m2.shape(0));
    int score = 0;
    for (size_t i = 0; i < m1.shape(0); i++) {
        score += m1(i) ^ m2(i);
    }
    return score;
}

int reverse_window_score_gain(
    const nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cpu>& masks,
    const nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& order,
    const int start,
    const int length
) {
    assert(start >= 1);
    assert(start + length < (int)order.shape(0));
    int forward_score = detail::row_delta(masks, order(start - 1), order(start))
                      + detail::row_delta(masks, order(start + length - 1), order(start + length));
    int reverse_score = detail::row_delta(masks, order(start - 1), order(start + length - 1))
                      + detail::row_delta(masks, order(start), order(start + length));
    return forward_score - reverse_score;
}

nb::ndarray<nb::numpy, int64_t, nb::shape<-1>> delta_minimization_order(
    const nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cpu>& all_masks,
    const int max_swap_size = 100,
    const int num_passes = 2
) {
    size_t n = all_masks.shape(0);

    int64_t* data = new int64_t[n];
    for (size_t i = 0; i < n; i++) data[i] = static_cast<int64_t>(i);

    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<int64_t*>(p);
    });

    int effective_max = std::min(max_swap_size, static_cast<int>(n));

    for (int pass = 0; pass < num_passes; pass++) {
        for (int length = 2; length < effective_max; length++) {
            for (int i = 1; i < static_cast<int>(n) - length; i++) {
                int forward = detail::row_delta(all_masks, data[i - 1], data[i])
                            + detail::row_delta(all_masks, data[i + length - 1], data[i + length]);
                int reverse = detail::row_delta(all_masks, data[i - 1], data[i + length - 1])
                            + detail::row_delta(all_masks, data[i], data[i + length]);
                if (forward > reverse) {
                    for (int j = 0; j < length / 2; j++) {
                        std::swap(data[i + j], data[i + length - j - 1]);
                    }
                }
            }
        }
    }

    size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, int64_t, nb::shape<-1>>(data, 1, shape, owner);
}

int pt_shuffle_rec(
    int i,
    nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& indexes,
    const nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& index_mask,
    const nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& partition_tree,
    int M,
    int pos
) {
    thread_local std::mt19937 rng{std::random_device{}()};
    return detail::pt_shuffle_rec(i, indexes, index_mask, partition_tree, M, pos, rng);
}

#endif // CLUSTERING_UTILS_H
