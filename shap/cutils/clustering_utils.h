#ifndef CLUSTERING_UTILS_H
#define CLUSTERING_UTILS_H

#include <cassert>
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

#endif // CLUSTERING_UTILS_H
