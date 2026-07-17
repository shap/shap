#ifndef CLUSTERING_UTILS_H
#define CLUSTERING_UTILS_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace clustering {
    void reverse_window(
        nb::ndarray<int64_t, nb::ndim<1>>& order,
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
        const nb::ndarray<int64_t, nb::ndim<1>>& order,
        const int64_t start,
        const int64_t length
    ) {
        auto o = order.view();
        int64_t forward_score = mask_delta_score(masks, o(start - 1), o(start)) + mask_delta_score(masks, o(start + length - 1), o(start + length));
        int64_t reverse_score = mask_delta_score(masks, o(start - 1), o(start + length - 1)) + mask_delta_score(masks, o(start), o(start + length));
        return forward_score - reverse_score;
    }

}

#endif // CLUSTERING_UTILS_H
