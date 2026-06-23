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

}

#endif // CLUSTERING_UTILS_H
