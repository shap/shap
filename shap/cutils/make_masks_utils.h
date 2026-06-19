#ifndef MAKE_MASKS_UTILS_H
#define MAKE_MASKS_UTILS_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace masks {
    void init_masks(
        const nb::ndarray<double, nb::ndim<2>>& cluster_matrix,
        const int64_t M,
        nb::ndarray<int64_t, nb::ndim<1>>& indices_row_pos,
        nb::ndarray<int64_t, nb::ndim<1>>& indptr
    ) {
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
}

#endif // MAKE_MASKS_UTILS_H
