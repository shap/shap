#ifndef MAKE_MASKS_UTILS_H
#define MAKE_MASKS_UTILS_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace masks {
    void init_masks(
        const nb::ndarray<double, nb::ndim<2>>& cluster_matrix,
        const int M,
        nb::ndarray<int, nb::ndim<1>>& indices_row_pos,
        nb::ndarray<int, nb::ndim<1>>& indptr
    ) {
        int pos = 0;
        auto c = cluster_matrix.view();
        auto irp = indices_row_pos.view();
        auto ip = indptr.view();

        for (int i = 0; i < 2 * M - 1; i++) {
            if (i < M) {
                pos++;
            } else {
                pos += static_cast<int>(c(i - M, 3));
            }
            ip(i+1) = pos;
            irp(i) = ip(i);
        }
    }
}

#endif // MAKE_MASKS_UTILS_H
