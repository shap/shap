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

    // literal mapping of the numba _rec_fill_masks; the caller (make_masks)
    // validates the clustering, because this recursion indexes unchecked
    void rec_fill_masks(
        const nb::ndarray<double, nb::shape<-1, 4>, nb::device::cpu>& cluster_matrix,
        const nb::ndarray<int64_t, nb::ndim<1>, nb::device::cpu>& indices_row_pos,
        nb::ndarray<int64_t, nb::ndim<1>, nb::device::cpu>& indices,
        const int64_t M,
        const int64_t ind
    ) {

        auto irp = indices_row_pos.view();
        auto idx = indices.view();

        const int64_t pos = irp(ind);

        if (ind < M) {
            idx(pos) = ind;
            return;
        }

        auto c = cluster_matrix.view();

        const int64_t lind = static_cast<int64_t>(c(ind - M, 0));
        const int64_t rind = static_cast<int64_t>(c(ind - M, 1));
        const int64_t lind_size = lind >= M ? static_cast<int64_t>(c(lind - M, 3)) : 1;
        const int64_t rind_size = rind >= M ? static_cast<int64_t>(c(rind - M, 3)) : 1;

        const int64_t lpos = irp(lind);
        const int64_t rpos = irp(rind);

        rec_fill_masks(cluster_matrix, indices_row_pos, indices, M, lind);
        for (int64_t i = 0; i < lind_size; i++) {
            idx(pos + i) = idx(lpos + i);
        }

        rec_fill_masks(cluster_matrix, indices_row_pos, indices, M, rind);
        for (int64_t i = 0; i < rind_size; i++) {
            idx(pos + lind_size + i) = idx(rpos + i);
        }
    }
}

#endif // MAKE_MASKS_UTILS_H
