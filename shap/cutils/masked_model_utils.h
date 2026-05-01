#ifndef MASKED_MODEL_UTILS_H
#define MASKED_MODEL_UTILS_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace detail {

// ──────────────────────────────────────────────────────────────────────
// init_masks / rec_fill_masks
// ──────────────────────────────────────────────────────────────────────

inline void init_masks(
    nb::object cluster_matrix_obj,
    int M,
    nb::object indices_row_pos_obj,
    nb::object indptr_obj
) {
    auto cluster_matrix = nb::cast<nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>>(cluster_matrix_obj);
    auto indices_row_pos = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(indices_row_pos_obj);
    auto indptr = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(indptr_obj);

    auto cm = cluster_matrix.view();
    auto irp = indices_row_pos.view();
    auto ip = indptr.view();

    int64_t pos = 0;
    for (int i = 0; i < 2 * M - 1; i++) {
        if (i < M) {
            pos += 1;
        } else {
            pos += static_cast<int64_t>(cm(i - M, 3));
        }
        ip(i + 1) = pos;
        irp(i) = ip(i);
    }
}

inline void rec_fill_masks(
    nb::object cluster_matrix_obj,
    nb::object indices_row_pos_obj,
    nb::object indptr_obj,
    nb::object indices_obj,
    int M,
    int ind
) {
    auto cluster_matrix = nb::cast<nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>>(cluster_matrix_obj);
    auto indices_row_pos = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(indices_row_pos_obj);
    auto indptr = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(indptr_obj);
    auto indices = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(indices_obj);

    auto cm = cluster_matrix.view();
    auto irp = indices_row_pos.view();
    auto idx = indices.view();

    int64_t pos = irp(ind);

    if (ind < M) {
        idx(pos) = ind;
        return;
    }

    int lind = static_cast<int>(cm(ind - M, 0));
    int rind = static_cast<int>(cm(ind - M, 1));
    int lind_size = lind >= M ? static_cast<int>(cm(lind - M, 3)) : 1;
    int rind_size = rind >= M ? static_cast<int>(cm(rind - M, 3)) : 1;

    int64_t lpos = irp(lind);
    int64_t rpos = irp(rind);

    rec_fill_masks(cluster_matrix_obj, indices_row_pos_obj, indptr_obj, indices_obj, M, lind);
    for (int i = 0; i < lind_size; i++) {
        idx(pos + i) = idx(lpos + i);
    }

    rec_fill_masks(cluster_matrix_obj, indices_row_pos_obj, indptr_obj, indices_obj, M, rind);
    for (int i = 0; i < rind_size; i++) {
        idx(pos + lind_size + i) = idx(rpos + i);
    }
}

// ──────────────────────────────────────────────────────────────────────
// build_fixed_single_output
// ──────────────────────────────────────────────────────────────────────

inline void build_fixed_single_output(
    nb::object averaged_outs_obj,
    nb::object last_outs_obj,
    nb::object outputs_obj,
    nb::object batch_positions_obj,
    nb::object varying_rows_obj,
    nb::object num_varying_rows_obj,
    nb::object link_fn_obj,
    nb::object linearizing_weights_obj
) {
    auto ao_base = nb::cast<nb::ndarray<nb::any_contig, nb::device::cpu>>(averaged_outs_obj);
    nb::callable link_fn = nb::cast<nb::callable>(link_fn_obj);
    bool has_weights = !linearizing_weights_obj.is_none();

    if (ao_base.dtype() == nb::dtype<float>()) {
        auto ao = nb::cast<nb::ndarray<float, nb::shape<-1>, nb::device::cpu>>(averaged_outs_obj).view();
        auto lo = nb::cast<nb::ndarray<float, nb::shape<-1>, nb::device::cpu>>(last_outs_obj).view();
        auto out = nb::cast<nb::ndarray<float, nb::shape<-1>, nb::device::cpu>>(outputs_obj).view();
        auto bp = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(batch_positions_obj).view();
        auto vr = nb::cast<nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cpu>>(varying_rows_obj).view();
        auto nvr = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(num_varying_rows_obj).view();

        int64_t sample_count = lo.shape(0);

        // Explicitly zero-fill lo and ao so no stale/uninitialised memory
        // can leak through partial-varying-rows updates.
        for (int64_t j = 0; j < sample_count; j++) { lo(j) = 0.0f; }
        for (size_t j = 0; j < ao.shape(0); j++) { ao(j) = 0.0f; }

        // Seed last_outs with the first outputs batch (the baseline mask)
        // so that partial-varying-rows updates on later iterations read
        // real baseline values instead of uninitialised memory.
        if (bp(0) < bp(1)) {
            for (int64_t j = 0; j < sample_count; j++) { lo(j) = out(bp(0) + j); }
        }

        for (size_t i = 0; i < ao.shape(0); i++) {
            if (bp(i) < bp(i + 1)) {
                if (nvr(i) == sample_count) {
                    for (int64_t j = 0; j < sample_count; j++) { lo(j) = out(bp(i) + j); }
                } else {
                    int64_t out_idx = bp(i);
                    for (int64_t j = 0; j < sample_count; j++) {
                        if (vr(i, j)) { lo(j) = out(out_idx++); }
                    }
                }
                if (has_weights) {
                    auto lw = nb::cast<nb::ndarray<float, nb::shape<-1>, nb::device::cpu>>(linearizing_weights_obj).view();
                    nb::object linked_obj = link_fn(last_outs_obj);
                    auto linked = nb::cast<nb::ndarray<float, nb::shape<-1>, nb::device::cpu>>(linked_obj).view();
                    double sum = 0.0;
                    for (int64_t j = 0; j < sample_count; j++) { sum += (double)lw(j) * (double)linked(j); }
                    ao(i) = (float)(sum / sample_count);
                } else {
                    double sum = 0.0;
                    for (int64_t j = 0; j < sample_count; j++) { sum += (double)lo(j); }
                    nb::object result = link_fn(nb::float_(sum / sample_count));
                    ao(i) = nb::cast<float>(result);
                }
            } else {
                ao(i) = (i > 0) ? ao(i - 1) : 0.0f;
            }
        }
    } else {
        auto ao = nb::cast<nb::ndarray<double, nb::shape<-1>, nb::device::cpu>>(averaged_outs_obj).view();
        auto lo = nb::cast<nb::ndarray<double, nb::shape<-1>, nb::device::cpu>>(last_outs_obj).view();
        auto out = nb::cast<nb::ndarray<double, nb::shape<-1>, nb::device::cpu>>(outputs_obj).view();
        auto bp = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(batch_positions_obj).view();
        auto vr = nb::cast<nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cpu>>(varying_rows_obj).view();
        auto nvr = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(num_varying_rows_obj).view();

        int64_t sample_count = lo.shape(0);

        // Explicitly zero-fill lo and ao so no stale/uninitialised memory
        // can leak through partial-varying-rows updates.
        for (int64_t j = 0; j < sample_count; j++) { lo(j) = 0.0; }
        for (size_t j = 0; j < ao.shape(0); j++) { ao(j) = 0.0; }

        // Seed last_outs with the first outputs batch (the baseline mask)
        if (bp(0) < bp(1)) {
            for (int64_t j = 0; j < sample_count; j++) { lo(j) = out(bp(0) + j); }
        }

        for (size_t i = 0; i < ao.shape(0); i++) {
            if (bp(i) < bp(i + 1)) {
                if (nvr(i) == sample_count) {
                    for (int64_t j = 0; j < sample_count; j++) { lo(j) = out(bp(i) + j); }
                } else {
                    int64_t out_idx = bp(i);
                    for (int64_t j = 0; j < sample_count; j++) {
                        if (vr(i, j)) { lo(j) = out(out_idx++); }
                    }
                }
                if (has_weights) {
                    auto lw = nb::cast<nb::ndarray<double, nb::shape<-1>, nb::device::cpu>>(linearizing_weights_obj).view();
                    nb::object linked_obj = link_fn(last_outs_obj);
                    auto linked = nb::cast<nb::ndarray<double, nb::shape<-1>, nb::device::cpu>>(linked_obj).view();
                    double sum = 0.0;
                    for (int64_t j = 0; j < sample_count; j++) { sum += lw(j) * linked(j); }
                    ao(i) = sum / sample_count;
                } else {
                    double sum = 0.0;
                    for (int64_t j = 0; j < sample_count; j++) { sum += lo(j); }
                    nb::object result = link_fn(nb::float_(sum / sample_count));
                    ao(i) = nb::cast<double>(result);
                }
            } else {
                ao(i) = (i > 0) ? ao(i - 1) : 0.0;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// build_fixed_multi_output
// ──────────────────────────────────────────────────────────────────────

inline void build_fixed_multi_output(
    nb::object averaged_outs_obj,
    nb::object last_outs_obj,
    nb::object outputs_obj,
    nb::object batch_positions_obj,
    nb::object varying_rows_obj,
    nb::object num_varying_rows_obj,
    nb::object link_fn_obj,
    nb::object linearizing_weights_obj
) {
    auto ao_base = nb::cast<nb::ndarray<nb::any_contig, nb::device::cpu>>(averaged_outs_obj);
    nb::callable link_fn = nb::cast<nb::callable>(link_fn_obj);
    bool has_weights = !linearizing_weights_obj.is_none();

    if (ao_base.dtype() == nb::dtype<float>()) {
        auto ao = nb::cast<nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu>>(averaged_outs_obj).view();
        auto lo = nb::cast<nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu>>(last_outs_obj).view();
        auto out = nb::cast<nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu>>(outputs_obj).view();
        auto bp = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(batch_positions_obj).view();
        auto vr = nb::cast<nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cpu>>(varying_rows_obj).view();
        auto nvr = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(num_varying_rows_obj).view();

        int64_t sample_count = lo.shape(0);
        int64_t num_outputs = lo.shape(1);

        // Explicitly zero-fill lo and ao so no stale/uninitialised memory
        // can leak through partial-varying-rows updates.
        for (int64_t j = 0; j < sample_count; j++) {
            for (int64_t k = 0; k < num_outputs; k++) { lo(j, k) = 0.0f; }
        }
        for (size_t j = 0; j < ao.shape(0); j++) {
            for (int64_t k = 0; k < num_outputs; k++) { ao(j, k) = 0.0f; }
        }

        // Seed last_outs with the first outputs batch (the baseline mask)
        if (bp(0) < bp(1)) {
            for (int64_t j = 0; j < sample_count; j++) {
                for (int64_t k = 0; k < num_outputs; k++) { lo(j, k) = out(bp(0) + j, k); }
            }
        }

        for (size_t i = 0; i < ao.shape(0); i++) {
            if (bp(i) < bp(i + 1)) {
                if (nvr(i) == sample_count) {
                    for (int64_t j = 0; j < sample_count; j++) {
                        for (int64_t k = 0; k < num_outputs; k++) { lo(j, k) = out(bp(i) + j, k); }
                    }
                } else {
                    int64_t out_idx = bp(i);
                    for (int64_t j = 0; j < sample_count; j++) {
                        if (vr(i, j)) {
                            for (int64_t k = 0; k < num_outputs; k++) { lo(j, k) = out(out_idx, k); }
                            out_idx++;
                        }
                    }
                }
                if (has_weights) {
                    auto lw = nb::cast<nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu>>(linearizing_weights_obj).view();
                    nb::object linked_obj = link_fn(last_outs_obj);
                    auto linked = nb::cast<nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu>>(linked_obj).view();
                    for (int64_t k = 0; k < num_outputs; k++) {
                        double sum = 0.0;
                        for (int64_t j = 0; j < sample_count; j++) { sum += (double)lw(j, k) * (double)linked(j, k); }
                        ao(i, k) = (float)(sum / sample_count);
                    }
                } else {
                    for (int64_t k = 0; k < num_outputs; k++) {
                        double sum = 0.0;
                        for (int64_t j = 0; j < sample_count; j++) { sum += (double)lo(j, k); }
                        nb::object result = link_fn(nb::float_(sum / sample_count));
                        ao(i, k) = nb::cast<float>(result);
                    }
                }
            } else {
                for (int64_t k = 0; k < num_outputs; k++) { ao(i, k) = (i > 0) ? ao(i - 1, k) : 0.0f; }
            }
        }
    } else {
        auto ao = nb::cast<nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>>(averaged_outs_obj).view();
        auto lo = nb::cast<nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>>(last_outs_obj).view();
        auto out = nb::cast<nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>>(outputs_obj).view();
        auto bp = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(batch_positions_obj).view();
        auto vr = nb::cast<nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cpu>>(varying_rows_obj).view();
        auto nvr = nb::cast<nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>>(num_varying_rows_obj).view();

        int64_t sample_count = lo.shape(0);
        int64_t num_outputs = lo.shape(1);

        // Explicitly zero-fill lo and ao so no stale/uninitialised memory
        // can leak through partial-varying-rows updates.
        for (int64_t j = 0; j < sample_count; j++) {
            for (int64_t k = 0; k < num_outputs; k++) { lo(j, k) = 0.0; }
        }
        for (size_t j = 0; j < ao.shape(0); j++) {
            for (int64_t k = 0; k < num_outputs; k++) { ao(j, k) = 0.0; }
        }

        // Seed last_outs with the first outputs batch (the baseline mask)
        if (bp(0) < bp(1)) {
            for (int64_t j = 0; j < sample_count; j++) {
                for (int64_t k = 0; k < num_outputs; k++) { lo(j, k) = out(bp(0) + j, k); }
            }
        }

        for (size_t i = 0; i < ao.shape(0); i++) {
            if (bp(i) < bp(i + 1)) {
                if (nvr(i) == sample_count) {
                    for (int64_t j = 0; j < sample_count; j++) {
                        for (int64_t k = 0; k < num_outputs; k++) { lo(j, k) = out(bp(i) + j, k); }
                    }
                } else {
                    int64_t out_idx = bp(i);
                    for (int64_t j = 0; j < sample_count; j++) {
                        if (vr(i, j)) {
                            for (int64_t k = 0; k < num_outputs; k++) { lo(j, k) = out(out_idx, k); }
                            out_idx++;
                        }
                    }
                }
                if (has_weights) {
                    auto lw = nb::cast<nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>>(linearizing_weights_obj).view();
                    nb::object linked_obj = link_fn(last_outs_obj);
                    auto linked = nb::cast<nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>>(linked_obj).view();
                    for (int64_t k = 0; k < num_outputs; k++) {
                        double sum = 0.0;
                        for (int64_t j = 0; j < sample_count; j++) { sum += lw(j, k) * linked(j, k); }
                        ao(i, k) = sum / sample_count;
                    }
                } else {
                    for (int64_t k = 0; k < num_outputs; k++) {
                        double sum = 0.0;
                        for (int64_t j = 0; j < sample_count; j++) { sum += lo(j, k); }
                        nb::object result = link_fn(nb::float_(sum / sample_count));
                        ao(i, k) = nb::cast<double>(result);
                    }
                }
            } else {
                for (int64_t k = 0; k < num_outputs; k++) { ao(i, k) = (i > 0) ? ao(i - 1, k) : 0.0; }
            }
        }
    }
}

} // namespace detail

#endif // MASKED_MODEL_UTILS_H
