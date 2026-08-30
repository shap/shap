#ifndef MASKED_MODEL_UTILS_H
#define MASKED_MODEL_UTILS_H

#include <cstdint>

#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace masked_model {

template <typename T>
void update_single_output(
    nb::ndarray<T, nb::ndim<1>, nb::device::cpu>& last_outs,
    const nb::ndarray<const T, nb::ndim<1>, nb::device::cpu>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>, nb::device::cpu>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& num_varying_rows,
    const size_t output_index
) {
    auto last_outs_view = last_outs.view();
    auto outputs_view = outputs.view();
    auto batch_positions_view = batch_positions.view();
    auto varying_rows_view = varying_rows.view();
    auto num_varying_rows_view = num_varying_rows.view();

    const size_t sample_count = last_outs_view.shape(0);
    const size_t output_start = batch_positions_view(output_index);

    if (num_varying_rows_view(output_index) == static_cast<int64_t>(sample_count)) {
        for (size_t row = 0; row < sample_count; ++row) {
            last_outs_view(row) = outputs_view(output_start + row);
        }
    } else {
        size_t output_offset = 0;
        for (size_t row = 0; row < sample_count; ++row) {
            if (varying_rows_view(output_index, row)) {
                last_outs_view(row) = outputs_view(output_start + output_offset);
                ++output_offset;
            }
        }
    }
}

template <typename T>
void build_fixed_single_output(
    nb::ndarray<T, nb::ndim<1>, nb::device::cpu>& averaged_outs,
    nb::ndarray<T, nb::ndim<1>, nb::device::cpu>& last_outs,
    const nb::ndarray<const T, nb::ndim<1>, nb::device::cpu>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>, nb::device::cpu>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& num_varying_rows
) {
    auto averaged_outs_view = averaged_outs.view();
    auto last_outs_view = last_outs.view();
    auto batch_positions_view = batch_positions.view();

    const size_t sample_count = last_outs_view.shape(0);
    for (size_t i = 0; i < averaged_outs_view.shape(0); ++i) {
        if (batch_positions_view(i) < batch_positions_view(i + 1)) {
            update_single_output(last_outs, outputs, batch_positions, varying_rows, num_varying_rows, i);
            T total = 0;
            for (size_t row = 0; row < sample_count; ++row) {
                total += last_outs_view(row);
            }
            averaged_outs_view(i) = total / static_cast<T>(sample_count);
        } else {
            // numba: averaged_outs[i] = averaged_outs[i - 1] — at i == 0 the
            // negative index wraps around to the last element
            averaged_outs_view(i) = averaged_outs_view(i == 0 ? averaged_outs_view.shape(0) - 1 : i - 1);
        }
    }
}

template <typename T>
void build_fixed_single_output_weighted(
    nb::ndarray<T, nb::ndim<1>, nb::device::cpu>& averaged_outs,
    nb::ndarray<T, nb::ndim<1>, nb::device::cpu>& last_outs,
    const nb::ndarray<const T, nb::ndim<1>, nb::device::cpu>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>, nb::device::cpu>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& num_varying_rows,
    const nb::ndarray<const T, nb::ndim<1>, nb::device::cpu>& linearizing_weights
) {
    auto averaged_outs_view = averaged_outs.view();
    auto last_outs_view = last_outs.view();
    auto batch_positions_view = batch_positions.view();
    auto linearizing_weights_view = linearizing_weights.view();

    const size_t sample_count = last_outs_view.shape(0);
    for (size_t i = 0; i < averaged_outs_view.shape(0); ++i) {
        if (batch_positions_view(i) < batch_positions_view(i + 1)) {
            update_single_output(last_outs, outputs, batch_positions, varying_rows, num_varying_rows, i);
            T total = 0;
            for (size_t row = 0; row < sample_count; ++row) {
                total += linearizing_weights_view(row) * last_outs_view(row);
            }
            averaged_outs_view(i) = total / static_cast<T>(sample_count);
        } else {
            // numba: averaged_outs[i] = averaged_outs[i - 1] — at i == 0 the
            // negative index wraps around to the last element
            averaged_outs_view(i) = averaged_outs_view(i == 0 ? averaged_outs_view.shape(0) - 1 : i - 1);
        }
    }
}

template <typename T>
void update_multi_output(
    nb::ndarray<T, nb::ndim<2>, nb::device::cpu>& last_outs,
    const nb::ndarray<const T, nb::ndim<2>, nb::device::cpu>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>, nb::device::cpu>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& num_varying_rows,
    const size_t output_index
) {
    auto last_outs_view = last_outs.view();
    auto outputs_view = outputs.view();
    auto batch_positions_view = batch_positions.view();
    auto varying_rows_view = varying_rows.view();
    auto num_varying_rows_view = num_varying_rows.view();

    const size_t sample_count = last_outs_view.shape(0);
    const size_t output_count = last_outs_view.shape(1);
    const size_t output_start = batch_positions_view(output_index);

    if (num_varying_rows_view(output_index) == static_cast<int64_t>(sample_count)) {
        for (size_t row = 0; row < sample_count; ++row) {
            for (size_t column = 0; column < output_count; ++column) {
                last_outs_view(row, column) = outputs_view(output_start + row, column);
            }
        }
    } else {
        size_t output_offset = 0;
        for (size_t row = 0; row < sample_count; ++row) {
            if (varying_rows_view(output_index, row)) {
                for (size_t column = 0; column < output_count; ++column) {
                    last_outs_view(row, column) = outputs_view(output_start + output_offset, column);
                }
                ++output_offset;
            }
        }
    }
}

template <typename T>
void build_fixed_multi_output(
    nb::ndarray<T, nb::ndim<2>, nb::device::cpu>& averaged_outs,
    nb::ndarray<T, nb::ndim<2>, nb::device::cpu>& last_outs,
    const nb::ndarray<const T, nb::ndim<2>, nb::device::cpu>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>, nb::device::cpu>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& num_varying_rows
) {
    auto averaged_outs_view = averaged_outs.view();
    auto last_outs_view = last_outs.view();
    auto batch_positions_view = batch_positions.view();

    const size_t sample_count = last_outs_view.shape(0);
    const size_t output_count = last_outs_view.shape(1);
    for (size_t i = 0; i < averaged_outs_view.shape(0); ++i) {
        if (batch_positions_view(i) < batch_positions_view(i + 1)) {
            update_multi_output(last_outs, outputs, batch_positions, varying_rows, num_varying_rows, i);
            for (size_t column = 0; column < output_count; ++column) {
                T total = 0;
                for (size_t row = 0; row < sample_count; ++row) {
                    total += last_outs_view(row, column);
                }
                averaged_outs_view(i, column) = total / static_cast<T>(sample_count);
            }
        } else {
            // numba: averaged_outs[i] = averaged_outs[i - 1] — at i == 0 the
            // negative index wraps around to the last row
            const size_t carry = i == 0 ? averaged_outs_view.shape(0) - 1 : i - 1;
            for (size_t column = 0; column < output_count; ++column) {
                averaged_outs_view(i, column) = averaged_outs_view(carry, column);
            }
        }
    }
}

template <typename T>
void build_fixed_multi_output_weighted(
    nb::ndarray<T, nb::ndim<2>, nb::device::cpu>& averaged_outs,
    nb::ndarray<T, nb::ndim<2>, nb::device::cpu>& last_outs,
    const nb::ndarray<const T, nb::ndim<2>, nb::device::cpu>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>, nb::device::cpu>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>& num_varying_rows,
    const nb::ndarray<const T, nb::ndim<2>, nb::device::cpu>& linearizing_weights
) {
    auto averaged_outs_view = averaged_outs.view();
    auto last_outs_view = last_outs.view();
    auto batch_positions_view = batch_positions.view();
    auto linearizing_weights_view = linearizing_weights.view();

    const size_t sample_count = last_outs_view.shape(0);
    const size_t output_count = last_outs_view.shape(1);
    for (size_t i = 0; i < averaged_outs_view.shape(0); ++i) {
        if (batch_positions_view(i) < batch_positions_view(i + 1)) {
            update_multi_output(last_outs, outputs, batch_positions, varying_rows, num_varying_rows, i);
            for (size_t column = 0; column < output_count; ++column) {
                T total = 0;
                for (size_t row = 0; row < sample_count; ++row) {
                    total += linearizing_weights_view(row, column) * last_outs_view(row, column);
                }
                averaged_outs_view(i, column) = total / static_cast<T>(sample_count);
            }
        } else {
            // numba: averaged_outs[i] = averaged_outs[i - 1] — at i == 0 the
            // negative index wraps around to the last row
            const size_t carry = i == 0 ? averaged_outs_view.shape(0) - 1 : i - 1;
            for (size_t column = 0; column < output_count; ++column) {
                averaged_outs_view(i, column) = averaged_outs_view(carry, column);
            }
        }
    }
}

}  // namespace masked_model

#endif  // MASKED_MODEL_UTILS_H
