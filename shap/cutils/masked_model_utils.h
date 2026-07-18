#ifndef MASKED_MODEL_UTILS_H
#define MASKED_MODEL_UTILS_H

#include <cstdint>

#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace masked_model {

template <typename T>
void update_single_output(
    nb::ndarray<T, nb::ndim<1>>& last_outs,
    const nb::ndarray<const T, nb::ndim<1>>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>>& num_varying_rows,
    const size_t output_index
) {
    const size_t sample_count = last_outs.shape(0);
    const size_t output_start = batch_positions(output_index);

    if (num_varying_rows(output_index) == static_cast<int64_t>(sample_count)) {
        for (size_t row = 0; row < sample_count; ++row) {
            last_outs(row) = outputs(output_start + row);
        }
    } else {
        size_t output_offset = 0;
        for (size_t row = 0; row < sample_count; ++row) {
            if (varying_rows(output_index, row)) {
                last_outs(row) = outputs(output_start + output_offset);
                ++output_offset;
            }
        }
    }
}

template <typename T>
void build_fixed_single_output(
    nb::ndarray<T, nb::ndim<1>>& averaged_outs,
    nb::ndarray<T, nb::ndim<1>>& last_outs,
    const nb::ndarray<const T, nb::ndim<1>>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>>& num_varying_rows
) {
    const size_t sample_count = last_outs.shape(0);
    for (size_t i = 0; i < averaged_outs.shape(0); ++i) {
        if (batch_positions(i) < batch_positions(i + 1)) {
            update_single_output(last_outs, outputs, batch_positions, varying_rows, num_varying_rows, i);
            T total = 0;
            for (size_t row = 0; row < sample_count; ++row) {
                total += last_outs(row);
            }
            averaged_outs(i) = total / static_cast<T>(sample_count);
        } else {
            averaged_outs(i) = i == 0 ? T(0) : averaged_outs(i - 1);
        }
    }
}

template <typename T>
void build_fixed_single_output_weighted(
    nb::ndarray<T, nb::ndim<1>>& averaged_outs,
    nb::ndarray<T, nb::ndim<1>>& last_outs,
    const nb::ndarray<const T, nb::ndim<1>>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>>& num_varying_rows,
    const nb::ndarray<const T, nb::ndim<1>>& linearizing_weights
) {
    const size_t sample_count = last_outs.shape(0);
    for (size_t i = 0; i < averaged_outs.shape(0); ++i) {
        if (batch_positions(i) < batch_positions(i + 1)) {
            update_single_output(last_outs, outputs, batch_positions, varying_rows, num_varying_rows, i);
            T total = 0;
            for (size_t row = 0; row < sample_count; ++row) {
                total += linearizing_weights(row) * last_outs(row);
            }
            averaged_outs(i) = total / static_cast<T>(sample_count);
        } else {
            averaged_outs(i) = i == 0 ? T(0) : averaged_outs(i - 1);
        }
    }
}

template <typename T>
void update_multi_output(
    nb::ndarray<T, nb::ndim<2>>& last_outs,
    const nb::ndarray<const T, nb::ndim<2>>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>>& num_varying_rows,
    const size_t output_index
) {
    const size_t sample_count = last_outs.shape(0);
    const size_t output_count = last_outs.shape(1);
    const size_t output_start = batch_positions(output_index);

    if (num_varying_rows(output_index) == static_cast<int64_t>(sample_count)) {
        for (size_t row = 0; row < sample_count; ++row) {
            for (size_t column = 0; column < output_count; ++column) {
                last_outs(row, column) = outputs(output_start + row, column);
            }
        }
    } else {
        size_t output_offset = 0;
        for (size_t row = 0; row < sample_count; ++row) {
            if (varying_rows(output_index, row)) {
                for (size_t column = 0; column < output_count; ++column) {
                    last_outs(row, column) = outputs(output_start + output_offset, column);
                }
                ++output_offset;
            }
        }
    }
}

template <typename T>
void build_fixed_multi_output(
    nb::ndarray<T, nb::ndim<2>>& averaged_outs,
    nb::ndarray<T, nb::ndim<2>>& last_outs,
    const nb::ndarray<const T, nb::ndim<2>>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>>& num_varying_rows
) {
    const size_t sample_count = last_outs.shape(0);
    const size_t output_count = last_outs.shape(1);
    for (size_t i = 0; i < averaged_outs.shape(0); ++i) {
        if (batch_positions(i) < batch_positions(i + 1)) {
            update_multi_output(last_outs, outputs, batch_positions, varying_rows, num_varying_rows, i);
            for (size_t column = 0; column < output_count; ++column) {
                T total = 0;
                for (size_t row = 0; row < sample_count; ++row) {
                    total += last_outs(row, column);
                }
                averaged_outs(i, column) = total / static_cast<T>(sample_count);
            }
        } else {
            for (size_t column = 0; column < output_count; ++column) {
                averaged_outs(i, column) = i == 0 ? T(0) : averaged_outs(i - 1, column);
            }
        }
    }
}

template <typename T>
void build_fixed_multi_output_weighted(
    nb::ndarray<T, nb::ndim<2>>& averaged_outs,
    nb::ndarray<T, nb::ndim<2>>& last_outs,
    const nb::ndarray<const T, nb::ndim<2>>& outputs,
    const nb::ndarray<const int64_t, nb::ndim<1>>& batch_positions,
    const nb::ndarray<const bool, nb::ndim<2>>& varying_rows,
    const nb::ndarray<const int64_t, nb::ndim<1>>& num_varying_rows,
    const nb::ndarray<const T, nb::ndim<2>>& linearizing_weights
) {
    const size_t sample_count = last_outs.shape(0);
    const size_t output_count = last_outs.shape(1);
    for (size_t i = 0; i < averaged_outs.shape(0); ++i) {
        if (batch_positions(i) < batch_positions(i + 1)) {
            update_multi_output(last_outs, outputs, batch_positions, varying_rows, num_varying_rows, i);
            for (size_t column = 0; column < output_count; ++column) {
                T total = 0;
                for (size_t row = 0; row < sample_count; ++row) {
                    total += linearizing_weights(row, column) * last_outs(row, column);
                }
                averaged_outs(i, column) = total / static_cast<T>(sample_count);
            }
        } else {
            for (size_t column = 0; column < output_count; ++column) {
                averaged_outs(i, column) = i == 0 ? T(0) : averaged_outs(i - 1, column);
            }
        }
    }
}

}  // namespace masked_model

#endif  // MASKED_MODEL_UTILS_H
