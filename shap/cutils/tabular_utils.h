#ifndef TABULAR_UTILS_H
#define TABULAR_UTILS_H

#include <cstdint>
#include <stdexcept>
#include <string>

#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace tabular {

// Argument types follow the conventions already used by grey_code_utils.h and
// kernel_explainer_utils.h: explicit extents plus nb::device::cpu. The numba
// implementation this replaced was host-only, so requiring CPU memory loses
// nothing; without the tag a CUDA tensor is accepted and its device pointer is
// dereferenced on the host.
template <typename T> using Vec = nb::ndarray<T, nb::shape<-1>, nb::device::cpu>;
template <typename T> using Mat = nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu>;

inline void require(bool ok, const std::string& message) {
    if (!ok) {
        throw std::invalid_argument("delta_masking: " + message);
    }
}

inline std::string shape_2d(size_t rows, size_t cols) {
    return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
}

/// Validate every index the masking loop is going to use, once, before any
/// writing starts.
///
/// The numpy implementation got these checks for free: fancy indexing raises
/// IndexError on an out-of-range feature, and
/// ``masked_inputs_out[output_pos:output_pos + N] = masked_inputs_tmp`` raises
/// ValueError unless the destination slice and the source have matching shapes.
/// Hand-rolling those as scalar loops threw the checks away, so they are
/// restored here.
///
/// This runs once per call, and is O(len(masks)) against a main loop that is
/// O(len(masks) * num_rows * num_features), so it is free in practice -- which
/// is why it lives here rather than in Python: it also covers callers that
/// reach `shap._cutils` directly, and `Tabular.__call__` is a hot path where a
/// per-call Python-side pass over the mask stream would not be.
template <typename T>
void validate_delta_masking(
    const Vec<const int64_t>& masks,
    const Vec<const T>& x,
    const Vec<int64_t>& curr_delta_inds,
    const Mat<bool>& varying_rows_out,
    const Mat<T>& masked_inputs_tmp,
    const Vec<bool>& last_mask,
    const Mat<const T>& data,
    const Mat<const bool>& variants,
    const Mat<double>& masked_inputs_out,
    const int64_t noop_code
) {
    const size_t num_rows = data.shape(0);
    const size_t num_features = data.shape(1);

    require(x.shape(0) == num_features,
            "x has " + std::to_string(x.shape(0)) + " entries but data has "
            + std::to_string(num_features) + " columns");
    require(masked_inputs_tmp.shape(0) == num_rows && masked_inputs_tmp.shape(1) == num_features,
            "masked_inputs_tmp has shape " + shape_2d(masked_inputs_tmp.shape(0), masked_inputs_tmp.shape(1))
            + " but data has shape " + shape_2d(num_rows, num_features));
    require(last_mask.shape(0) == num_features,
            "last_mask has " + std::to_string(last_mask.shape(0)) + " entries but data has "
            + std::to_string(num_features) + " columns");
    require(variants.shape(0) == num_rows && variants.shape(1) == num_features,
            "variants has shape " + shape_2d(variants.shape(0), variants.shape(1))
            + " but data has shape " + shape_2d(num_rows, num_features));
    require(masked_inputs_out.shape(1) == num_features,
            "masked_inputs_out has " + std::to_string(masked_inputs_out.shape(1))
            + " columns but data has " + std::to_string(num_features));

    // Walk the mask stream exactly as the main loop will, so every read it
    // performs is proven in range before anything is written.
    auto masks_view = masks.view();
    const size_t num_mask_entries = masks.shape(0);
    size_t num_masks = 0;
    size_t pos = 0;
    while (pos < num_mask_entries) {
        size_t dpos = 0;
        while (true) {
            require(pos + dpos < num_mask_entries,
                    "masks ends part-way through a delta run; the final entry must be non-negative");
            const int64_t value = masks_view(pos + dpos);
            const int64_t dind = value >= 0 ? value : -value - 1;
            require(dind == noop_code || (dind >= 0 && static_cast<size_t>(dind) < num_features),
                    "masks[" + std::to_string(pos + dpos) + "] selects feature " + std::to_string(dind)
                    + ", out of range for data with " + std::to_string(num_features) + " columns");
            if (value >= 0) {
                break;
            }
            ++dpos;
        }
        require(dpos < curr_delta_inds.shape(0),
                "curr_delta_inds has " + std::to_string(curr_delta_inds.shape(0))
                + " entries but a delta run needs " + std::to_string(dpos + 1));
        pos += dpos + 1;
        ++num_masks;
    }

    require(varying_rows_out.shape(0) >= num_masks && varying_rows_out.shape(1) == num_rows,
            "varying_rows_out has shape " + shape_2d(varying_rows_out.shape(0), varying_rows_out.shape(1))
            + " but " + std::to_string(num_masks) + " mask(s) over " + std::to_string(num_rows)
            + " background rows need " + shape_2d(num_masks, num_rows));
    require(masked_inputs_out.shape(0) >= num_masks * num_rows,
            "masked_inputs_out has " + std::to_string(masked_inputs_out.shape(0)) + " rows but "
            + std::to_string(num_masks) + " mask(s) over " + std::to_string(num_rows)
            + " background rows need " + std::to_string(num_masks * num_rows));
}

/// Flip one feature, mirroring the numba ``_single_delta_mask``.
///
/// ``dind`` is not range-checked here: validate_delta_masking has already
/// proven every decoded mask entry is either ``noop_code`` or a valid column.
template <typename T>
void single_delta_mask(
    const int64_t dind,
    Mat<T>& masked_inputs,
    Vec<bool>& last_mask,
    const Mat<const T>& data,
    const Vec<const T>& x,
    const int64_t noop_code
) {
    if (dind == noop_code) {
        return;
    }

    auto masked_inputs_view = masked_inputs.view();
    auto last_mask_view = last_mask.view();
    auto data_view = data.view();
    auto x_view = x.view();

    if (last_mask_view(dind)) {
        for (size_t row = 0; row < masked_inputs.shape(0); ++row) {
            masked_inputs_view(row, dind) = data_view(row, dind);
        }
        last_mask_view(dind) = false;
    } else {
        for (size_t row = 0; row < masked_inputs.shape(0); ++row) {
            masked_inputs_view(row, dind) = x_view(dind);
        }
        last_mask_view(dind) = true;
    }
}

template <typename T>
void delta_masking(
    const Vec<const int64_t>& masks,
    const Vec<const T>& x,
    Vec<int64_t>& curr_delta_inds,
    Mat<bool>& varying_rows_out,
    Mat<T>& masked_inputs_tmp,
    Vec<bool>& last_mask,
    const Mat<const T>& data,
    const Mat<const bool>& variants,
    Mat<double>& masked_inputs_out,
    const int64_t noop_code
) {
    validate_delta_masking<T>(
        masks, x, curr_delta_inds, varying_rows_out, masked_inputs_tmp,
        last_mask, data, variants, masked_inputs_out, noop_code
    );

    auto masks_view = masks.view();
    auto curr_delta_inds_view = curr_delta_inds.view();
    auto varying_rows_out_view = varying_rows_out.view();
    auto masked_inputs_tmp_view = masked_inputs_tmp.view();
    auto variants_view = variants.view();
    auto masked_inputs_out_view = masked_inputs_out.view();

    size_t masks_pos = 0;
    size_t output_pos = 0;
    size_t mask_index = 0;
    const size_t num_rows = masked_inputs_tmp.shape(0);
    const size_t num_features = masked_inputs_tmp.shape(1);

    while (masks_pos < masks.shape(0)) {
        size_t dpos = 0;
        curr_delta_inds_view(0) = masks_view(masks_pos);
        while (curr_delta_inds_view(dpos) < 0) {
            curr_delta_inds_view(dpos) = -curr_delta_inds_view(dpos) - 1;
            single_delta_mask(
                curr_delta_inds_view(dpos), masked_inputs_tmp, last_mask, data, x, noop_code
            );
            ++dpos;
            curr_delta_inds_view(dpos) = masks_view(masks_pos + dpos);
        }
        single_delta_mask(curr_delta_inds_view(dpos), masked_inputs_tmp, last_mask, data, x, noop_code);

        // masked_inputs_out[output_pos : output_pos + N] = masked_inputs_tmp
        for (size_t out_row = output_pos; out_row < output_pos + num_rows; ++out_row) {
            for (size_t column = 0; column < num_features; ++column) {
                masked_inputs_out_view(out_row, column) = masked_inputs_tmp_view(out_row - output_pos, column);
            }
        }
        masks_pos += dpos + 1;

        if (mask_index == 0) {
            for (size_t row = 0; row < num_rows; ++row) {
                varying_rows_out_view(mask_index, row) = true;
            }
        } else if (dpos == 0) {
            for (size_t row = 0; row < num_rows; ++row) {
                varying_rows_out_view(mask_index, row) = variants_view(row, curr_delta_inds_view(0));
            }
        } else {
            for (size_t row = 0; row < num_rows; ++row) {
                bool varies = false;
                for (size_t delta_pos = 0; delta_pos <= dpos; ++delta_pos) {
                    varies |= variants_view(row, curr_delta_inds_view(delta_pos));
                }
                varying_rows_out_view(mask_index, row) = varies;
            }
        }

        output_pos += num_rows;
        ++mask_index;
    }
}

}  // namespace tabular

#endif  // TABULAR_UTILS_H
