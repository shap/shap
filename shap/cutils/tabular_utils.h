#ifndef TABULAR_UTILS_H
#define TABULAR_UTILS_H

#include <cstdint>

#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace tabular {

template <typename T>
void single_delta_mask(
    const int64_t dind,
    nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu>& masked_inputs,
    nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& last_mask,
    const nb::ndarray<const T, nb::shape<-1, -1>, nb::device::cpu>& data,
    const nb::ndarray<const T, nb::shape<-1>, nb::device::cpu>& x,
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
    const nb::ndarray<const int64_t, nb::shape<-1>, nb::device::cpu>& masks,
    const nb::ndarray<const T, nb::shape<-1>, nb::device::cpu>& x,
    nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& curr_delta_inds,
    nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cpu>& varying_rows_out,
    nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu>& masked_inputs_tmp,
    nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& last_mask,
    const nb::ndarray<const T, nb::shape<-1, -1>, nb::device::cpu>& data,
    const nb::ndarray<const bool, nb::shape<-1, -1>, nb::device::cpu>& variants,
    nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& masked_inputs_out,
    const int64_t noop_code
) {
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

        for (size_t row = 0; row < num_rows; ++row) {
            for (size_t column = 0; column < masked_inputs_tmp.shape(1); ++column) {
                masked_inputs_out_view(output_pos + row, column) = masked_inputs_tmp_view(row, column);
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
