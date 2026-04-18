#ifndef GREY_CODE_UTILS_H
#define GREY_CODE_UTILS_H

#include <cassert>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>

namespace nb = nanobind;
using namespace nb::literals;


void compute_grey_code_row_values_2d(
    nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& row_values,
    nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& mask,
    const nb::ndarray<uint64_t, nb::shape<-1>, nb::device::cpu>& inds,
    nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& outputs,
    const nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& shapley_coeff,
    const nb::ndarray<uint64_t, nb::shape<-1>, nb::device::cpu>& extended_delta_indexes,
    const int noop_code
) {
	assert(row_values.shape(0) == mask.shape(0));
	size_t set_size = 0;
	size_t shapley_idx = 0;
	int M = inds.shape(0);
	auto rv = row_values.view();
	int delta_ind;
	double on_coeff;
	double off_coeff = shapley_coeff(0);
	double multiplication_factor;
	for (size_t i=0; i<pow(2, M); i++) {
                assert(i < extended_delta_indexes.shape(0));
		assert(i < outputs.shape(0));

		delta_ind = extended_delta_indexes(i);
		if (delta_ind != noop_code) {
			assert((delta_ind < mask.shape(0)) && (delta_ind >= 0));
			mask(delta_ind) = !mask(delta_ind);
			if (mask(delta_ind)) {
				set_size += 1;
			}
			else {
				set_size -= 1;
			}
		}
		if (set_size == 0) {
			shapley_idx = shapley_coeff.shape(0) - 1;
		}
		else {
			shapley_idx = set_size - 1;
		}
	        assert((shapley_idx < shapley_coeff.shape(0)) && (shapley_idx >= 0));
		on_coeff = shapley_coeff(shapley_idx);
		if (set_size < (size_t)M) {
			off_coeff = shapley_coeff((shapley_idx + 1) % shapley_coeff.shape(0));
		}

		for (size_t ii = 0; ii < inds.shape(0); ii++) {
			assert (inds(ii) < mask.shape(0));
			assert (inds(ii) < rv.shape(0));
			if (mask(inds(ii))) {
				multiplication_factor = on_coeff;
			}
			else {
				multiplication_factor = -off_coeff;
			}
			assert (i < outputs.shape(0));
			for (size_t rvj = 0; rvj < rv.shape(1); rvj++) {
				assert(rvj < outputs.shape(1));
				rv(inds(ii), rvj) += multiplication_factor * outputs(i, rvj);
			}
		}
        }
}

void compute_grey_code_row_values_1d(
    nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& row_values,
    nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& mask,
    const nb::ndarray<uint64_t, nb::shape<-1>, nb::device::cpu>& inds,
    nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& outputs,
    const nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& shapley_coeff,
    const nb::ndarray<uint64_t, nb::shape<-1>, nb::device::cpu>& extended_delta_indexes,
    const int noop_code
) {
	assert(row_values.shape(0) == mask.shape(0));
	// assert(row_values.shape(0) == mask.shape(0));
	size_t set_size = 0;
	size_t shapley_idx = 0;
	int M = inds.shape(0);
	auto rv = row_values.view();
	int delta_ind;
	double on_coeff;
	double off_coeff = shapley_coeff(0);
	double multiplication_factor;
	for (size_t i=0; i<pow(2, M); i++) {
                assert(i < extended_delta_indexes.shape(0));
		assert(i < outputs.shape(0));

		delta_ind = extended_delta_indexes(i);
		if (delta_ind != noop_code) {
			assert((delta_ind < mask.shape(0)) && (delta_ind >= 0));
			mask(delta_ind) = !mask(delta_ind);
			if (mask(delta_ind)) {
				set_size += 1;
			}
			else {
				set_size -= 1;
			}
		}
		if (set_size == 0) {
			shapley_idx = shapley_coeff.shape(0) - 1;
		}
		else {
			shapley_idx = set_size - 1;
		}
	        assert((shapley_idx < shapley_coeff.shape(0)) && (shapley_idx >= 0));
		on_coeff = shapley_coeff(shapley_idx);
		if (set_size < (size_t)M) {
			off_coeff = shapley_coeff((shapley_idx + 1) % shapley_coeff.shape(0));
		}
		for (size_t ii = 0; ii < inds.shape(0); ii++) {
			assert (inds(ii) < mask.shape(0));
			if (mask(inds(ii))) {
				multiplication_factor = on_coeff;
			}
			else {
				multiplication_factor = -off_coeff;
			}
			assert (i < outputs.shape(0));
		        rv(inds(ii)) += multiplication_factor * outputs(i);
		}
        }
}

#endif // GREY_CODE_UTILS_H
