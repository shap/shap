#ifndef GREY_CODE_UTILS_H
#define GREY_CODE_UTILS_H

#include <cassert>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// The assert()s previously used here are no-ops in release builds (NDEBUG), so
// every bound they claimed to enforce was unenforced. gc_require is a real
// check; all uses are hoisted out of the hot loops so they cost O(1) or O(2^M)
// against a main loop that is at least O(2^M * M).
inline void gc_require(bool ok, const std::string& what) {
    if (!ok) {
        throw std::invalid_argument("compute_grey_code_row_values: " + what);
    }
}

// Shared entry validation: the mask stream and the iteration count.
// `mask` must start all-false because `set_size` starts at 0 and thereafter
// tracks the number of set entries; a pre-set mask would desynchronise the two
// and make the `set_size - 1` / `set_size - 2` indices below go negative.
template <typename EDI, typename MASK>
size_t gc_validate_stream(const EDI& extended_delta_indexes, const MASK& mask,
                          size_t M, int noop_code) {
    gc_require(M < 63, "inds has " + std::to_string(M) + " entries; 2**M would overflow");
    const size_t n_iter = static_cast<size_t>(1) << M;
    gc_require(extended_delta_indexes.shape(0) >= n_iter,
               "extended_delta_indexes has " + std::to_string(extended_delta_indexes.shape(0))
               + " entries but 2**M is " + std::to_string(n_iter));
    for (size_t i = 0; i < mask.shape(0); i++) {
        gc_require(!mask(i), "mask must be all-false on entry");
    }
    for (size_t i = 0; i < n_iter; i++) {
        const int64_t d = extended_delta_indexes(i);
        gc_require(d == noop_code || (d >= 0 && static_cast<size_t>(d) < mask.shape(0)),
                   "extended_delta_indexes[" + std::to_string(i) + "] = " + std::to_string(d)
                   + " is not a valid mask position");
    }
    return n_iter;
}


void compute_grey_code_row_values_2d(
    nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& row_values,
    nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& mask,
    const nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& inds,
    nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& outputs,
    const nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& shapley_coeff,
    const nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& extended_delta_indexes,
    const int noop_code
) {
	size_t set_size = 0;
	size_t shapley_idx = 0;
	int M = inds.shape(0);
	auto rv = row_values.view();
	gc_require(row_values.shape(0) == mask.shape(0), "row_values and mask disagree on length");
	gc_require(shapley_coeff.shape(0) > 0, "shapley_coeff is empty");
	const size_t n_iter = gc_validate_stream(extended_delta_indexes, mask, (size_t)M, noop_code);
	gc_require(outputs.shape(0) >= n_iter, "outputs has fewer rows than 2**M");
	gc_require(rv.shape(1) <= outputs.shape(1), "row_values has more columns than outputs");
	for (size_t ii = 0; ii < inds.shape(0); ii++) {
		gc_require(inds(ii) >= 0 && (size_t)inds(ii) < mask.shape(0), "inds is out of range for mask");
		gc_require((size_t)inds(ii) < rv.shape(0), "inds is out of range for row_values");
	}
	int delta_ind;
	double on_coeff;
	double off_coeff = shapley_coeff(0);
	double multiplication_factor;
	for (size_t i = 0; i < n_iter; i++) {

		delta_ind = extended_delta_indexes(i);
		if (delta_ind != noop_code) {
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
		on_coeff = shapley_coeff(shapley_idx);
		if (set_size < (size_t)M) {
			off_coeff = shapley_coeff((shapley_idx + 1) % shapley_coeff.shape(0));
		}

		for (size_t ii = 0; ii < inds.shape(0); ii++) {
			if (mask(inds(ii))) {
				multiplication_factor = on_coeff;
			}
			else {
				multiplication_factor = -off_coeff;
			}
			for (size_t rvj = 0; rvj < rv.shape(1); rvj++) {
				rv(inds(ii), rvj) += multiplication_factor * outputs(i, rvj);
			}
		}
        }
}

void compute_grey_code_row_values_1d(
    nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& row_values,
    nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& mask,
    const nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& inds,
    nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& outputs,
    const nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& shapley_coeff,
    const nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& extended_delta_indexes,
    const int noop_code
) {
	size_t set_size = 0;
	size_t shapley_idx = 0;
	int M = inds.shape(0);
	auto rv = row_values.view();
	gc_require(row_values.shape(0) == mask.shape(0), "row_values and mask disagree on length");
	gc_require(shapley_coeff.shape(0) > 0, "shapley_coeff is empty");
	const size_t n_iter = gc_validate_stream(extended_delta_indexes, mask, (size_t)M, noop_code);
	gc_require(outputs.shape(0) >= n_iter, "outputs has fewer entries than 2**M");
	for (size_t ii = 0; ii < inds.shape(0); ii++) {
		gc_require(inds(ii) >= 0 && (size_t)inds(ii) < mask.shape(0), "inds is out of range for mask");
		gc_require((size_t)inds(ii) < rv.shape(0), "inds is out of range for row_values");
	}
	int delta_ind;
	double on_coeff;
	double off_coeff = shapley_coeff(0);
	double multiplication_factor;
	for (size_t i = 0; i < n_iter; i++) {

		delta_ind = extended_delta_indexes(i);
		if (delta_ind != noop_code) {
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
		on_coeff = shapley_coeff(shapley_idx);
		if (set_size < (size_t)M) {
			off_coeff = shapley_coeff((shapley_idx + 1) % shapley_coeff.shape(0));
		}
		for (size_t ii = 0; ii < inds.shape(0); ii++) {
			if (mask(inds(ii))) {
				multiplication_factor = on_coeff;
			}
			else {
				multiplication_factor = -off_coeff;
			}
		        rv(inds(ii)) += multiplication_factor * outputs(i);
		}
        }
}

void compute_grey_code_row_values_st_1d(
    nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& row_values,
    nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& mask,
    const nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& inds,
    nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& outputs,
    const nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& shapley_coeff,
    const nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& extended_delta_indexes,
    const int noop_code
) {
    // signed, so a desynchronised set_size lands on a negative index that the
    // checks below catch rather than wrapping to ~1.8e19 and reading wild memory
    int64_t set_size = 0;
    const size_t M = inds.shape(0);
    auto rv = row_values.view();
    gc_require(shapley_coeff.shape(0) > 0, "shapley_coeff is empty");
    gc_require(mask.shape(0) >= M, "mask is shorter than inds");
    gc_require(row_values.shape(0) >= M && row_values.shape(1) >= M, "row_values is smaller than M x M");
    const size_t n_iter = gc_validate_stream(extended_delta_indexes, mask, M, noop_code);
    gc_require(outputs.shape(0) >= n_iter, "outputs has fewer entries than 2**M");

    for (size_t i = 0; i < n_iter; i++) {
        const int delta_ind = extended_delta_indexes(i);
        if (delta_ind != noop_code) {
            mask(delta_ind) = !mask(delta_ind);
            if (mask(delta_ind)) {
                set_size += 1;
            } else {
                set_size -= 1;
            }
        }

        for (size_t j = 0; j < M; j++) {
            for (size_t k = j + 1; k < M; k++) {
                double delta;
                if (!mask(j) && !mask(k)) {
                    delta = outputs(i) * shapley_coeff(set_size);
                } else if (mask(j) != mask(k)) {
                    delta = -outputs(i) * shapley_coeff(set_size - 1);
                } else {
                    delta = outputs(i) * shapley_coeff(set_size - 2);
                }
                rv(j, k) += delta;
                rv(k, j) += delta;
            }
        }
    }
}

void compute_grey_code_row_values_st_2d(
    nb::ndarray<double, nb::shape<-1, -1, -1>, nb::device::cpu>& row_values,
    nb::ndarray<bool, nb::shape<-1>, nb::device::cpu>& mask,
    const nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& inds,
    nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& outputs,
    const nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& shapley_coeff,
    const nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& extended_delta_indexes,
    const int noop_code
) {
    // signed, so a desynchronised set_size lands on a negative index that the
    // checks below catch rather than wrapping to ~1.8e19 and reading wild memory
    int64_t set_size = 0;
    const size_t M = inds.shape(0);
    auto rv = row_values.view();
    gc_require(shapley_coeff.shape(0) > 0, "shapley_coeff is empty");
    gc_require(mask.shape(0) >= M, "mask is shorter than inds");
    gc_require(row_values.shape(0) >= M && row_values.shape(1) >= M, "row_values is smaller than M x M");
    const size_t n_iter = gc_validate_stream(extended_delta_indexes, mask, M, noop_code);
    gc_require(outputs.shape(0) >= n_iter, "outputs has fewer rows than 2**M");
    gc_require(row_values.shape(2) >= outputs.shape(1), "row_values has fewer outputs than outputs");

    for (size_t i = 0; i < n_iter; i++) {
        const int delta_ind = extended_delta_indexes(i);
        if (delta_ind != noop_code) {
            mask(delta_ind) = !mask(delta_ind);
            if (mask(delta_ind)) {
                set_size += 1;
            } else {
                set_size -= 1;
            }
        }

        for (size_t j = 0; j < M; j++) {
            for (size_t k = j + 1; k < M; k++) {
                for (size_t output_ind = 0; output_ind < outputs.shape(1); output_ind++) {
                    double delta;
                    if (!mask(j) && !mask(k)) {
                        delta = outputs(i, output_ind) * shapley_coeff(set_size);
                    } else if (mask(j) != mask(k)) {
                        delta = -outputs(i, output_ind) * shapley_coeff(set_size - 1);
                    } else {
                        delta = outputs(i, output_ind) * shapley_coeff(set_size - 2);
                    }
                    rv(j, k, output_ind) += delta;
                    rv(k, j, output_ind) += delta;
                }
            }
        }
    }
}

#endif // GREY_CODE_UTILS_H
