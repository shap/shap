#ifndef KERNEL_EXPLAINER_UTILS_H
#define KERNEL_EXPLAINER_UTILS_H

#include <cassert>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

int compute_exp_val(
    int nsamples_run,
    const int nsamples_added,
    const int D,
    const int N,
    const nb::ndarray<double, nb::shape<-1>, nb::device::cpu>& weights,
    const nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& y,
    nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu>& ey
) {
	std::vector<double> eyVal(D, 0.0);

        // I assume we could choose i in the range of std::min(nsamples_added, nsamples_run)
	for(size_t i = 0; i < nsamples_added; i++) {
		if (i < nsamples_run) {
			continue;
		}
		std::fill(eyVal.begin(), eyVal.end(), 0.);
		for(size_t j = 0; j < N; j++) {
			for (size_t k = 0; k < D; k++) {
				eyVal[k] += y(i * N + j, k) * weights(j);
			}
		}
		assert(ey.shape(1) <= D);
		for(size_t colIdx = 0; colIdx < ey.shape(1); colIdx++) {
			ey(i, colIdx) = eyVal[colIdx];
		}
		nsamples_run += 1;
	}
	return nsamples_run;

}

#endif // KERNEL_EXPLAINER_UTILS_H
