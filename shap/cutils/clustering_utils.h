#ifndef CLUSTERING_UTILS_H
#define CLUSTERING_UTILS_H

#include <cassert>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

void reverse_window(
	nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>& order,
	const int start,
	const int length
) {
	assert(length >= 0);
	assert(start >= 0);
	assert((size_t) start + (size_t) length <= order.shape(0));
	for(size_t i = 0; i < (size_t) length / 2; i++) {
		int64_t tmp = order(start + i);
		order((size_t) start + i) = order((size_t) start + (size_t) length - i - 1);
		order((size_t) start + (size_t) length - i - 1) = tmp;
	}
}

#endif // CLUSTERING_UTILS_H
