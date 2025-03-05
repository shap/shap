#cython: language_level=3


import numpy as np
cimport numpy as cnp
import cython

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def _exp_val(int nsamples_run,
             int nsamples_added,
             int D,
             int N,
             double[::1] weights,
             double[:,:] y,
             double[:,:] ey):

    cdef:
        double[::1] ref = np.zeros(D)
        double[::1] eyVal = np.zeros(D)
        int i, j, k

    for i in range(nsamples_added):
        if i < nsamples_run:
            continue
        eyVal[:] = ref
        for j in range(N):
            for k in range(D):
                eyVal[k] += y[i * N + j, k] * weights[j]

        ey[i, :] = eyVal
        nsamples_run += 1
    return ey, nsamples_run
