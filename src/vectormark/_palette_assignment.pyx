"""Nearest-color assignment without a pixels-by-palette distance array."""

import numpy as np
cimport cython
from libc.math cimport sqrt, isnan, INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
def nearest_palette(const double[:, ::1] pixels, const double[:, ::1] palette):
    """Return first-minimum indices for three-channel Euclidean distances."""
    if pixels.shape[1] != 3 or palette.shape[1] != 3:
        raise ValueError("pixels and palette must have three channels")
    if palette.shape[0] == 0:
        raise ValueError("attempt to get argmin of an empty sequence")
    result = np.empty(pixels.shape[0], dtype=np.intp)
    cdef Py_ssize_t[::1] labels = result
    cdef Py_ssize_t i, j, best_index
    cdef double dx, dy, dz, distance, best
    with nogil:
        for i in range(pixels.shape[0]):
            best_index = 0
            best = INFINITY
            for j in range(palette.shape[0]):
                dx = pixels[i, 0] - palette[j, 0]
                dy = pixels[i, 1] - palette[j, 1]
                dz = pixels[i, 2] - palette[j, 2]
                # Retain sqrt and summation order: rounding-induced ties
                # must select the same first entry as NumPy norm/argmin.
                distance = sqrt(dx * dx + dy * dy + dz * dz)
                if isnan(distance):
                    best_index = j
                    break
                if distance < best:
                    best = distance
                    best_index = j
            labels[i] = best_index
    return result
