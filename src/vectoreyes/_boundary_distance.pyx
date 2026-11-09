"""Compiled boundary distance; preserves the linearized edge-pair algorithm."""

from libc.math cimport hypot, INFINITY
cimport cython
import numpy as np


cdef inline double orientation(double ax, double ay, double bx, double by,
                               double cx, double cy) noexcept nogil:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


cdef inline bint on_segment(double px, double py, double qx, double qy,
                            double rx, double ry) noexcept nogil:
    return min(px, rx) <= qx <= max(px, rx) and min(py, ry) <= qy <= max(py, ry)


cdef inline double point_distance(double px, double py, double ax, double ay,
                                  double bx, double by) noexcept nogil:
    cdef double dx = bx - ax, dy = by - ay
    cdef double length_sq = dx * dx + dy * dy
    cdef double t
    if length_sq == 0.0:
        return hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return hypot(px - (ax + t * dx), py - (ay + t * dy))


cdef inline double segment_distance(double ax, double ay, double bx, double by,
                                    double cx, double cy, double dx, double dy) noexcept nogil:
    cdef double o1 = orientation(ax, ay, bx, by, cx, cy)
    cdef double o2 = orientation(ax, ay, bx, by, dx, dy)
    cdef double o3 = orientation(cx, cy, dx, dy, ax, ay)
    cdef double o4 = orientation(cx, cy, dx, dy, bx, by)
    if ((o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)) or (
        (o1 == 0 and on_segment(ax, ay, cx, cy, bx, by)) or
        (o2 == 0 and on_segment(ax, ay, dx, dy, bx, by)) or
        (o3 == 0 and on_segment(cx, cy, ax, ay, dx, dy)) or
        (o4 == 0 and on_segment(cx, cy, bx, by, dx, dy))
    ):
        return 0.0
    return min(point_distance(ax, ay, cx, cy, dx, dy),
               point_distance(bx, by, cx, cy, dx, dy),
               point_distance(cx, cy, ax, ay, bx, by),
               point_distance(dx, dy, ax, ay, bx, by))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double edge_distance(const double[:, ::1] a, const double[:, ::1] b) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double distance, best = INFINITY
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            distance = segment_distance(a[i, 0], a[i, 1], a[i, 2], a[i, 3],
                                        b[j, 0], b[j, 1], b[j, 2], b[j, 3])
            if distance < best:
                best = distance
            if best == 0.0:
                return 0.0
    return best


def pack_edges(rings):
    """Pack closed rings without adding or removing any boundary segments."""
    arrays = []
    for ring in rings:
        points = np.asarray(ring, dtype=np.float64)
        if len(points) == 0:
            continue
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("boundary rings must contain two-dimensional points")
        arrays.append(np.concatenate((points, np.roll(points, -1, axis=0)), axis=1))
    return np.ascontiguousarray(np.concatenate(arrays) if arrays else np.empty((0, 4)))


def boundary_distance(a_rings, b_rings):
    """Minimum boundary distance, with the existing empty-path convention."""
    if not a_rings or not b_rings:
        return 0.0
    cdef const double[:, ::1] a = pack_edges(a_rings)
    cdef const double[:, ::1] b = pack_edges(b_rings)
    cdef double result
    with nogil:
        result = edge_distance(a, b)
    return result
