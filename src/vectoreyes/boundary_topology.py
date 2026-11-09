"""Bounded nonintersection certificates for retained polynomial boundaries.

No sampling or Boolean repair is used. Exact rational control hulls and binary
de Casteljau subdivision prove separation. Unresolved cases reject, including
work/depth exhaustion. Only adjacent declared endpoints may meet. The same
proof is applied to six-decimal emitted SVG controls as well as retained ones.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from .path_geometry import LineCommand, QuadraticCommand, CubicCommand


@dataclass
class _Budget:
    remaining: int

    def consume(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError('boundary nonintersection certificate work limit exceeded')


def _split(curve):
    rows = [curve]
    while len(rows[-1]) > 1:
        rows.append(tuple(tuple((a + b) / 2 for a, b in zip(left, right))
            for left, right in zip(rows[-1], rows[-1][1:])))
    return tuple(row[0] for row in rows), tuple(row[-1] for row in reversed(rows))


def _monotone_pieces(curve, budget, depth=0):
    budget.consume()
    for axis in (0, 1):
        values = [point[axis] for point in curve]
        if values[0] != values[-1] and (
                all(a <= b for a, b in zip(values, values[1:]))
                or all(a >= b for a, b in zip(values, values[1:]))):
            return [curve]
    if depth >= 30:
        raise ValueError('boundary nonintersection could not certify a simple curve')
    left, right = _split(curve)
    return [*_monotone_pieces(left, budget, depth + 1), *_monotone_pieces(right, budget, depth + 1)]


def _only_allowed_contacts(curve, projections, value, allowed):
    # Positive Bernstein bases make the projection strictly interior unless
    # every coefficient lies on the plane. Thus only zero-valued endpoints can
    # touch the separating plane; all such endpoints must be explicitly allowed.
    if all(item == value for item in projections):
        return False
    return all(point in allowed for point, projection in (
        (curve[0], projections[0]), (curve[-1], projections[-1])) if projection == value)


def _separated(first, second, allowed):
    axes = [(1, 0), (0, 1)]
    for curve in (first, second):
        for left, right in combinations(curve, 2):
            dx, dy = right[0] - left[0], right[1] - left[1]
            if dx or dy:
                axes.append((-dy, dx))
    for x, y in axes:
        a = tuple(x * point[0] + y * point[1] for point in first)
        b = tuple(x * point[0] + y * point[1] for point in second)
        amin, amax, bmin, bmax = min(a), max(a), min(b), max(b)
        if amax < bmin or bmax < amin:
            return True
        if allowed and (amax == bmin or bmax == amin):
            value = amax if amax == bmin else bmax
            if (_only_allowed_contacts(first, a, value, allowed)
                    or _only_allowed_contacts(second, b, value, allowed)):
                return True
    return False


def _disjoint(first, second, allowed, budget, depth=0):
    budget.consume()
    if ({first[0], first[-1]} & {second[0], second[-1]}) - allowed:
        raise ValueError('boundary nonintersection failed: undeclared boundary contact')
    if _separated(first, second, allowed):
        return
    if depth >= 30 or len(first) == len(second) == 2:
        raise ValueError('boundary nonintersection failed: crossing or unresolved contact')
    extent = lambda curve: sum(max(p[axis] for p in curve) - min(p[axis] for p in curve) for axis in (0, 1))
    if len(second) == 2 or (len(first) != 2 and extent(first) >= extent(second)):
        left, right = _split(first)
        _disjoint(left, second, allowed, budget, depth + 1)
        _disjoint(right, second, allowed, budget, depth + 1)
    else:
        left, right = _split(second)
        _disjoint(first, left, allowed, budget, depth + 1)
        _disjoint(first, right, allowed, budget, depth + 1)


def _curves(path, *, emitted):
    def point(value):
        return tuple(Fraction(f'{coordinate:.6f}') if emitted else Fraction(float(coordinate))
            for coordinate in value)
    start = point(path.path.start)
    current = start
    result = []
    for command in path.path.drawable_commands:
        end = point(command.end)
        if isinstance(command, LineCommand):
            curve = (current, end)
        elif isinstance(command, QuadraticCommand):
            curve = (current, point(command.control), end)
        elif isinstance(command, CubicCommand):
            curve = (current, point(command.control_1), point(command.control_2), end)
        else:
            raise ValueError('boundary nonintersection certificate requires L/Q/C commands')
        if any(value != curve[0] for value in curve[1:]):
            result.append(curve)
        current = end
    if current != start:
        result.append((current, start))
    return result


def certify_simple_boundaries(paths, *, work_limit=65536):
    """Prove simple, mutually noncrossing L/Q/C contours without changing them.

    Concavity, inflections and nested hole contours do not change the proof.
    Unsupported arcs or ambiguous/touching nonadjacent boundaries are rejected,
    not flattened or normalized into apparent validity. Work is strictly bounded.
    """
    if isinstance(work_limit, bool) or not isinstance(work_limit, int) or work_limit <= 0:
        raise ValueError('boundary nonintersection work limit must be a positive integer')
    paths = tuple(paths)
    budget = _Budget(work_limit)
    for emitted in (False, True):
        contours = []
        for path in paths:
            pieces = [piece for curve in _curves(path, emitted=emitted)
                for piece in _monotone_pieces(curve, budget)]
            if len(pieces) < 2:
                raise ValueError('boundary nonintersection failed: degenerate contour')
            contours.append(pieces)
        indexed = [(contour, index, curve) for contour, pieces in enumerate(contours)
            for index, curve in enumerate(pieces)]
        for (a_contour, a_index, first), (b_contour, b_index, second) in combinations(indexed, 2):
            allowed = set()
            if a_contour == b_contour:
                if b_index == a_index + 1:
                    allowed.add(first[-1])
                if a_index == 0 and b_index == len(contours[a_contour]) - 1:
                    allowed.add(first[0])
            _disjoint(first, second, allowed, budget)
