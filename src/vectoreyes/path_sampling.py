"""Shared SVG command sampling; independent of refinement region records."""

from __future__ import annotations

import math
import re
import numpy as np

_NUM = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PATH_TOKEN = re.compile(rf"[MLQCAZ]|{_NUM}")


def _parse_subpaths(d: str) -> list[list[tuple[str, list[float]]]]:
    raw = _PATH_TOKEN.findall(d)
    if not raw:
        return []

    coord_count = {"M": 2, "L": 2, "Q": 4, "C": 6, "A": 7, "Z": 0}
    subpaths: list[list[tuple[str, list[float]]]] = []
    current: list[tuple[str, list[float]]] = []
    i = 0
    while i < len(raw):
        cmd = raw[i]
        if cmd not in coord_count:
            raise ValueError(f"expected path command, got {cmd!r}")
        i += 1
        count = coord_count[cmd]
        values = [float(token) for token in raw[i:i + count]]
        if len(values) != count:
            raise ValueError(f"incomplete SVG path command {cmd!r}")
        i += count
        if cmd == "M" and current:
            subpaths.append(current)
            current = []
        current.append((cmd, values))
    if current:
        subpaths.append(current)
    return subpaths


def _point_tuple(pt: np.ndarray) -> tuple[float, float]:
    return (float(pt[0]), float(pt[1]))


def _sample_arc(
    start: np.ndarray,
    values: list[float],
    samples: int,
) -> list[tuple[float, float]]:
    rx, ry, xrot_deg, large_arc, sweep, x2, y2 = values
    end = np.array([x2, y2], dtype=float)
    if np.allclose(start, end):
        return []

    rx = abs(rx)
    ry = abs(ry)
    if rx == 0.0 or ry == 0.0:
        return [_point_tuple(end)]

    phi = math.radians(xrot_deg % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    dx2 = (start[0] - end[0]) / 2.0
    dy2 = (start[1] - end[1]) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale

    rx2 = rx * rx
    ry2 = ry * ry
    x1p2 = x1p * x1p
    y1p2 = y1p * y1p

    denom = rx2 * y1p2 + ry2 * x1p2
    if denom == 0.0:
        return [_point_tuple(end)]
    numer = max(0.0, rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2)
    sign = -1.0 if bool(large_arc) == bool(sweep) else 1.0
    coef = sign * math.sqrt(numer / denom)
    cxp = coef * (rx * y1p) / ry
    cyp = coef * (-ry * x1p) / rx

    cx = cos_phi * cxp - sin_phi * cyp + (start[0] + end[0]) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (start[1] + end[1]) / 2.0

    ux = (x1p - cxp) / rx
    uy = (y1p - cyp) / ry
    vx = (-x1p - cxp) / rx
    vy = (-y1p - cyp) / ry

    theta1 = math.atan2(uy, ux)
    delta = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    if not sweep and delta > 0.0:
        delta -= 2.0 * math.pi
    elif sweep and delta < 0.0:
        delta += 2.0 * math.pi

    pts: list[tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, samples)[1:]:
        theta = theta1 + delta * float(t)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        x = cos_phi * rx * cos_theta - sin_phi * ry * sin_theta + cx
        y = sin_phi * rx * cos_theta + cos_phi * ry * sin_theta + cy
        pts.append((float(x), float(y)))
    return pts

def _sample_subpath(
    tokens: list[tuple[str, list[float]]],
    samples: int,
) -> list[tuple[float, float]]:
    cur = start = None
    pts: list[tuple[float, float]] = []
    for kind, values in tokens:
        if kind == "M":
            cur = np.array(values[:2], dtype=float)
            start = cur
            pts.append(_point_tuple(cur))
        elif kind == "L":
            cur = np.array(values[:2], dtype=float)
            pts.append(_point_tuple(cur))
        elif kind == "Q":
            assert cur is not None
            c = np.array(values[:2], dtype=float)
            p = np.array(values[2:4], dtype=float)
            for t in np.linspace(0, 1, samples)[1:]:
                pt = (1 - t) ** 2 * cur + 2 * (1 - t) * t * c + t**2 * p
                pts.append(_point_tuple(pt))
            cur = p
        elif kind == "C":
            assert cur is not None
            c1 = np.array(values[:2], dtype=float)
            c2 = np.array(values[2:4], dtype=float)
            p = np.array(values[4:6], dtype=float)
            for t in np.linspace(0, 1, samples)[1:]:
                pt = (
                    (1 - t) ** 3 * cur
                    + 3 * (1 - t) ** 2 * t * c1
                    + 3 * (1 - t) * t**2 * c2
                    + t**3 * p
                )
                pts.append(_point_tuple(pt))
            cur = p
        elif kind == "A":
            assert cur is not None
            pts.extend(_sample_arc(cur, values, samples))
            cur = np.array(values[5:7], dtype=float)
        elif kind == "Z":
            cur = start
    return pts
