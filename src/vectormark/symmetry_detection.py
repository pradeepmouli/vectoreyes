"""Geometry-only symmetry discovery shared by automatic and agent actions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .skia_geometry import SkPath, unary_union
from .contour import region_corner_radius
from .fit import Shape
from .refine import fit_path_half_fit
from .refinement.region_sample import to_polygon
from .drawing_symmetry import (
    Axis2D, _svg_reflection_matrix, _reflect_flat, _pin_half_shape_to_axis,
    _canonical_half, _fit_geometry_to_path_shape, _vertical_half_side,
    _symmetric_refine_half_shapes,
)


@dataclass(frozen=True)
class _SelfReconstruction:
    axis: Axis2D
    half: SkPath
    reflected_half: SkPath
    shape: Shape
    residual: float
    fit_kind: str
    side: str
    corner_radius: float

_ANGLE_EPS = 1e-9
_AREA_RATIO_TOL = 0.03
_PERIMETER_RATIO_TOL = 0.03
_PAIR_RESIDUAL_TOL = 0.02
# Candidate threshold only.  A proposed symmetric reconstruction must still
# pass the fitted-geometry and framework scene-fidelity checks below.
_SELF_RESIDUAL_TOL = 0.04
_FITTED_RECONSTRUCTION_RESIDUAL_TOL = 0.04
_MAX_AUTO_SYMMETRY_INTERIORS = 8
_MAX_AUTO_SYMMETRY_COMPONENTS = 4


def _automatic_symmetry_is_tractable(flat: SkPath) -> bool:
    """Keep automatic reconstruction bounded on noisy compound trace roots.

    A region with many counters/components has ambiguous semantic symmetry and
    makes each reflected-union residual disproportionately expensive.  The
    original path remains intact; callers may still apply explicit symmetry
    relationships when that intent is known.
    """
    return len(flat.interiors) <= _MAX_AUTO_SYMMETRY_INTERIORS and len(flat.geoms) <= _MAX_AUTO_SYMMETRY_COMPONENTS


def _normalize_angle(theta: float) -> float:
    out = math.fmod(theta, math.pi)
    if out < 0.0:
        out += math.pi
    if abs(out) < _ANGLE_EPS or abs(out - math.pi) < _ANGLE_EPS:
        return 0.0
    return float(out)


def _axis_key(axis: Axis2D) -> tuple[float, float, float]:
    theta = _normalize_angle(float(axis.theta))
    offset = float(axis.cx * math.sin(theta) - axis.cy * math.cos(theta))
    return (round(theta, 9), round(offset, 6), round(float(axis.cx), 6))


def _within_ratio(a: float, b: float, tol: float) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= tol


def _residual(a: SkPath, b: SkPath) -> float:
    scale = max(float(a.area), float(b.area), 1.0)
    return float(a.symmetric_difference(b).area / scale)


def _orientation_axes(flat: SkPath) -> list[float]:
    angles = [0.0, math.pi / 2.0]
    rect = flat.minimum_rotated_rectangle
    if not rect.is_empty:
        coords = list(rect.exterior.coords)
        for (x1, y1), (x2, y2) in zip(coords, coords[1:], strict=False):
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < _ANGLE_EPS and abs(dy) < _ANGLE_EPS:
                continue
            angle = _normalize_angle(math.atan2(dy, dx))
            angles.extend([angle, _normalize_angle(angle + math.pi / 2.0)])

    out: list[float] = []
    for angle in angles:
        if all(abs(_normalize_angle(angle - existing)) > 1e-6 for existing in out):
            out.append(angle)
    return sorted(out)


def _self_axis_candidates(flat: SkPath) -> list[Axis2D]:
    centroid = flat.centroid
    axes = [Axis2D(theta, float(centroid.x), float(centroid.y)) for theta in _orientation_axes(flat)]
    return sorted(axes, key=_axis_key)


def _mirror_pair_axis(
    canonical: SkPath,
    target: SkPath,
) -> Axis2D | None:
    cc = canonical.centroid
    tc = target.centroid
    dx = float(tc.x - cc.x)
    dy = float(tc.y - cc.y)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    theta = _normalize_angle(math.atan2(dy, dx) + math.pi / 2.0)
    return Axis2D(theta, float((cc.x + tc.x) / 2.0), float((cc.y + tc.y) / 2.0))


def _best_self_axis(flat: SkPath) -> tuple[Axis2D, float] | None:
    """Find a valid reflection axis without changing the represented geometry."""
    candidates = [
        (_residual(_reflect_flat(flat, axis), flat), axis)
        for axis in _self_axis_candidates(flat)
    ]
    candidates = [candidate for candidate in candidates if candidate[0] <= _SELF_RESIDUAL_TOL]
    if not candidates:
        return None
    residual, axis = min(candidates, key=lambda item: (round(item[0], 12), _axis_key(item[1])))
    return axis, residual


def _best_self_reconstruction(
    flat: SkPath,
    mask: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
    preserve_explicit_geometry: bool = False,
) -> _SelfReconstruction | None:
    corner_radius = region_corner_radius(mask)
    candidates: list[_SelfReconstruction] = []
    for axis in _self_axis_candidates(flat):
        reflected = _reflect_flat(flat, axis)
        if _residual(reflected, flat) > _SELF_RESIDUAL_TOL:
            continue
        half = _canonical_half(flat, axis)
        if half is None:
            continue
        reflected_half = _reflect_flat(half, axis)
        reconstructed = unary_union([half, reflected_half])
        if reconstructed is None:
            continue
        if reconstructed.is_empty or not isinstance(reconstructed, SkPath):
            continue
        shapes: tuple[tuple[str, Shape], ...] = ()
        if not preserve_explicit_geometry:
            shapes = _symmetric_refine_half_shapes(
                flat,
                half,
                axis,
                epsilon=epsilon,
                max_error=max_error,
                corner_radius=corner_radius,
                cubic=cubic,
            )
        def accept(shape: Shape) -> tuple[float, Shape] | None:
            pinned = _pin_half_shape_to_axis(shape, axis)
            if pinned is None:
                return None
            fitted_half = to_polygon(pinned)
            if fitted_half.is_empty:
                return None
            fitted_reconstructed = unary_union([fitted_half, _reflect_flat(fitted_half, axis)])
            fitted_residual = _residual(fitted_reconstructed, flat)
            if fitted_residual > _FITTED_RECONSTRUCTION_RESIDUAL_TOL:
                return None
            return fitted_residual, pinned

        selected: tuple[str, tuple[float, Shape]] | None = next(
            ((fit_kind, result) for fit_kind, shape in shapes if (result := accept(shape)) is not None),
            None,
        )
        side = _vertical_half_side(half, axis)
        if selected is None:
            contour = np.asarray(flat.exterior.coords, dtype=float)
            generic_shapes = [
                ("path_half", fit_path_half_fit(
                    contour,
                    float(axis.cx),
                    side=side,
                    epsilon=epsilon,
                    max_error=max_error,
                    cubic=cubic,
                )),
                ("geometry_path", _fit_geometry_to_path_shape(
                    half, epsilon=epsilon, max_error=max_error, cubic=cubic
                )),
            ]
            selected = next(
                (
                    (fit_kind, result)
                    for fit_kind, shape in generic_shapes
                    if shape is not None and (result := accept(shape)) is not None
                ),
                None,
            )
        if selected is not None:
            fit_kind, (fitted_residual, pinned) = selected
            candidates.append(_SelfReconstruction(
                axis,
                half,
                reflected_half,
                pinned,
                fitted_residual,
                fit_kind,
                side,
                corner_radius,
            ))

    if not candidates:
        return None

    winner = min(
        candidates,
        key=lambda item: (
            round(item.residual, 12),
            _axis_key(item.axis),
            item.shape.params["d"],
        ),
    )
    return winner


def _axis_tuple(axis: Axis2D) -> tuple[float, float, float]:
    return float(axis.theta), float(axis.cx), float(axis.cy)


def mirror_pair(
    canonical: SkPath, target: SkPath,
) -> tuple[Axis2D, tuple[float, float, float, float, float, float]] | None:
    """Return the existing reflection winner, without publishing a region."""
    if not _within_ratio(float(canonical.area), float(target.area), _AREA_RATIO_TOL):
        return None
    if not _within_ratio(float(canonical.length), float(target.length), _PERIMETER_RATIO_TOL):
        return None
    axis = _mirror_pair_axis(canonical, target)
    if axis is None:
        return None
    matrix = _svg_reflection_matrix(axis)
    if _residual(canonical.affine_transform(matrix), target) > _PAIR_RESIDUAL_TOL:
        return None
    return axis, matrix
