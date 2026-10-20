from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ...skia_geometry import SkPath, unary_union
from ...candidate import FlatFill
from ...drawing_adapter import bind_region_inputs
from ...contour import region_corner_radius
from ...fit import Shape
from ...refine import fit_path_half_fit
from ..automatic_replay import (
    DeclareIntrinsicSymmetryOperation,
    MirrorRegionOperation,
    ReconstructSelfSymmetryOperation,
    draft_input_ref,
    operation_intent,
    apply_intrinsic_symmetry_choice,
)
from ..framework import Proposal
from ..gate import rasterize
from ..symmetry_materialize import (remap_symmetry_clones,
    _is_geometry_root_leaf, _geometry_root_footprint, _materialized_self_symmetry_region)
from ..vector_region import VectorRegion, to_polygon

from ...drawing_symmetry import (
    Axis2D, _polygonal_flat,
    _svg_reflection_matrix, _reflect_flat, _pin_half_shape_to_axis, _canonical_half,
    _fit_geometry_to_path_shape,
    _vertical_half_side, _symmetric_refine_half_shapes,
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


def _is_material_surface(obj: VectorRegion) -> bool:
    """Whether this leaf is a fill partition rather than a semantic shape."""
    return isinstance(obj.diagnostics.get("material_surface"), dict)


def _is_material_surface_root(obj: VectorRegion) -> bool:
    """Whether a branch owns one semantic outline partitioned into materials."""
    return (
        obj.is_branch
        and isinstance(obj.diagnostics.get("geometry_seed"), dict)
        and len(obj.children) > 1
        and all(child.is_leaf and _is_material_surface(child) for child in obj.children)
    )


def _flat_fill_hex(fill: object) -> str | None:
    if isinstance(fill, FlatFill):
        return fill.hex
    return None


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


def _intrinsic_proposal(obj: VectorRegion, axis: Axis2D, residual: float) -> Proposal:
    target_ref = draft_input_ref(obj.id)
    action = DeclareIntrinsicSymmetryOperation(target_ref, _axis_tuple(axis), residual)
    return Proposal(
        (obj.id,),
        [apply_intrinsic_symmetry_choice(obj, axis, residual)],
        intent=operation_intent(action, (target_ref,), (obj.id,)),
    )


def _pair_proposal(
    canonical: VectorRegion,
    canonical_flat: SkPath,
    target: VectorRegion,
    target_flat: SkPath,
    masks: dict[int, np.ndarray],
) -> Proposal | None:
    del masks
    if not _within_ratio(float(canonical_flat.area), float(target_flat.area), _AREA_RATIO_TOL):
        return None
    if not _within_ratio(float(canonical_flat.length), float(target_flat.length), _PERIMETER_RATIO_TOL):
        return None
    axis = _mirror_pair_axis(canonical_flat, target_flat)
    if axis is None:
        return None
    matrix = _svg_reflection_matrix(axis)
    reflected = canonical_flat.affine_transform(matrix)
    if _residual(reflected, target_flat) > _PAIR_RESIDUAL_TOL:
        return None
    source_ref = draft_input_ref(canonical.id)
    target_ref = draft_input_ref(target.id)
    action = MirrorRegionOperation(source_ref, target_ref, _axis_tuple(axis), matrix)
    from ..automatic_replay import apply_relationship_choice
    return Proposal(
        (canonical.id, target.id),
        apply_relationship_choice(action, bind_region_inputs({'source': canonical, 'target': target})),
        intent=operation_intent(
            action,
            (source_ref, target_ref),
            (canonical.id, target.id),
        ),
    )


def _material_root_self_symmetry_proposal(
    obj: VectorRegion,
    flat: SkPath,
    mask: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Proposal | None:
    """Mirror a semantic root, then reclip its material children to that outline.

    A drawing trace keeps continuous fills as child regions so later fill work
    can remain localized.  Those children are not the object whose exterior is
    symmetric; applying the leaf-only reconstruction to them either changes a
    colour boundary or does nothing.  Reconstruct the seed footprint once and
    preserve every child as an intersection with the new exterior instead.
    """
    best = _best_self_reconstruction(
        flat,
        mask,
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
    )
    if best is None:
        return None
    axis = best.axis
    half_shape = best.shape
    residual = best.residual
    fitted_half = to_polygon(half_shape)
    if fitted_half.is_empty:
        return None
    reconstructed = unary_union([fitted_half, _reflect_flat(fitted_half, axis)])
    if not isinstance(reconstructed, SkPath) or reconstructed.is_empty:
        return None

    children: list[VectorRegion] = []
    for child in obj.children:
        child_flat = _polygonal_flat(child.footprint)
        if child_flat is None:
            return None
        clipped = child_flat.intersection(reconstructed)
        if clipped.is_empty:
            return None
        shape = _fit_geometry_to_path_shape(clipped, epsilon=epsilon, max_error=max_error, cubic=cubic)
        if shape is None:
            return None
        children.append(
            child.with_current(
                shape,
                footprint=clipped,
                raster=rasterize(clipped, mask.shape),
                diagnostics={
                    "symmetry": {
                        "accepted": True,
                        "mode": "root_clip",
                        "root": int(obj.id),
                    }
                },
            )
        )

    return Proposal(
        (obj.id,),
        [
            VectorRegion.branch(
                id=obj.id,
                children=children,
                z=obj.z,
                raster=rasterize(reconstructed, mask.shape),
                footprint=reconstructed,
                fill=obj.fill,
                source_label=obj.source_label,
                color_hex=obj.color_hex,
                drawing_id=obj.drawing_id,
                source_regions=obj.source_regions,
                diagnostics={
                    "symmetry": {
                        "accepted": True,
                        "mode": "root_self",
                        "axis": {
                            "theta": float(axis.theta),
                            "cx": float(axis.cx),
                            "cy": float(axis.cy),
                        },
                        "residual": float(residual),
                    }
                },
            )
        ],
    )


def symmetry_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    cubic: bool = False,
) -> list[Proposal]:
    proposals: list[Proposal] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not _is_material_surface_root(obj) or obj.id not in masks:
            continue
        # A material-surface root renders its children, not its own footprint.
        # Mirroring the root then clipping/re-fitting the original children can
        # emit a scene that is less symmetric than the input despite the root
        # footprint itself being symmetric.  Until geometry and material layers
        # have distinct render representations, automatic self-symmetry must
        # fail closed for these roots.  Explicit plans may still reconstruct
        # their geometry with a deliberate fill assignment.
        continue

    usable: list[tuple[VectorRegion, SkPath]] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        # Compound splitting creates implementation children for nested
        # cutouts.  They are not independently trace-owned semantic shapes,
        # so automatic self-reflection must never reconstruct them directly.
        if isinstance(obj.diagnostics.get("compound"), dict):
            continue
        flat = _polygonal_flat(obj.footprint)
        if flat is None or obj.current.kind == "use" or obj.id not in masks:
            continue
        if not _automatic_symmetry_is_tractable(flat):
            continue
        usable.append((obj, flat))

    paired_ids: set[int] = set()

    for index, (canonical, canonical_flat) in enumerate(usable):
        if canonical.id in paired_ids:
            continue
        for target, target_flat in usable[index + 1:]:
            if target.id in paired_ids:
                continue
            proposal = _pair_proposal(
                canonical,
                canonical_flat,
                target,
                target_flat,
                masks,
            )
            if proposal is None:
                continue
            proposals.append(proposal)
            paired_ids.update({canonical.id, target.id})
            break

    for obj, flat in usable:
        if obj.id in paired_ids:
            continue
        # ``material_surface`` leaves partition one geometry root by colour.
        # They are not independent semantic shapes, so a self-reflection can
        # preserve their footprint while materially changing the rendered
        # artwork.  Automatic symmetry belongs on geometry roots; callers may
        # still apply an explicit symmetry plan to a selected material leaf.
        geometry_root_leaf = _is_geometry_root_leaf(obj)
        if _is_material_surface(obj) and not geometry_root_leaf:
            continue
        # A gradient is fitted onto finalized geometry.  It can carry a
        # verified symmetry relationship, but must not cause a self-symmetry
        # rewrite that replaces that geometry with a mirrored half.
        if not isinstance(obj.fill, FlatFill) and not geometry_root_leaf:
            intrinsic = _best_self_axis(flat)
            if intrinsic is not None:
                axis, residual = intrinsic
                proposals.append(_intrinsic_proposal(obj, axis, residual))
            continue
        # Native primitives already encode exact geometric symmetry.  Verify
        # and report the relation without replacing their deterministic form.
        if obj.current.kind in {"circle", "ellipse", "rect"}:
            intrinsic = _best_self_axis(flat)
            if intrinsic is not None:
                axis, residual = intrinsic
                proposals.append(_intrinsic_proposal(obj, axis, residual))
            continue
        geometry = obj.diagnostics.get("geometry") if isinstance(obj.diagnostics, dict) else None
        explicit_kind = geometry.get("explicit") if isinstance(geometry, dict) else None
        if explicit_kind in {"circle", "ellipse", "rect", "rounded_rect", "cap", "trapezoid", "rounded_trapezoid"}:
            intrinsic = _best_self_axis(flat)
            if intrinsic is not None:
                axis, residual = intrinsic
                proposals.append(_intrinsic_proposal(obj, axis, residual))
            continue
        if obj.current.kind != "path":
            continue
        preserve_explicit_geometry = explicit_kind is not None
        reconstruction_flat = _geometry_root_footprint(obj, flat)
        best = _best_self_reconstruction(
            reconstruction_flat,
            masks[obj.id],
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
            preserve_explicit_geometry=preserve_explicit_geometry,
        )
        if best is None:
            # Detection is useful even when fitting a replacement would damage
            # an already symmetric path. Only report an unchanged freeform path
            # as intrinsic when its reflected residual is effectively zero;
            # a merely approximate axis still needs an accepted reconstruction.
            intrinsic = _best_self_axis(flat)
            if intrinsic is not None and intrinsic[1] <= 1e-9:
                axis, residual = intrinsic
                proposals.append(_intrinsic_proposal(obj, axis, residual))
            continue
        axis = best.axis
        half = best.half
        reflected_half = best.reflected_half
        shape = best.shape
        residual = best.residual
        target_ref = draft_input_ref(obj.id)
        action = ReconstructSelfSymmetryOperation(
            target_ref,
            (),
            _axis_tuple(axis),
            residual,
            best.fit_kind,
            best.side,
            best.corner_radius,
            epsilon,
            max_error,
            cubic,
        )
        proposals.append(
            Proposal(
                (obj.id,),
                [
                    _materialized_self_symmetry_region(
                        obj,
                        axis=axis,
                        half=half,
                        reflected_half=reflected_half,
                        half_shape=shape,
                        residual=residual,
                    )
                ],
                intent=operation_intent(action, (target_ref,), (obj.id,)),
            )
        )

    # A source reconstruction also changes every symbolic instance. Include
    # affected clones in the proposal so the existing gate assesses the scene
    # that will actually be retained and emitted.
    for index, proposal in enumerate(proposals):
        sources = {
            (replacement.id,): (replacement.id,) for replacement in proposal.new_objects
            if replacement.diagnostics.get("symmetry", {}).get("materialized")
        }
        if not sources:
            continue
        remaining = [obj for obj in objects if obj.id not in proposal.obj_ids]
        candidate = [*proposal.new_objects, *remaining]
        remapped = remap_symmetry_clones(candidate, sources, before=objects)
        changed = [after for before, after in zip(remaining, remapped[len(proposal.new_objects):]) if before is not after]
        if changed:
            proposals[index] = Proposal(
                (*proposal.obj_ids, *(obj.id for obj in changed)),
                [*proposal.new_objects, *changed],
                geometry_policy=proposal.geometry_policy,
                intent=(
                    operation_intent(
                        replace(
                            proposal.intent.action,
                            dependent_clones=tuple(draft_input_ref(obj.id) for obj in changed),
                        ),
                        (
                            *proposal.intent.inputs,
                            *(draft_input_ref(obj.id) for obj in changed),
                        ),
                        (*proposal.obj_ids, *(obj.id for obj in changed)),
                    )
                    if proposal.intent is not None
                    and isinstance(proposal.intent.action, ReconstructSelfSymmetryOperation)
                    else proposal.intent
                ),
            )
    return proposals
