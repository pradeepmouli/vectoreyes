from __future__ import annotations

from dataclasses import replace

import numpy as np

from ...skia_geometry import SkPath, unary_union
from ...candidate import FlatFill
from ... import symmetry_detection as detection
from ...drawing_adapter import bind_region_inputs
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
from ..region_sample import RegionSample, to_polygon

from ...drawing_symmetry import (
    Axis2D, _polygonal_flat,
    _reflect_flat, _fit_geometry_to_path_shape,
)


def _is_material_surface(obj: RegionSample) -> bool:
    """Whether this leaf is a fill partition rather than a semantic shape."""
    return isinstance(obj.diagnostics.get("material_surface"), dict)


def _is_material_surface_root(obj: RegionSample) -> bool:
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


def _intrinsic_proposal(obj: RegionSample, axis: Axis2D, residual: float) -> Proposal:
    target_ref = draft_input_ref(obj.id)
    action = DeclareIntrinsicSymmetryOperation(target_ref, detection._axis_tuple(axis), residual)
    return Proposal(
        (obj.id,),
        [apply_intrinsic_symmetry_choice(obj, axis, residual)],
        intent=operation_intent(action, (target_ref,), (obj.id,)),
    )


def _pair_proposal(
    canonical: RegionSample,
    canonical_flat: SkPath,
    target: RegionSample,
    target_flat: SkPath,
    masks: dict[int, np.ndarray],
) -> Proposal | None:
    del masks
    match = detection.mirror_pair(canonical_flat, target_flat)
    if match is None:
        return None
    axis, matrix = match
    source_ref = draft_input_ref(canonical.id)
    target_ref = draft_input_ref(target.id)
    action = MirrorRegionOperation(source_ref, target_ref, detection._axis_tuple(axis), matrix)
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
    obj: RegionSample,
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
    best = detection._best_self_reconstruction(
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

    children: list[RegionSample] = []
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
            RegionSample.branch(
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
    objects: list[RegionSample],
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

    usable: list[tuple[RegionSample, SkPath]] = []
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
        if not detection._automatic_symmetry_is_tractable(flat):
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
            intrinsic = detection._best_self_axis(flat)
            if intrinsic is not None:
                axis, residual = intrinsic
                proposals.append(_intrinsic_proposal(obj, axis, residual))
            continue
        # Native primitives already encode exact geometric symmetry.  Verify
        # and report the relation without replacing their deterministic form.
        if obj.current.kind in {"circle", "ellipse", "rect"}:
            intrinsic = detection._best_self_axis(flat)
            if intrinsic is not None:
                axis, residual = intrinsic
                proposals.append(_intrinsic_proposal(obj, axis, residual))
            continue
        geometry = obj.diagnostics.get("geometry") if isinstance(obj.diagnostics, dict) else None
        explicit_kind = geometry.get("explicit") if isinstance(geometry, dict) else None
        if explicit_kind in {"circle", "ellipse", "rect", "rounded_rect", "cap", "trapezoid", "rounded_trapezoid"}:
            intrinsic = detection._best_self_axis(flat)
            if intrinsic is not None:
                axis, residual = intrinsic
                proposals.append(_intrinsic_proposal(obj, axis, residual))
            continue
        if obj.current.kind != "path":
            continue
        preserve_explicit_geometry = explicit_kind is not None
        reconstruction_flat = _geometry_root_footprint(obj, flat)
        best = detection._best_self_reconstruction(
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
            intrinsic = detection._best_self_axis(flat)
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
            detection._axis_tuple(axis),
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
