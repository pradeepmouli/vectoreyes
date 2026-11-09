"""Top-level orchestration: raster path/array -> structured SVG string."""
from __future__ import annotations

import dataclasses
import re
import types
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .color import extract_palette, quantize
from .contour import region_corner_radius
from .emit import (
    apply_affine_point,
    fill_rule_for,
    mirror_use,
    path_svg,
    reflect_path_d,
    render_svg_doc,
    resolve_fill,
    resolve_use_shape,
    shape_to_path_d,
    shape_to_svg,
    transform_path_d,
)
from .candidate import Candidate, Fill, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .components import decompose_components
from .fill_fit import fit_fill
from .fit import Shape, _fmt
from .occlusion import ScenePrimitive, reconstruct_scene
from .surface_merge import merge_surfaces
from .segment import segment
from .selection import SelectionPolicy
from .refinement_settings import RefinementPasses
from .selector import select_geometry
from .symmetry import detect_axis, detect_symmetry_groups, detect_symmetry_rotation, sym_off_ratio, Axis2D
from .types import Axis, Region


@dataclass
class Options:
    epsilon: float = 1.5          # primitive/polygon recognition tolerance (px)
    max_error: float = 1.0        # Bézier fit tolerance (px)
    cubic_paths: bool = False     # fit curved runs with cubic (vs quadratic) Béziers;
                                  # off by default — cubics chase raster-staircase noise.
    aa_contours: bool = False     # trace geometry from the sub-pixel coverage field instead
                                  # of the binary mask; off by default — coverage isn't
                                  # left-right symmetric, so it drifts flat symmetric regions.
    max_sampling_colors: int = 16
    min_region_size: int = 16       # absolute pixel-area floor before relative filtering
    min_region_fraction: float = 0.02  # drop regions smaller than this × largest region
    flatten: bool = False
    no_symmetry: bool = False
    corner_radius: float | None = None  # shared fillet radius; None = auto from geometry
    fidelity_tol: float = 0.06        # selector's render-ΔE gate (slice 4a)
    selection: SelectionPolicy | None = None  # manual candidate selection (slice 4b)
    sym_tol: float = 0.10        # mirror-axis detection tolerance (reflection mismatch)
    straddle_iou: float = 0.96   # min self-reflection IoU to force a SINGLE region symmetric
    pair_iou: float = 0.90       # min IoU to treat two regions as a mirror pair
    working_max_dim: int | None = None  # downscale inputs whose longest side exceeds this
                                        # (LANCZOS) before segmentation; None = off (default).
                                        # Set e.g. working_max_dim=768 for noisy AI-raster inputs
                                        # where high resolution amplifies quantization noise.
    refinement: bool = False             # experimental geometry refinement path; default off.
    corner_normalize: bool = False      # canonicalize recognised rounded corners to Q/point.
    refinement_passes: RefinementPasses = dataclasses.field(default_factory=RefinementPasses)


COVERAGE_HOLE_TOL = 0.05   # if >5% of a region's eroded interior would fall below the 0.5
                           # contour level, the K-way soft field disagrees with the mask
                           # (a gradient surface drifting toward a similar palette color) —
                           # keep the binary mask there instead of punching holes.


def attach_coverage_field(
    regions: list[Region],
    rgb: np.ndarray,
    max_sampling_colors: int,
    *,
    palette: np.ndarray | None = None,
) -> None:
    """Attach `region.coverage` from ONE shared soft label field, computed over the given
    regions' colors + background, evaluated on each region's CURRENT mask — so merged and
    reconstructed regions are covered, not just freshly-segmented ones. A region whose field
    coverage would hole its interior (gradient surface) keeps coverage=None (mask contour).

    The soft field is built from ALL image palette colors (not just region colors) so
    gradient-adjacent colors compete as labels — enabling the hole-guard to fire even when
    the competing color belongs to a merged-away region or a non-segmented image area."""
    from .softlabel import soft_label_field, local_region_coverage, coverage_competitors, label_coverage
    from .segment import _background_color
    if not regions:
        return
    palette_cols = (
        np.asarray(palette, dtype=np.uint8)
        if palette is not None
        else extract_palette(rgb, max_sampling_colors=max_sampling_colors)
    )
    q = quantize(rgb, palette_cols)
    bg = _background_color(q)
    # Use ALL image palette colors + background as rows so gradient-competing colors are
    # represented in the field even when they're not (or no longer) separate regions.
    rows: list[tuple[int, int, int]] = [tuple(int(c) for c in col) for col in palette_cols]
    if bg not in rows:
        rows = rows + [bg]
    row_hex = ["#{:02X}{:02X}{:02X}".format(r[0], r[1], r[2]) for r in rows]
    L = soft_label_field(rgb.astype(float), np.array(rows, np.uint8))
    hex_to_idx = {hx: i for i, hx in enumerate(row_hex)}
    by_label: dict[int, list[Region]] = {}
    for r in regions:
        idx = hex_to_idx.get(r.color_hex)
        if idx is None:
            r.coverage = None
            continue
        by_label.setdefault(idx, []).append(r)
    competitors = coverage_competitors(L)
    # Process one color at a time: share its global field without retaining a
    # second full palette-sized cache. Region ordering and masks stay unchanged.
    for idx, members in by_label.items():
        shared = label_coverage(L, idx, competitors=competitors)
        for r in members:
            cov = local_region_coverage(L, idx, r.mask, label_cov=shared)
            if cov is None:
                r.coverage = None
                continue
            x, y = cov.origin
            height, width = cov.values.shape
            local_mask = r.mask[y:y + height, x:x + width]
            interior = ndi.binary_erosion(local_mask, iterations=3)
            if not interior.any():
                interior = local_mask
            r.coverage = None if (cov.values[interior] < 0.5).mean() > COVERAGE_HOLE_TOL else cov


def _segment_image(arr: np.ndarray, opt: Options) -> tuple[int, int, list[Region]]:
    """Quantize + segment an RGB array into flat-color regions, dropping ones too
    small to be intentional.

    The size threshold is taken relative to the *largest region* (a proxy for the
    mark's scale), not the canvas, so it is resolution-independent: padding the
    image or feeding a higher-res copy does not change which regions survive. A
    small absolute floor first removes single-pixel quantization noise."""
    h, w, _ = arr.shape
    palette = extract_palette(arr, max_sampling_colors=opt.max_sampling_colors)
    q = quantize(arr, palette)
    regions = segment(q, min_area=opt.min_region_size)
    if regions:
        cut = opt.min_region_fraction * max(r.area for r in regions)
        regions = [r for r in regions if r.area >= cut]
    if opt.aa_contours:
        attach_coverage_field(regions, arr, opt.max_sampling_colors)
    return w, h, regions


Affine = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class AxisLine:
    """A detected mirror axis as a segment in output-frame (viewBox) coords."""
    x1: float
    y1: float
    x2: float
    y2: float


def _to_json_safe(v):
    """Recursively convert any value to a JSON-serializable form.
    Dataclasses are expanded to dicts; sets are sorted lists; all else -> repr."""
    if isinstance(v, (bool, int, float, str, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_to_json_safe(x) for x in v]
    if isinstance(v, Mapping):
        return {k: _to_json_safe(vv) for k, vv in v.items()}
    if isinstance(v, (set, frozenset)):
        return sorted(_to_json_safe(x) for x in v)
    if dataclasses.is_dataclass(v):
        return {f.name: _to_json_safe(getattr(v, f.name)) for f in dataclasses.fields(v)}
    return repr(v)


def _fill_kind(fill: Fill) -> str:
    """Canonical fill-type label for diagnostics schema."""
    if isinstance(fill, FlatFill):
        return "flat"
    if isinstance(fill, LinearGradientFill):
        return "linear_gradient"
    if isinstance(fill, RadialGradientFill):
        return "radial_gradient"
    if isinstance(fill, RasterFill):
        return "raster"
    return "unknown"


class ReportDiag:
    """Structured JSON-ready diagnostics for one idealize() run.

    Call `.to_dict()` to get the schema dict.  The schema is stable across
    non-breaking pipeline changes and is designed for downstream tooling
    (visualisers, regression dashboards, agent post-processing).

    Schema shape (top level)::

        {
            "options": { ...all Options fields... },
            "stats":   { "regions": N, "components": M, "elements": K,
                         "gradients": G, "axes": A },
            "axes":    [ { "theta": f, "cx": f, "cy": f,
                           "weight": f, "primary": bool }, ... ],
            "regions": [ { "id": label, "area": px, "color_hex": "#..",
                           "bbox": [x0,y0,x1,y1], "options": {...},
                           "symmetry": { "role": "straddler|pair|loner",
                                         "axis": {theta,cx,cy}|null,
                                         "off_ratio": float,
                                         "partner": label|null },
                           "strategies": {
                               "geom": { "<strategy>": {"chosen": bool} },
                               "fill": { "<kind>": {"chosen": bool} } }
                         }, ... ]
        }
    """

    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


@dataclass(frozen=True)
class IdealizeReport:
    """What the pipeline actually emitted for one idealize() run: the histogram of
    fitter strategies the scorer chose per region, the gradient-fill count, the total
    emitted element count, the detected mirror axes (one segment per component
    with a vertical mirror, in output-frame coords), and per-region symmetry
    diagnostics. Diagnostic annotation."""

    strategies: Mapping[str, int]
    gradients: int
    elements: int
    axes: tuple[AxisLine, ...]
    symmetry: tuple  # per-region (label, self_iou, decision) entries from classify_regions
    diagnostics: "ReportDiag | None" = None  # structured JSON-ready diagnostics; None when not built

    @staticmethod
    def empty() -> "IdealizeReport":
        return IdealizeReport(types.MappingProxyType({}), 0, 0, (), ())


def _map_axis(a: AxisLine, affine: Affine) -> AxisLine:
    x1, y1 = apply_affine_point(affine, a.x1, a.y1)
    x2, y2 = apply_affine_point(affine, a.x2, a.y2)
    return AxisLine(x1, y1, x2, y2)


def _build_report(
    cands: list[Candidate],
    axes: list[AxisLine],
    sym_diags: list | None = None,
    *,
    opt: Options | None = None,
    regions: list[Region] | None = None,
    diag_extra: dict | None = None,
) -> IdealizeReport:
    strategies: dict[str, int] = {}
    gradients = 0
    for c in cands:
        if isinstance(c.fill, (LinearGradientFill, RadialGradientFill)):
            gradients += 1
        if c.strategy is not None:                 # None for occlusion / lens
            strategies[c.strategy] = strategies.get(c.strategy, 0) + 1
    diag = _build_diag(cands, axes, sym_diags or [], opt, regions, diag_extra)
    return IdealizeReport(
        types.MappingProxyType(dict(strategies)),
        gradients,
        len(cands),
        tuple(axes),
        tuple(sym_diags) if sym_diags is not None else (),
        diag,
    )


def _build_diag(
    cands: list[Candidate],
    axes: list[AxisLine],
    sym_diags: list,
    opt: Options | None,
    regions: list[Region] | None,
    diag_extra: dict | None,
) -> "ReportDiag | None":
    """Build the structured ReportDiag from pipeline outputs.

    Returns None when opt/regions/diag_extra are not supplied (e.g. from legacy
    call sites). The per-region ``strategies.geom`` dict contains only the chosen
    strategy for a first cut; non-chosen alternatives are a TODO for slice 4c+."""
    if opt is None or regions is None or not diag_extra:
        return None

    # options: every Options field, complex values serialised to JSON-safe form.
    opt_dict = _to_json_safe(opt)

    # stats
    n_components = diag_extra.get("n_components", 0)
    n_gradients = sum(
        1 for c in cands if isinstance(c.fill, (LinearGradientFill, RadialGradientFill))
    )
    stats: dict = {
        "regions": len(regions),
        "components": n_components,
        "elements": len(cands),
        "gradients": n_gradients,
        "axes": len(axes),
    }

    # per-candidate lookup by region_label (first encountered = chosen winner)
    cand_by_label: dict[int, Candidate] = {}
    for c in cands:
        if c.source == "region" and c.region_label is not None:
            cand_by_label.setdefault(c.region_label, c)

    # sym_diags lookup
    role_by_label: dict[int, str] = {}
    off_by_label: dict[int, float] = {}
    for lbl, off, role in sym_diags:
        role_by_label[int(lbl)] = str(role)
        off_by_label[int(lbl)] = float(off)

    region_axis_map: dict[int, Axis2D | None] = diag_extra.get("region_axis", {})
    region_partner_map: dict[int, int | None] = diag_extra.get("region_partner", {})

    # per-region dicts
    region_dicts: list[dict] = []
    for r in regions:
        lbl = r.label
        ys, xs = np.nonzero(r.mask)
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            if xs.size else [0, 0, 0, 0]
        )

        role = role_by_label.get(lbl, "loner")
        off_ratio = off_by_label.get(lbl, 0.0)
        ax: Axis2D | None = region_axis_map.get(lbl)
        partner: int | None = region_partner_map.get(lbl)

        ax_dict = (
            {"theta": float(ax.theta), "cx": float(ax.cx), "cy": float(ax.cy)}
            if ax is not None else None
        )

        # strategies — first cut: chosen only; TODO: include non-chosen alternatives
        chosen_cand = cand_by_label.get(lbl)
        geom_strats: dict = {}
        fill_strats: dict = {}
        if chosen_cand is not None:
            geom_k = chosen_cand.strategy or "unknown"
            geom_strats[geom_k] = {"chosen": True}  # TODO(slice-4c): add non-chosen with chosen=False
            fill_k = _fill_kind(chosen_cand.fill)
            fill_strats[fill_k] = {"chosen": True}

        region_dicts.append({
            "id": lbl,
            "area": int(r.area),
            "color_hex": r.color_hex,
            "bbox": bbox,
            "options": opt_dict,
            "symmetry": {
                "role": role,
                "axis": ax_dict,
                "off_ratio": off_ratio,
                "partner": partner,
            },
            "strategies": {
                "geom": geom_strats,
                "fill": fill_strats,
            },
        })

    data: dict = {
        "options": opt_dict,
        "stats": stats,
        "axes": diag_extra.get("axis_info", []),
        "regions": region_dicts,
    }
    return ReportDiag(data)


def build_candidates(
    reconstructed: list, straddlers: list[Region], pairs: list[tuple[Region, Region]],
    loners: list[Region], fills: dict[int, Fill],
    opt: Options, axis: Axis | None,
    source_rgb: np.ndarray | None, *, base: int = 0,
) -> list[Candidate]:
    """Decide geometry + fill per element and return the candidate list in exact
    paint order: occlusion (by z) -> regions (by area desc). Elements whose geometry
    fit returns None are dropped (matching the old per-loop `continue`, so the
    emit-time id sequence is unchanged). A region absent from `fills` falls back to
    FlatFill(region.color_hex)."""
    cands: list[Candidate] = []

    for elem in sorted(
        reconstructed,
        key=lambda e: e.z if isinstance(e, ScenePrimitive) else e.params["z"],
    ):
        if isinstance(elem, ScenePrimitive):
            cands.append(Candidate(Shape(elem.kind, dict(elem.params)),
                                   FlatFill(elem.color_hex), "occlusion"))
        else:  # lens Shape("path", {"d", "color_hex", "z"})
            cands.append(Candidate(Shape("path", {"d": elem.params["d"]}),
                                   FlatFill(elem.params["color_hex"]), "lens"))

    drawn = (
        [(r, axis, False) for r in straddlers]
        + [(canon, None, True) for canon, _ in pairs]
        + [(r, None, False) for r in loners]
    )
    drawn.sort(key=lambda rp: rp[0].area, reverse=True)
    for region, fit_axis, is_pair in drawn:
        # eid = sN where N = base + current cands length. The occlusion/lens loop already
        # filled cands[0..]; a None-return below skips the append, so only emitted
        # elements consume an id — exactly matching the SVG emit-loop's GLOBAL id sequence
        # (base = candidates from prior components, so per-component lookups address sN).
        eid = f"s{base + len(cands)}"
        element = opt.selection.for_id(eid) if opt.selection is not None else None
        cr = opt.corner_radius if opt.corner_radius is not None else region_corner_radius(region.mask, coverage=region.coverage)
        shape, strategy = select_geometry(region, opt, fit_axis, cr, source_rgb,
                                          element=element, eid=eid)
        if shape is None:
            continue
        fill = fills.get(region.label, FlatFill(region.color_hex))
        cands.append(Candidate(shape, fill, "region",
                               mirror=axis if is_pair else None, strategy=strategy,
                               region_label=region.label))

    return cands


def _render_body(
    w: int, h: int, regions: list[Region], opt: Options, *,
    bake: Affine | None = None, rgb: np.ndarray | None = None,
) -> tuple[list[str], list[str], list[Candidate], list[AxisLine], list, dict]:
    """Decompose regions into gutter-separated components, then per component detect
    symmetry, reconstruct occlusion, and fit regions — accumulating candidates that the
    single emit loop renders in z-order with globally continuous sN ids. Operates
    entirely in the frame of `regions` (which may be a rectified frame) so the caller
    can wrap the result in an inverse transform.

    When `bake` is given (only in flatten mode), the inverse transform is applied
    directly to each path's coordinates instead — flatten emits pure baked geometry
    with no wrapping `<g transform>`.

    Returns a 6-tuple: ``(body, defs, cands, frame_axes, sym_diags, diag_extra)``
    where ``diag_extra`` is a dict carrying structured diagnostics data
    (``n_components``, ``axis_info``, ``region_axis``, ``region_partner``)
    for use by ``_build_report``."""
    components = decompose_components(regions, (h, w))
    defs: list[str] = []
    cands: list[Candidate] = []
    frame_axes: list[AxisLine] = []
    sym_diags: list = []
    # Diagnostics extras — populated per component then merged into one dict.
    _diag_axis_info: list[dict] = []       # one entry per detected group primary axis
    _diag_region_axis: dict[int, Axis2D | None] = {}   # label -> primary Axis2D
    _diag_region_partner: dict[int, int | None] = {}   # label -> mirror partner label

    for comp in components:
        silhouette = np.any([r.mask for r in comp], axis=0)

        # Per-component region-level symmetry detection.  Calling detect_symmetry_groups
        # on comp (rather than all regions) avoids the cross-component bias where equal-
        # height regions in DIFFERENT components form a dominant horizontal axis that
        # outweighs the vertical pair axis within a single component.
        # For single-component images (e.g. a full-bleed radish+wordmark) this is
        # identical to a global call, so the radish body's vertical axis still wins.
        _comp_groups = [] if opt.no_symmetry else detect_symmetry_groups(
            comp,
            straddle_iou=opt.straddle_iou,
            pair_iou=opt.pair_iou,
        )
        _comp_region_axis: dict[int, Axis2D | None] = {}
        _comp_region_role: dict[int, str] = {}
        _comp_pair_partner: dict[int, int] = {}
        _SEG = 50.0
        _comp_ys = np.nonzero(silhouette)[0]
        _gy_min = float(_comp_ys.min()) if _comp_ys.size else 0.0
        _gy_max = float(_comp_ys.max()) if _comp_ys.size else float(h - 1)
        # Pick the dominant symmetric group: the group with the largest total pixel area
        # of its straddlers + pair members.  Only this group drives reconstruct_scene —
        # folding ALL detected groups about one axis mixes per-component axes and
        # mangles secondary shapes (e.g. text glyphs folded about the radish body axis).
        # Regions belonging to non-dominant groups are treated as loners and fit
        # faithfully.  Groups whose straddlers+pairs are empty (all-loner groups)
        # are excluded from contention.
        def _sym_area(g) -> int:
            return (sum(r.area for r in g.straddlers)
                    + sum(a.area + b.area for a, b in g.pairs))

        _dominant_group = max(
            (
                _g for _g in _comp_groups
                if _sym_area(_g) > 0
                and _g.axes
                and abs(_g.axes[0].theta - np.pi / 2) < 0.05
            ),
            key=_sym_area,
            default=None,
        )
        _primary: Axis2D | None = (
            _dominant_group.axes[0]
            if _dominant_group is not None and _dominant_group.axes
            else None
        )

        # Populate role/axis lookups from dominant group only.
        # Regions not in the dominant group implicitly stay as loners
        # (absent from _comp_region_role → fallback "loner" later).
        if _dominant_group is not None:
            for _r in _dominant_group.straddlers:
                _comp_region_axis[_r.label] = _primary
                _comp_region_role[_r.label] = "straddler"
            for _a, _b in _dominant_group.pairs:
                _comp_region_axis[_a.label] = _primary
                _comp_region_axis[_b.label] = _primary
                _comp_region_role[_a.label] = "pair"
                _comp_region_role[_b.label] = "pair"
                _comp_pair_partner[_a.label] = _b.label
                _comp_pair_partner[_b.label] = _a.label
            for _r in _dominant_group.loners:
                _comp_region_role[_r.label] = "loner"

        # Emit one AxisLine per component (dominant group primary only).
        # A disk has 12 candidate axes; only the highest-weight primary is meaningful.
        if _primary is not None:
            if abs(_primary.theta - np.pi / 2) < 0.05:
                frame_axes.append(AxisLine(float(_primary.cx), _gy_min, float(_primary.cx), _gy_max))
            else:
                _dxax, _dyax = np.cos(_primary.theta), np.sin(_primary.theta)
                frame_axes.append(AxisLine(
                    _primary.cx - _SEG * _dxax, _primary.cy - _SEG * _dyax,
                    _primary.cx + _SEG * _dxax, _primary.cy + _SEG * _dyax,
                ))
            # Approx weight = total pixel area of dominant group members.
            _group_weight = float(
                sum(r.area for r in _dominant_group.straddlers)
                + sum(a.area + b.area for a, b in _dominant_group.pairs)
                + sum(r.area for r in _dominant_group.loners)
            )
            _diag_axis_info.append({
                "theta": float(_primary.theta),
                "cx": float(_primary.cx),
                "cy": float(_primary.cy),
                "weight": _group_weight,
                "primary": True,
            })

        # Diagnostics: axis/partner for dominant group members; non-dominant → None.
        if _dominant_group is not None:
            for _r in _dominant_group.straddlers:
                _diag_region_axis[_r.label] = _primary
                _diag_region_partner[_r.label] = None
            for _a, _b in _dominant_group.pairs:
                _diag_region_axis[_a.label] = _primary
                _diag_region_axis[_b.label] = _primary
                _diag_region_partner[_a.label] = _b.label
                _diag_region_partner[_b.label] = _a.label
            for _r in _dominant_group.loners:
                _diag_region_axis[_r.label] = None
                _diag_region_partner[_r.label] = None

        # Derive vertical axis for reconstruct_scene (vertical-only today).
        # Use the first comp region that belongs to a near-vertical group primary.
        _comp_primary_ax: Axis2D | None = next(
            (_comp_region_axis[r.label] for r in comp
             if r.label in _comp_region_axis and _comp_region_axis[r.label] is not None),
            None,
        )
        if _comp_primary_ax is not None and abs(_comp_primary_ax.theta - np.pi / 2) < 0.05:
            axis: Axis | None = Axis(x=_comp_primary_ax.cx)
        else:
            axis = None
        reconstructed, comp = reconstruct_scene(comp, axis, (h, w))

        # Seam-merge adjacent regions into surfaces, then fit fill per merged surface.
        # This 2-pass drives only path A (seam_is_soft): flat fills are passed into
        # merge_surfaces so none of the surfaces entering the merge loop carry gradient
        # fills, and gradients_continuous (path B) therefore returns False for every pair.
        # Path B is intentionally preserved in merge_surfaces and is covered by its own
        # unit tests, but is NOT exercised end-to-end from here because intermediate
        # gradient-type mismatches (radial vs linear on partial merges) made per-region
        # path-B merges unreliable.
        # TODO(follow-up): revisit B-primary once per-region gradient-kind is stable.
        if rgb is not None:
            flat_filled = [(r, FlatFill(r.color_hex)) for r in comp]
            merged = merge_surfaces(flat_filled, rgb)
            comp = [r for r, _ in merged]
            fills = {r.label: fit_fill(r.mask, rgb, flat_hex=r.color_hex).fill for r, _ in merged}
        else:
            fills = {}

        # Straddlers/pairs/loners from per-component lookup.
        straddlers = [r for r in comp if _comp_region_role.get(r.label) == "straddler"]
        _comp_labels = {r.label for r in comp}
        _pair_by_lbl = {r.label: r for r in comp if _comp_region_role.get(r.label) == "pair"}
        _seen_pairs: set[int] = set()
        pairs = []
        for _lbl in sorted(_pair_by_lbl):
            if _lbl in _seen_pairs:
                continue
            _partner = _comp_pair_partner.get(_lbl)
            if _partner is not None and _partner in _comp_labels and _partner in _pair_by_lbl:
                pairs.append((_pair_by_lbl[_lbl], _pair_by_lbl[_partner]))
                _seen_pairs |= {_lbl, _partner}
        loners = [r for r in comp if _comp_region_role.get(r.label, "loner") not in ("straddler", "pair")]
        for r in comp:
            _role = _comp_region_role.get(r.label, "loner")
            # Record the actual off/peri ratio so over-fires are visible in diagnostics.
            # For loners there is no assigned axis so we fall back to 0.0.
            _ax4diag = _comp_region_axis.get(r.label)
            _score = sym_off_ratio(r.mask, _ax4diag) if _ax4diag is not None else 0.0
            sym_diags.append((r.label, _score, _role))

        cands += build_candidates(
            reconstructed, straddlers, pairs, loners, fills, opt, axis, rgb,
            base=len(cands),
        )

    def emit(d: str, fill: str, rule: str | None = None) -> str:
        return path_svg(transform_path_d(d, bake) if bake is not None else d, fill, rule)

    def _fill_attr(fill: Fill) -> str:
        if isinstance(fill, RasterFill):
            # userSpaceOnUse pattern: map to the baked frame via patternTransform
            # (bake is set only in flatten mode; None otherwise -> absolute coords).
            return resolve_fill(fill, defs, transform=bake)
        baked = None
        if not isinstance(fill, FlatFill) and bake is not None:
            kind = "linear" if isinstance(fill, LinearGradientFill) else "radial"
            baked = _bake_gradient_geometry(fill.geometry, kind, bake)
        return resolve_fill(fill, defs, geometry=baked)

    body: list[str] = []
    eid = 0
    for cand in cands:
        geom = cand.geometry
        fill = _fill_attr(cand.fill)
        if opt.flatten:
            d = shape_to_path_d(geom)
            rule = fill_rule_for(geom)
            body.append(emit(d, fill, rule))
            if cand.mirror is not None:
                body.append(emit(reflect_path_d(d, cand.mirror.x), fill, rule))
        elif cand.source == "lens":
            body.append(emit(geom.params["d"], fill))
        else:
            elem_id = f"s{eid}"
            body.append(shape_to_svg(geom, fill, elem_id))
            if cand.mirror is not None:
                body.append(mirror_use(elem_id, cand.mirror))
        eid += 1

    diag_extra = {
        "n_components": len(components),
        "axis_info": _diag_axis_info,
        "region_axis": _diag_region_axis,
        "region_partner": _diag_region_partner,
    }
    return body, defs, cands, frame_axes, sym_diags, diag_extra


def _rectify_affine(rho: float, w0: int, h0: int, rw: int, rh: int) -> Affine:
    """The SVG affine (a, b, c, d, e, f) that maps a point in the rectified frame
    back to the original: translate(-rw/2,-rh/2) → rotate(-rho) → translate(w0/2,h0/2)."""
    th = np.radians(-rho)
    cos, sin = np.cos(th), np.sin(th)
    rot = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    to_centre = np.array([[1.0, 0.0, w0 / 2], [0.0, 1.0, h0 / 2], [0.0, 0.0, 1.0]])
    from_centre = np.array([[1.0, 0.0, -rw / 2], [0.0, 1.0, -rh / 2], [0.0, 0.0, 1.0]])
    m = to_centre @ rot @ from_centre
    return (m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2])


def _bake_gradient_geometry(geom: dict, kind: str, bake: Affine) -> dict:
    """Map gradient geometry from the rectified frame to the original via the bake affine
    (a, b, c, d, e, f): x' = a*x + c*y + e, y' = b*x + d*y + f. The rectify affine is a rigid
    rotation+translation, so the radial radius is preserved."""
    if kind == "linear":
        x1, y1 = apply_affine_point(bake, geom["x1"], geom["y1"])
        x2, y2 = apply_affine_point(bake, geom["x2"], geom["y2"])
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    cx, cy = apply_affine_point(bake, geom["cx"], geom["cy"])
    return {"cx": cx, "cy": cy, "r": geom["r"]}


def _idealize_rectified(arr: np.ndarray, opt: Options, rho: float, w0: int, h0: int) -> tuple[str | None, list[Candidate], list[AxisLine], list, dict]:
    """Rotate the image so the tilted mirror axis is vertical and idealize there.
    Non-flatten output keeps the symmetry (`<use>` mirror about the vertical axis)
    and is wrapped in one inverse-rotation `<g>`; flatten output bakes that same
    rotation into the path coordinates so no transform survives. Returns (None, [], [], [], {})
    (so the caller falls back to upright) if the rectified frame yields no usable
    regions or its vertical symmetry no longer registers."""
    rot = ndi.rotate(arr.astype(float), -rho, reshape=True, order=1, cval=255.0)
    rot = np.clip(rot, 0.0, 255.0).astype(np.uint8)
    rw, rh, regions = _segment_image(rot, opt)
    if not regions:
        return None, [], [], [], {}
    if detect_axis(np.any([r.mask for r in regions], axis=0), tol=opt.sym_tol) is None:
        return None, [], [], [], {}
    affine = _rectify_affine(rho, w0, h0, rw, rh)
    bake = affine if opt.flatten else None
    body, defs, cands, frame_axes, sym_diags, diag_extra = _render_body(rw, rh, regions, opt, bake=bake, rgb=rot)
    # Bail unless the rectified frame actually exploited mirror symmetry (some region
    # classified as a straddler or pair). A tilted silhouette can register a vertical
    # axis yet have every region fall below the straddle gate (e.g. telegram's lone
    # paper-plane at IoU 0.917 < 0.96, classified a loner): then the rotated re-fit only
    # adds resampling churn versus the upright idealization, so fall back to upright
    # instead of committing a distorted result.
    if not any(decision in ("straddler", "pair") for _, _, decision in sym_diags):
        return None, [], [], [], {}
    if opt.flatten:
        doc = render_svg_doc(w0, h0, body, defs)
    else:
        wrap = (f'<g transform="translate({_fmt(w0 / 2)} {_fmt(h0 / 2)}) '
                f'rotate({_fmt(round(-rho, 3))}) translate({_fmt(-rw / 2)} {_fmt(-rh / 2)})">')
        doc = render_svg_doc(w0, h0, [wrap, *body, "</g>"], defs)
    axes = [_map_axis(a, affine) for a in frame_axes]
    return doc, cands, axes, sym_diags, diag_extra


def _render_refinement_body(
    objects,
    *,
    flatten: bool = False,
    epsilon: float = 1.0,
    max_error: float = 1.0,
    cubic: bool = False,
) -> tuple[list[str], list[str]]:
    """Serialize optimized objects, resolving object-id based <use> references."""
    from .refinement.region_sample import RegionSample
    from .refinement.symmetry_materialize import index_region_paths, resolve_source_path

    defs: list[str] = []
    fill_cache: dict[str, str] = {}
    body: list[str] = []

    def _child_label(parent_label: str, child: RegionSample) -> str:
        return f"{parent_label}-{int(child.id)}"

    def _resolved_fill(fill_obj: Fill) -> str:
        key = repr(fill_obj)
        if key not in fill_cache:
            fill_cache[key] = resolve_fill(fill_obj, defs)
        return fill_cache[key]

    def _inlined_use_shape(source_shape: Shape, transform) -> Shape:
        params = {
            "d": shape_to_path_d(source_shape),
            "transform": transform,
        }
        rule = fill_rule_for(source_shape)
        if rule is not None:
            params["fill_rule"] = rule
        return Shape("use", params)


    def _render_items(regions: list[RegionSample]):
        items = []

        def visit(region: RegionSample, label: str, scope_ids: dict[int, str], scope_objects: dict[int, RegionSample], path: tuple[int, ...]) -> None:
            symmetry = region.diagnostics.get("symmetry")
            if isinstance(symmetry, Mapping) and not symmetry.get('materialized') and (
                (region.is_branch and symmetry.get("mode") == "self")
                or symmetry.get("mode") == "self_mirror"
            ):
                raise ValueError("self-symmetry must be materialized before rendering")
            if region.is_leaf:
                items.append((region, label, scope_ids, scope_objects, path))
                return
            children = sorted(region.children, key=lambda child: (float(child.z), int(child.id)))
            child_scope_ids = {int(child.id): _child_label(label, child) for child in children}
            child_scope_objects = {int(child.id): child for child in children}
            for child in children:
                visit(child, child_scope_ids[int(child.id)], child_scope_ids, child_scope_objects, (*path, child.id))

        top = sorted(regions, key=lambda obj: (float(obj.z), int(obj.id)))
        top_scope_ids = {int(obj.id): f"s{int(obj.id)}" for obj in top}
        top_scope_objects = {int(obj.id): obj for obj in top}
        for obj in top:
            visit(obj, top_scope_ids[int(obj.id)], top_scope_ids, top_scope_objects, (obj.id,))
        return sorted(items, key=lambda item: (float(item[0].z), item[1]))

    objects = list(objects)
    located_items = _render_items(objects)
    reference_index = index_region_paths(objects)
    labels = {path: label for _, label, _, _, path in located_items}
    render_items = []
    for obj, label, scope_ids, scope_objects, path in located_items:
        if obj.current is not None and obj.current.kind == "use":
            source_path = resolve_source_path(reference_index, path, obj.current.params)
            if source_path not in labels:
                raise ValueError("clone source must be retained concrete geometry")
            source_id = int(obj.current.params["href_obj_id"])
            # Lookup only: keep the retained IDs and existing emitted labels.
            scope_ids = {**scope_ids, source_id: labels[source_path]}
            scope_objects = {**scope_objects, source_id: reference_index[source_path]}
        render_items.append((obj, label, scope_ids, scope_objects))
    # A clone can retain geometry while using a different fill.  Keep the
    # shared shape unpainted in ``defs`` and paint every visible instance via
    # ``<use>``; otherwise the source element's explicit ``fill`` wins over a
    # clone instance's fill attribute.
    clone_definition_ids: dict[tuple[int, str], str] = {}
    clone_sources: dict[tuple[int, str], RegionSample] = {}
    for obj, _elem_id, scope_ids, scope_objects in render_items:
        if obj.current is None or obj.current.kind != "use":
            continue
        if not any(isinstance(obj.diagnostics.get(key), Mapping) for key in ("clone", "clones")):
            continue
        source_id = obj.current.params.get("href_obj_id")
        if not isinstance(source_id, int):
            continue
        source = scope_objects.get(source_id)
        source_label = scope_ids.get(source_id)
        if source is None or source_label is None or source.current is None or source.current.kind == "use":
            continue
        key = (id(source), source_label)
        clone_sources[key] = source
        clone_definition_ids[key] = f"{source_label}-geometry"

    for key, source in clone_sources.items():
        source_shape = source.current
        assert source_shape is not None
        defs.append(shape_to_svg(source_shape, None, clone_definition_ids[key]))

    for obj, elem_id, scope_ids, scope_objects in render_items:
        assert obj.current is not None
        assert obj.fill is not None
        fill = _resolved_fill(obj.fill)
        shape = resolve_use_shape(obj.current, scope_ids)
        if flatten:
            if shape.kind == "use":
                source = scope_objects[int(obj.current.params["href_obj_id"])]
                shape = _inlined_use_shape(source.current, shape.params["transform"])
                fill = str(obj.current.params.get("fill", fill))
            body.append(path_svg(shape_to_path_d(shape), fill, fill_rule_for(shape)))
            continue

        source_definition_id = clone_definition_ids.get((id(obj), elem_id))
        if source_definition_id is not None:
            source_shape = Shape("use", {"href": source_definition_id, "transform": (1, 0, 0, 1, 0, 0)})
            body.append(shape_to_svg(source_shape, fill, elem_id))
            continue

        if shape.kind == "use":
            source = scope_objects[int(obj.current.params["href_obj_id"])]
            assert source.fill is not None
            use_fill = str(obj.current.params.get("fill", fill))
            if any(isinstance(obj.diagnostics.get(key), Mapping) for key in ("clone", "clones")):
                source_label = scope_ids[int(obj.current.params["href_obj_id"])]
                definition_id = clone_definition_ids.get((id(source), source_label))
                if definition_id is not None:
                    params = dict(shape.params)
                    params["href"] = definition_id
                    shape = Shape("use", params)
                body.append(shape_to_svg(shape, fill, elem_id))
                continue
            shape = _inlined_use_shape(source.current, shape.params["transform"])
            body.append(path_svg(shape_to_path_d(shape), use_fill, fill_rule_for(shape)))
            continue

        body.append(shape_to_svg(shape, fill, elem_id))
    return body, defs


def _refinement_passes(opt: Options):
    """Pass order for the experimental geometry refinement."""
    from .refinement.passes import (
        clones_pass,
        corner_normalize_pass,
        occlusion_pass,
        primitives_pass,
        seams_pass,
        simplify_pass,
        smooth_pass,
        split_compound_pass,
        symmetry_pass,
    )

    def _configured_primitives_pass(objects, masks):
        return primitives_pass(objects, masks, epsilon=opt.epsilon)

    def _configured_occlusion_pass(objects, masks):
        return occlusion_pass(objects, masks, no_symmetry=opt.no_symmetry)

    def _configured_symmetry_pass(objects, masks):
        return symmetry_pass(
            objects,
            masks,
            epsilon=opt.epsilon,
            max_error=opt.max_error,
            cubic=opt.cubic_paths,
        )

    def _configured_seams_pass(objects, masks):
        return seams_pass(
            objects,
            masks,
            epsilon=opt.epsilon,
            max_error=opt.max_error,
            cubic=opt.cubic_paths,
            # Earlier passes can independently refit neighbouring leaves.  A
            # final full-scene stitch is therefore required to restore their
            # shared edge after primitive fitting, straightening, smoothing,
            # or symmetry.  ``post_symmetry=True`` only repaired a new
            # self-mirror axis and left ordinary material seams drifting.
            post_symmetry=False,
        )

    _configured_primitives_pass.__name__ = "primitives_pass"
    _configured_occlusion_pass.__name__ = "occlusion_pass"
    _configured_symmetry_pass.__name__ = "symmetry_pass"
    _configured_seams_pass.__name__ = "seams_pass"
    # Establish semantic surfaces before assigning a primitive.  In particular,
    # a compound path can hide a simple exterior behind one or more counters;
    # fitting it both before and after splitting gives the same trace-owned
    # region two competing primitive interpretations.
    def _configured_smooth_pass(objects, masks):
        return smooth_pass(objects, masks, max_error=opt.max_error)

    _configured_smooth_pass.__name__ = "smooth_pass"

    def _configured_linelet_simplify_pass(objects, masks):
        return simplify_pass(
            objects,
            masks,
            epsilon=opt.epsilon,
            max_error=opt.max_error,
            cubic=opt.cubic_paths,
            linelet_only=True,
            max_boundary_error=opt.max_error,
        )

    _configured_linelet_simplify_pass.__name__ = "linelet_simplify_pass"

    def _configured_corner_normalize_pass(objects, masks):
        return corner_normalize_pass(
            objects,
            masks,
            max_error=opt.max_error,
            enabled=opt.corner_normalize,
        )

    _configured_corner_normalize_pass.__name__ = "corner_normalize_pass"
    # Congruence is the more general relationship between separate regions:
    # rotation and reflection are both valid clone transforms. Establish clone
    # families before symmetry can independently rewrite their members or
    # claim them as mirrored pairs. Self-symmetry then applies only to each
    # remaining canonical geometry.
    # A compound trace mixes an exterior with its counters.  It was split
    # above so automatic symmetry only sees addressable semantic leaves. The
    # symmetry pass explicitly excludes compound implementation children, so
    # a cutout is never reflected as an independent shape.
    # A corner normalizer keeps all endpoints fixed, so it cannot reopen a
    # seam, but it can still change the curve between them. Run it before the
    # terminal stitch so neighbouring children finish with one shared path.
    from .refinement_settings import refinement_pass_names
    from .drawing_bands import bands_pass
    configured = {
        'occlusion_pass': _configured_occlusion_pass,
        'split_compound_pass': split_compound_pass,
        'primitives_pass': _configured_primitives_pass,
        'clones_pass': clones_pass,
        'symmetry_pass': _configured_symmetry_pass,
        'linelet_simplify_pass': _configured_linelet_simplify_pass,
        'smooth_pass': _configured_smooth_pass,
        'corner_normalize_pass': _configured_corner_normalize_pass,
        'bands_pass': bands_pass,
        'seams_pass': _configured_seams_pass,
    }
    return [configured[name] for name in refinement_pass_names(opt.refinement_passes, no_symmetry=opt.no_symmetry)]


_SVG_ELEMENT_RE = re.compile(r"<(path|circle|ellipse|rect|polygon|use)\b")
_SVG_PATH_D_RE = re.compile(r"<path\b[^>]*\bd=\"([^\"]*)\"")
_SVG_PATH_COMMAND_RE = re.compile(r"[MLQCAZ]")


def _svg_element_count(svg: str) -> int:
    return len(_SVG_ELEMENT_RE.findall(svg))


def _svg_path_segment_count(svg: str) -> int:
    return sum(len(_SVG_PATH_COMMAND_RE.findall(d)) for d in _SVG_PATH_D_RE.findall(svg))


def _prefer_refinement_svg(trace_svg: str, optimized_svg: str) -> bool:
    """Use refinement output; structural counts are diagnostics, not gates."""
    del trace_svg, optimized_svg
    return True


def _refinement_report(objects, opt: Options, *, fallback_reason: str | None = None) -> IdealizeReport:
    from .refinement.region_sample import leaves

    strategies: dict[str, int] = {}
    gradients = 0
    region_diags = []
    object_diags = []
    flat_objects = leaves(objects)
    for region in sorted(objects, key=lambda item: (float(item.z), int(item.id))):
        region_diags.append({
            "id": int(region.id),
            "z": float(region.z),
            "kind": "leaf" if region.is_leaf else "branch",
            "children": len(region.children),
            "diagnostics": _to_json_safe(dict(region.diagnostics)),
        })
    for obj in sorted(flat_objects, key=lambda item: (float(item.z), int(item.id))):
        assert obj.current is not None
        strategy = f"refinement_{obj.current.kind}"
        strategies[strategy] = strategies.get(strategy, 0) + 1
        if isinstance(obj.fill, (LinearGradientFill, RadialGradientFill)):
            gradients += 1
        bounds = tuple(float(v) for v in getattr(obj.footprint, "bounds", (0.0, 0.0, 0.0, 0.0)))
        object_diags.append({
            "id": int(obj.id),
            "z": float(obj.z),
            "shape": obj.current.kind,
            "fill": _fill_kind(obj.fill) if obj.fill is not None else "none",
            "bounds": bounds,
            "area": float(getattr(obj.footprint, "area", 0.0)),
            "diagnostics": _to_json_safe(dict(obj.diagnostics)),
        })
    diag = ReportDiag({
        "options": _to_json_safe(opt),
        "stats": {
            "regions": 0,
            "components": 0,
            "elements": len(flat_objects),
            "gradients": gradients,
            "axes": 0,
        },
        "axes": [],
        "regions": [],
        "refinement_regions": region_diags,
        "refinement_objects": object_diags,
        "refinement_fallback": fallback_reason,
    })
    return IdealizeReport(
        types.MappingProxyType(dict(strategies)),
        gradients,
        len(flat_objects),
        (),
        (),
        diag,
    )


def _idealize_refinement(arr: np.ndarray, opt: Options, width: int, height: int) -> tuple[str, IdealizeReport]:
    """Experimental trace-vectorize + geometry-refinement pipeline."""
    from .refinement.trace import trace_regions
    from .drawing import Canvas
    from .refinement.replay_binding import capture_trace_replay_state, working_regions_from_state
    from .drawing_refinement import refine_canonical_drawing

    objects, masks = trace_regions(arr, opt)
    state = capture_trace_replay_state(objects, Canvas(width, height))
    selected = refine_canonical_drawing(state, epsilon=opt.epsilon, max_error=opt.max_error,
        cubic=opt.cubic_paths, no_symmetry=opt.no_symmetry,
        corner_normalize=opt.corner_normalize, refinement_passes=opt.refinement_passes)
    optimized = working_regions_from_state(selected.state)
    trace_body, trace_defs = _render_refinement_body(
        objects,
        flatten=opt.flatten,
        epsilon=opt.epsilon,
        max_error=opt.max_error,
        cubic=opt.cubic_paths,
    )
    trace_svg = render_svg_doc(width, height, trace_body, trace_defs)
    optimized_body, optimized_defs = _render_refinement_body(
        optimized,
        flatten=opt.flatten,
        epsilon=opt.epsilon,
        max_error=opt.max_error,
        cubic=opt.cubic_paths,
    )
    optimized_svg = render_svg_doc(width, height, optimized_body, optimized_defs)
    fallback_reason = None
    if not _prefer_refinement_svg(trace_svg, optimized_svg):
        fallback_reason = "optimized output has more path segments than trace baseline"
    report = _refinement_report(optimized, opt, fallback_reason=fallback_reason)
    if opt.refinement_passes.bands:
        data = report.diagnostics.to_dict()
        report = replace(report, diagnostics=ReportDiag({**data, 'refinement_passes': {
            name: dict(stage.diagnostics) for name, stage in selected.passes if stage.diagnostics}}))
    return optimized_svg, report


def _flatten_on_white(im: Image.Image) -> np.ndarray:
    """RGB (H,W,3) uint8 with any alpha composited onto WHITE (a transparent surround is
    background, not a mark). PIL's plain `convert("RGB")` instead DROPS alpha, keeping each
    pixel's stored RGB — so transparent icon backgrounds (typically stored black) become a
    black region, semi-transparent edges keep over-saturated colours, and a spurious white
    anti-aliasing ring appears: mangling the most common logo input."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba)
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _condition_input(arr: np.ndarray, working_max_dim: int | None) -> np.ndarray:
    """Downscale an oversized RGB array to a working resolution before segmentation, so
    input noise stops fragmenting at high pixel counts. Longest side -> working_max_dim,
    aspect-preserving, LANCZOS. Returns arr unchanged when disabled or already small.
    Downscale only — never upscale, never denoise."""
    if working_max_dim is None:
        return arr
    h, w = arr.shape[:2]
    if max(h, w) <= working_max_dim:
        return arr
    scale = working_max_dim / max(h, w)
    img = Image.fromarray(arr).resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _set_svg_output_size(svg: str, width: int, height: int) -> str:
    """Rewrite the <svg> width/height to the original input size, leaving viewBox (working
    space) intact — a pure display scale (SVG is resolution-free)."""
    return re.sub(r'(<svg\b[^>]*?)\bwidth="\d+"\s+height="\d+"',
                  rf'\1width="{width}" height="{height}"', svg, count=1)


def idealize(image, *, options: Options | None = None, report: bool = False) -> str | tuple[str, IdealizeReport]:
    """Idealize a raster mark into SVG. With `report=True`, returns
    `(svg, IdealizeReport)`; otherwise returns the SVG string (back-compatible)."""
    opt = options or Options()
    if isinstance(image, str):
        with Image.open(image) as im:
            arr = _flatten_on_white(im)
    else:
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:            # RGBA array -> composite on white
            arr = _flatten_on_white(Image.fromarray(arr, "RGBA"))
    orig_h, orig_w = arr.shape[:2]
    arr = _condition_input(arr, opt.working_max_dim)
    h0, w0 = arr.shape[:2]

    if opt.refinement:
        svg, opt_report = _idealize_refinement(arr, opt, w0, h0)
        if (arr.shape[1], arr.shape[0]) != (orig_w, orig_h):
            svg = _set_svg_output_size(svg, orig_w, orig_h)
        return (svg, opt_report) if report else svg

    w, h, regions = _segment_image(arr, opt)
    if not regions:
        svg, cands, axes, sym_diags, diag_extra = render_svg_doc(w, h, []), [], [], [], {}
    else:
        svg, cands, axes, sym_diags, diag_extra = None, [], [], [], {}
        # Any-axis symmetry: rectify a tilted mirror upright, idealize there, wrap back.
        if not opt.no_symmetry:
            silhouette = np.any([r.mask for r in regions], axis=0)
            if detect_axis(silhouette, tol=opt.sym_tol) is None:
                rho = detect_symmetry_rotation(silhouette, tol=opt.sym_tol)
                if rho is not None:
                    rectified, rcands, raxes, rsym_diags, rdiag_extra = _idealize_rectified(arr, opt, rho, w0, h0)
                    if rectified is not None:
                        svg, cands, axes, sym_diags, diag_extra = rectified, rcands, raxes, rsym_diags, rdiag_extra
        if svg is None:
            body, defs, cands, axes, sym_diags, diag_extra = _render_body(w, h, regions, opt, rgb=arr)
            svg = render_svg_doc(w, h, body, defs)

    if (arr.shape[1], arr.shape[0]) != (orig_w, orig_h):
        svg = _set_svg_output_size(svg, orig_w, orig_h)
    return (
        svg,
        _build_report(cands, axes, sym_diags, opt=opt, regions=regions, diag_extra=diag_extra),
    ) if report else svg
