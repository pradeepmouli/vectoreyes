"""Corner provenance and conservative corner-local normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

from . import corner_matching, corner_normalization
from .path_geometry import command_from_token, parse_svg_paths
from .path_sampling import _parse_subpaths

def path_corner_diagnostics(
    d: str,
    *,
    declared_corners: Sequence[tuple[float, float]] = (),
    declared_corner_spans: Sequence[tuple[tuple[float, float], tuple[float, float]]] = (),
    corner_tolerance: float = 0.02,
) -> dict[str, object]:
    """Bind pre-fit declarations; boundary anchors do not own adjacent sides."""
    declarations = []
    for subpath, _tokens in enumerate(_parse_subpaths(d)):
        declarations.extend({
            "corner_id": f"p{subpath}.corner{number}", "kind": "point",
            "subpath": subpath, "source": "prefit", "point": tuple(point),
        } for number, point in enumerate(declared_corners))
        declarations.extend({
            "corner_id": f"p{subpath}.rounded_corner{number}", "kind": "curve",
            "subpath": subpath, "source": "prefit",
            "start_point": tuple(start), "end_point": tuple(end),
        } for number, (start, end) in enumerate(declared_corner_spans))
    return _bind_prefit_spans(d, declarations, tolerance=corner_tolerance)


def _bind_prefit_spans(d: str, declarations: Sequence, *, tolerance: float) -> dict[str, object]:
    """Project SVG input/output around the shared typed evidence matcher."""
    paths = {subpath: tuple(command_from_token(kind, values, command_id=f"p{subpath}.c{index}")
        for index, (kind, values) in enumerate(tokens) if kind != "Z")
        for subpath, tokens in enumerate(_parse_subpaths(d))} if declarations else {}
    matched, skipped = corner_matching.match_prefit_corner_spans(paths, declarations, tolerance=tolerance)
    spans = [{**span, **{field: list(span[field]) for field in
        ("commands", "curve_commands", "anchor_commands")}} for span in matched]
    anchors: dict[tuple[int, int], dict[str, object]] = {}
    for span in spans:
        subpath = span["subpath"]
        for index in span["anchor_commands"]:
            key = (subpath, index)
            anchor = anchors.setdefault(key, {"subpath": subpath, "command_index": index,
                "anchor_kind": "corner", "corner_id": span["corner_id"], "corner_ids": []})
            if span["corner_id"] not in anchor["corner_ids"]:
                anchor["corner_ids"].append(span["corner_id"])
    return recount_corner_diagnostics(d, {
        "path": d, "anchors": [anchors[key] for key in sorted(anchors)],
        "spans": spans, "skipped": [dict(item) for item in skipped],
    })


def recount_corner_diagnostics(d: str, diagnostics: Mapping[str, object]) -> dict[str, object]:
    """Derive counters from the retained commands and exact declared ownership."""
    paths = _parse_subpaths(d)
    total = sum(command not in {"M", "Z"} for tokens in paths for command, _ in tokens)
    spans = diagnostics.get("spans", ())
    anchors = diagnostics.get("anchors", ())
    owned = {(span["subpath"], index) for span in spans for index in span.get("commands", ())}
    return {
        **diagnostics,
        "commands": {"total": total, "corner": len(owned), "free": total - len(owned)},
        "anchor_counts": {"total": total + len(paths), "corner": len(anchors)},
        "corner_counts": {"total": len(spans), "point": sum(s.get("kind") == "point" for s in spans),
                          "curve": sum(s.get("kind") == "curve" for s in spans)},
    }


def source_ordered_corner_diagnostics(d: str, evidence: Mapping[str, object]) -> dict[str, object]:
    """Flatten the root fitter's per-path declarations by original path index.

    The producer emits local subpath zero in each record. This is identity
    transport only: stale/malformed records are explicit skips, never inferred
    from coordinates or from the path after topology classification.
    """
    if "spans" in evidence or "subpaths" not in evidence or evidence.get("path") != d:
        return dict(evidence)
    records = evidence["subpaths"]
    raw = tuple(part.strip() for part in re.findall(r"M[^M]*", d))
    skipped = list(evidence.get("skipped", ()))
    spans, anchors = [], []
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or len(records) != len(raw):
        return {**evidence, "spans": [], "anchors": [], "skipped": [*skipped, {"reason": "invalid_corner_provenance"}]}
    for index, (record, path_d) in enumerate(zip(records, raw, strict=True)):
        if not isinstance(record, Mapping) or record.get("path") != path_d:
            skipped.append({"subpath": index, "reason": "missing_or_stale_provenance"})
            continue
        skipped.extend(record.get("skipped", ()))
        for field, output in (("spans", spans), ("anchors", anchors)):
            for declaration in record.get(field, ()):
                if not isinstance(declaration, Mapping) or declaration.get("subpath") != 0:
                    skipped.append({"subpath": index, "reason": "invalid_corner_provenance"})
                    continue
                identity = str(declaration.get("corner_id", ""))
                if identity.startswith("p0."):
                    identity = f"p{index}." + identity[3:]
                output.append({**declaration, "subpath": index, "corner_id": identity})
    return {**evidence, "spans": spans, "anchors": anchors, "skipped": skipped}


def rebind_prefit_corner_diagnostics(
    d: str,
    diagnostics: Mapping[str, object],
    *,
    tolerance: float,
) -> dict[str, object]:
    """Carry existing pre-fit corner provenance across a path rewrite.

    A rewrite may move an endpoint slightly or expand one command into several.
    It may not create corner semantics.  Rebind only the explicitly recorded
    points/spans to their nearest retained endpoints in the rewritten path.
    """
    spans = diagnostics.get("spans", ())
    return _bind_prefit_spans(d, spans if isinstance(spans, Sequence) else (), tolerance=tolerance)


def _subpath_d(tokens: list[tuple[str, list[float]]]) -> str:
    # Only newly fitted controls are rounded. Retained endpoints/side handles
    # must not drift merely because another corner in this path was normalized.
    return " ".join(
        command if not values else f"{command}{' '.join(str(float(value)) for value in values)}"
        for command, values in tokens
    )


def retained_corner_declaration_matches(
    d: str, diagnostics: object
) -> bool:
    """Whether diagnostics are pre-fit corner evidence for this exact path."""
    return isinstance(diagnostics, Mapping) and diagnostics.get("path") == d


def normalize_corners_path_d(
    d: str, *, max_error: float, diagnostics: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Project text around the canonical declared-corner operation."""
    if not retained_corner_declaration_matches(d, diagnostics):
        result = path_corner_diagnostics(d)
        result["skipped"] = [{"reason": "missing_or_stale_provenance"}]
        result["normalized"] = []
        return d, result
    # An authoritative empty selection has no geometry to fit or parse.
    paths = () if not diagnostics.get('spans') else parse_svg_paths(d, region_id="corner",
        path_ids=tuple(f"p{index}" for index in range(len(_parse_subpaths(d)))))
    _, report = corner_normalization.normalize_declared_paths(
        paths, declaration=diagnostics, max_error=max_error, source_text=d)
    return str(report["path"]), dict(report)


def anchor_annotations(diagnostics: Mapping[str, object], subpath: int) -> dict[int, dict[str, object]]:
    """Return trace-command annotations for one subpath."""
    anchors = diagnostics.get("anchors", ())
    if not isinstance(anchors, list):
        return {}
    return {
        int(anchor["command_index"]): dict(anchor)
        for anchor in anchors
        if isinstance(anchor, Mapping) and int(anchor.get("subpath", -1)) == subpath
    }
