#!/usr/bin/env python3
"""Spike: skia-python for SVG path operations (Simplify, Op, OpBuilder).

Demonstrates the three key path-operation primitives from skia-python and shows
where each fits into the vectoreyes pipeline today.

Run:
    pip install skia-python          # or: uv pip install skia-python
    python prototypes/skia_path_ops.py

Requires:
    skia-python >= 87.4 (developed against 144.0.post2; any release since 87.4
                         should work — check the skia-python changelog for
                         PathOp/Simplify API changes if upgrading across major
                         Skia milestones)
    libegl1               (apt install libegl1 on Debian/Ubuntu headless)
"""

from __future__ import annotations

import math
import re
import time

import skia

# ---------------------------------------------------------------------------
# SVG path ↔ skia.Path conversion helpers
# ---------------------------------------------------------------------------
# Supported commands: M L Q C Z  (no A/arc — vectoreyes paths don't emit them
# via the normal fit path, but the round-trip helpers cope with everything the
# pipeline produces)

_NUM_RE = re.compile(r"[MLQCAZ]|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

# Verb enum integers from skia.Path.RawIter (not exposed as named constants in
# the Python bindings at this version)
_MOVE  = 0
_LINE  = 1
_QUAD  = 2
_CUBIC = 4
_CLOSE = 5


def _fmt(v: float) -> str:
    """Format a float the same way vectoreyes.fit._fmt does."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


def svg_d_to_skia(d: str, *, fill_type: skia.PathFillType = skia.PathFillType.kWinding) -> skia.Path:
    """Parse an SVG path ``d`` string into a :class:`skia.Path`.

    Supports the subset of path commands that vectoreyes emits: M, L, Q, C, Z.
    """
    path = skia.Path()
    path.setFillType(fill_type)
    tokens = _NUM_RE.findall(d)
    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        i += 1
        if cmd == "M":
            path.moveTo(float(tokens[i]), float(tokens[i + 1]))
            i += 2
        elif cmd == "L":
            path.lineTo(float(tokens[i]), float(tokens[i + 1]))
            i += 2
        elif cmd == "Q":
            path.quadTo(float(tokens[i]), float(tokens[i + 1]),
                        float(tokens[i + 2]), float(tokens[i + 3]))
            i += 4
        elif cmd == "C":
            path.cubicTo(float(tokens[i]),     float(tokens[i + 1]),
                         float(tokens[i + 2]), float(tokens[i + 3]),
                         float(tokens[i + 4]), float(tokens[i + 5]))
            i += 6
        elif cmd == "Z":
            path.close()
    return path


def skia_to_svg_d(path: skia.Path) -> str:
    """Serialise a :class:`skia.Path` back to an SVG ``d`` string.

    Produces absolute M/L/Q/C/Z commands (no arc commands — Skia may emit conic
    segments if it chooses, but for the winding-clean paths returned by Simplify
    and Op that doesn't occur in practice).
    """
    parts: list[str] = []
    for verb, pts in skia.Path.RawIter(path):
        v = int(verb)
        if v == _MOVE:
            parts.append(f"M{_fmt(pts[0].x())} {_fmt(pts[0].y())}")
        elif v == _LINE:
            parts.append(f"L{_fmt(pts[1].x())} {_fmt(pts[1].y())}")
        elif v == _QUAD:
            parts.append(
                f"Q{_fmt(pts[1].x())} {_fmt(pts[1].y())} "
                f"{_fmt(pts[2].x())} {_fmt(pts[2].y())}"
            )
        elif v == _CUBIC:
            parts.append(
                f"C{_fmt(pts[1].x())} {_fmt(pts[1].y())} "
                f"{_fmt(pts[2].x())} {_fmt(pts[2].y())} "
                f"{_fmt(pts[3].x())} {_fmt(pts[3].y())}"
            )
        elif v == _CLOSE:
            parts.append("Z")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Helper: circle path d using cubic-Bézier approximation (four quarter arcs)
# — identical to what vectoreyes.emit._ellipse_path_d produces.
# ---------------------------------------------------------------------------

_KAPPA = 0.5522847498


def _circle_d(cx: float, cy: float, r: float) -> str:
    k = r * _KAPPA
    return (
        f"M{_fmt(cx + r)} {_fmt(cy)} "
        f"C{_fmt(cx + r)} {_fmt(cy + k)} {_fmt(cx + k)} {_fmt(cy + r)} {_fmt(cx)} {_fmt(cy + r)} "
        f"C{_fmt(cx - k)} {_fmt(cy + r)} {_fmt(cx - r)} {_fmt(cy + k)} {_fmt(cx - r)} {_fmt(cy)} "
        f"C{_fmt(cx - r)} {_fmt(cy - k)} {_fmt(cx - k)} {_fmt(cy - r)} {_fmt(cx)} {_fmt(cy - r)} "
        f"C{_fmt(cx + k)} {_fmt(cy - r)} {_fmt(cx + r)} {_fmt(cy - k)} {_fmt(cx + r)} {_fmt(cy)} Z"
    )


# ===========================================================================
# Section 1: skia.Simplify
# ===========================================================================
# Converts a path with self-intersecting or overlapping sub-paths into a
# minimal set of non-overlapping contours that describe the same filled area.
# This is the Skia equivalent of Shapely's Polygon(...).buffer(0) but operates
# directly on Bézier path data — no rasterisation, no polygon approximation.
#
# Pipeline fit: the output of vectoreyes's fit_path / refine passes can produce
# compound paths (fill rule: evenodd) where two same-winding sub-paths overlap
# near a symmetry axis.  Simplify resolves them to a single clean contour.

def demo_simplify() -> None:
    print("=" * 60)
    print("1. skia.Simplify — remove self-intersections")
    print("=" * 60)

    # 1a: Two overlapping rectangles as a single compound path.
    # Without simplification the renderer relies on fill rule to resolve the
    # overlap, but the path itself has a self-intersection at the crossing seam.
    p = skia.Path()
    for x0, y0 in [(0, 0), (40, 40)]:
        p.moveTo(x0, y0)
        p.lineTo(x0 + 80, y0)
        p.lineTo(x0 + 80, y0 + 80)
        p.lineTo(x0, y0 + 80)
        p.close()

    simplified = skia.Simplify(p)
    print(f"Two overlapping rects  — input verbs: {p.countVerbs():3d}  "
          f"→  simplified: {simplified.countVerbs()}")
    print(f"  fill type: {simplified.getFillType()}")

    # 1b: Annulus path — two concentric circles with even-odd fill.
    # vectoreyes.emit._annulus_path_d produces exactly this.
    d_annulus = _circle_d(60, 60, 50) + " " + _circle_d(60, 60, 25)
    p_ann = svg_d_to_skia(d_annulus, fill_type=skia.PathFillType.kEvenOdd)
    s_ann = skia.Simplify(p_ann)
    print(f"\nAnnulus (even-odd)     — input verbs: {p_ann.countVerbs():3d}  "
          f"→  simplified: {s_ann.countVerbs()}")
    print(f"  fill type: {s_ann.getFillType()}")
    print(f"  output d (first 120): {skia_to_svg_d(s_ann)[:120]}…")

    # 1c: A self-intersecting star (5-pointed using a single closed polyline that
    # crosses itself).  The simplified path has a separate centre pentagon.
    star = skia.Path()
    for i in range(5):
        a = 2 * math.pi * i / 5 - math.pi / 2
        x = 100 + 80 * math.cos(a)
        y = 100 + 80 * math.sin(a)
        ia = a + math.pi / 5
        ix = 100 + 33 * math.cos(ia)
        iy = 100 + 33 * math.sin(ia)
        if i == 0:
            star.moveTo(x, y)
        else:
            star.lineTo(x, y)
        star.lineTo(ix, iy)
    star.close()
    s_star = skia.Simplify(star)
    print(f"\nSelf-intersecting star — input verbs: {star.countVerbs():3d}  "
          f"→  simplified: {s_star.countVerbs()}")


# ===========================================================================
# Section 2: skia.Op — pairwise boolean path operations
# ===========================================================================
# Op(a, b, operator) computes the geometric boolean of two paths and returns a
# clean, non-self-intersecting result.  Operators:
#   kUnion_PathOp           a ∪ b
#   kIntersect_PathOp       a ∩ b
#   kDifference_PathOp      a − b
#   kReverseDifference_PathOp  b − a
#   kXOR_PathOp             a ⊕ b  (symmetric difference)
#
# Pipeline fit:
#   • Replacing compound paths (fill rule: evenodd, outer + inner) with an
#     Op(outer, inner, kDifference_PathOp) result gives the same geometry but
#     expressed as a winding-safe single-contour-pair path that any renderer
#     interprets correctly without needing fill-rule.
#   • compound.py's split_compound_pass separates a multi-subpath shape into
#     individual primitives; Op(difference) could instead punch precise holes.

def demo_op() -> None:
    print("\n" + "=" * 60)
    print("2. skia.Op — boolean path operations")
    print("=" * 60)

    outer = svg_d_to_skia(_circle_d(60, 60, 50))
    inner = svg_d_to_skia(_circle_d(60, 60, 25))

    for label, op_const in [
        ("kUnion_PathOp       (a ∪ b)", skia.kUnion_PathOp),
        ("kDifference_PathOp  (a − b)", skia.kDifference_PathOp),
        ("kIntersect_PathOp   (a ∩ b)", skia.kIntersect_PathOp),
        ("kXOR_PathOp         (a ⊕ b)", skia.kXOR_PathOp),
    ]:
        result = skia.Op(outer, inner, op_const)
        print(f"  Op({label}): {result.countVerbs()} verbs, "
              f"fill type={result.getFillType()}")

    # Practical demo: build an annulus via difference rather than even-odd
    diff = skia.Op(outer, inner, skia.kDifference_PathOp)
    d_result = skia_to_svg_d(diff)
    print(f"\nAnnulus via kDifference_PathOp:")
    print(f"  verbs: {diff.countVerbs()}, fill type: {diff.getFillType()}")
    print(f"  d (first 150): {d_result[:150]}…")


# ===========================================================================
# Section 3: skia.OpBuilder — batch boolean over N paths
# ===========================================================================
# OpBuilder accumulates paths with their operator and resolves them all at once.
# This is more efficient than chaining pairwise Op() calls for large N.
#
# Pipeline fit:
#   • When multiple regions share the same color and overlap (e.g. two mirrored
#     gradient bands that meet at the axis) their filled shapes can be unioned
#     into a single path, eliminating a layer and any seam rendering artefact.
#   • Replaces the current Shapely-based unary_union + contour-extraction path
#     in vectoreyes.optimizer.passes.simplify.

def demo_op_builder() -> None:
    print("\n" + "=" * 60)
    print("3. skia.OpBuilder — batch union of N shapes")
    print("=" * 60)

    # Simulate three overlapping rounded-quad patches (like same-color regions
    # that overlap near the symmetry axis after mirroring).
    rects = [(0, 0, 70, 70), (30, 30, 100, 100), (60, 0, 130, 70)]
    builder = skia.OpBuilder()
    for x0, y0, x1, y1 in rects:
        p = skia.Path()
        p.moveTo(x0, y0)
        p.lineTo(x1, y0)
        p.lineTo(x1, y1)
        p.lineTo(x0, y1)
        p.close()
        builder.add(p, skia.kUnion_PathOp)

    result = builder.resolve()
    print(f"Union of {len(rects)} overlapping rects: {result.countVerbs()} verbs")
    print(f"  d: {skia_to_svg_d(result)}")


# ===========================================================================
# Section 4: Benchmark — skia.Op vs Shapely for annulus construction
# ===========================================================================

def _bench(fn, *, iterations: int = 200) -> float:
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iterations * 1000  # ms per call


def demo_benchmark() -> None:
    print("\n" + "=" * 60)
    print("4. Benchmark: skia.Op vs Shapely (annulus construction)")
    print("=" * 60)

    d_outer = _circle_d(60, 60, 50)
    d_inner = _circle_d(60, 60, 25)

    def skia_annulus():
        p1 = svg_d_to_skia(d_outer)
        p2 = svg_d_to_skia(d_inner)
        return skia.Op(p1, p2, skia.kDifference_PathOp)

    def skia_simplify_annulus():
        d = d_outer + " " + d_inner
        p = svg_d_to_skia(d, fill_type=skia.PathFillType.kEvenOdd)
        return skia.Simplify(p)

    rows: list[tuple[str, float, int]] = []

    ms = _bench(skia_annulus)
    r = skia_annulus()
    rows.append(("skia Op(difference)", ms, r.countVerbs()))

    ms = _bench(skia_simplify_annulus)
    r = skia_simplify_annulus()
    rows.append(("skia Simplify(even-odd)", ms, r.countVerbs()))

    # Optional: Shapely comparison
    try:
        from shapely.geometry import Polygon  # type: ignore[import-untyped]
        from shapely.ops import unary_union   # noqa: F401

        def shapely_annulus():
            # Approximate circles as 64-point polygons (the standard Shapely approach)
            n = 64
            outer = Polygon([(60 + 50 * math.cos(2 * math.pi * i / n),
                              60 + 50 * math.sin(2 * math.pi * i / n)) for i in range(n)])
            inner = Polygon([(60 + 25 * math.cos(2 * math.pi * i / n),
                              60 + 25 * math.sin(2 * math.pi * i / n)) for i in range(n)])
            return outer.difference(inner)

        ms = _bench(shapely_annulus)
        rows.append(("Shapely difference (64-pt polygon)", ms, 0))
    except ImportError:
        pass

    print(f"  {'Method':<38}  {'ms/call':>8}  {'verbs':>6}")
    print(f"  {'-'*38}  {'-'*8}  {'-'*6}")
    for label, ms, verbs in rows:
        verb_str = str(verbs) if verbs else "n/a"
        print(f"  {label:<38}  {ms:8.3f}  {verb_str:>6}")


# ===========================================================================
# Section 5: Pipeline integration summary
# ===========================================================================

def demo_integration_notes() -> None:
    print("\n" + "=" * 60)
    print("5. Pipeline integration notes")
    print("=" * 60)
    notes = """
Where skia-python path ops could replace or augment the current stack
----------------------------------------------------------------------

A. vectoreyes.optimizer.passes.simplify — path simplification pass
   Current: samples the contour → fits quadratic/cubic Béziers → Shapely
            Polygon to validate.
   Skia:    skia.Simplify(path) on the path-d directly, no rasterisation.
            Works on cubic Bézier data; Shapely approximates with polygons.

B. vectoreyes.emit._annulus_path_d — compound paths with even-odd fill rule
   Current: emits outer + inner circle sub-paths and relies on fill rule
            evenodd in the SVG renderer.
   Skia:    Op(outer, inner, kDifference_PathOp) produces a winding-safe
            two-contour path without needing fill-rule — compatible with
            any renderer including ones that ignore fill-rule.

C. vectoreyes.optimizer.passes.compound — split_compound_pass
   Current: splits a multi-subpath shape using Shapely for area/footprint.
   Skia:    Op could punch precise Bézier holes into the parent path before
            splitting, giving exact boolean geometry vs polygon approximation.

D. vectoreyes.optimizer.passes.seams — future region merging
   Current: uses mask-level union (numpy boolean arrays).
   Skia:    OpBuilder.add(path, kUnion_PathOp) + resolve() for a multi-region
            union at the Bézier level — avoids re-tracing after union.

Key caveats
-----------
• skia-python requires libeGL (libegl1) on Linux headless environments.
• skia.Simplify always returns kEvenOdd fill type; callers must set fill-rule
  in the emitted SVG accordingly, or post-process to kWinding.
• Op results with kDifference produce two-subpath kEvenOdd paths for the
  annulus case — same structure as the current emitter, just Bézier-exact.
• Skia path verbs are ints 0/1/2/4/5 for Move/Line/Quad/Cubic/Close; the
  Python binding does not expose a named enum usable in comparisons (use
  int(verb)).
"""
    print(notes)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print(f"skia-python version: {skia.__version__}\n")
    demo_simplify()
    demo_op()
    demo_op_builder()
    demo_benchmark()
    demo_integration_notes()

