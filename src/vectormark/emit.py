"""C6: serialise fitted shapes into a structured SVG document."""

from __future__ import annotations

import math
import re

from .candidate import FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .fit import Shape, _fmt
from .types import Axis

_KAPPA = 0.5522847498  # cubic-bézier circle constant
_PATH_TOKEN = re.compile(r"[MLCQZ]|-?\d*\.?\d+")
_COORD_COUNT = {"M": 2, "L": 2, "C": 6, "Q": 4, "Z": 0}
_AFFINE_TOKEN = re.compile(r"[MLCQAZ]|-?\d*\.?\d+")


def shape_to_svg(shape: Shape, fill: str | None, elem_id: str) -> str:
    p = shape.params
    common = f'id="{elem_id}"' if fill is None else f'id="{elem_id}" fill="{fill}"'
    if shape.kind == "circle":
        return f'<circle {common} cx="{_fmt(p["cx"])}" cy="{_fmt(p["cy"])}" r="{_fmt(p["r"])}"/>'
    if shape.kind == "ellipse":
        return (f'<ellipse {common} cx="{_fmt(p["cx"])}" cy="{_fmt(p["cy"])}" '
                f'rx="{_fmt(p["rx"])}" ry="{_fmt(p["ry"])}"/>')
    if shape.kind == "rect":
        corners = ""
        if "rx" in p:
            corners += f' rx="{_fmt(p["rx"])}"'
        if "ry" in p:
            corners += f' ry="{_fmt(p["ry"])}"'
        return (f'<rect {common} x="{_fmt(p["x"])}" y="{_fmt(p["y"])}" '
                f'width="{_fmt(p["w"])}" height="{_fmt(p["h"])}"{corners}/>')
    if shape.kind == "polygon":
        pts = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in p["points"])
        return f'<polygon {common} points="{pts}"/>'
    if shape.kind == "annulus":
        d = _annulus_path_d(p["cx"], p["cy"], p["r_outer"], p["r_inner"])
        return f'<path {common} fill-rule="evenodd" d="{d}"/>'
    if shape.kind == "path":
        rule = f' fill-rule="{p["fill_rule"]}"' if p.get("fill_rule") else ""
        return f'<path {common}{rule} d="{p["d"]}"/>'
    if shape.kind == "use":
        a, b, c, d, e, f = p["transform"]
        use_fill = p.get("fill", fill)
        href = p.get("href")
        if href is None:
            if "href_obj_id" in p:
                raise ValueError("use shape href_obj_id must be resolved before SVG emission")
            raise ValueError("use shape requires href")
        fill_attr = "" if use_fill is None else f' fill="{use_fill}"'
        return (
            f'<use id="{elem_id}" href="#{href}" '
            f'transform="matrix({_fmt(a)} {_fmt(b)} {_fmt(c)} {_fmt(d)} {_fmt(e)} {_fmt(f)})"'
            f'{fill_attr}/>'
        )
    raise ValueError(f"unknown shape kind: {shape.kind}")


def resolve_use_shape(shape: Shape, id_map: dict[int, str]) -> Shape:
    """Resolve optimizer object-id references in Shape(\"use\") to emitted SVG ids."""
    if shape.kind != "use" or "href_obj_id" not in shape.params:
        return shape
    obj_id = int(shape.params["href_obj_id"])
    if obj_id not in id_map:
        raise ValueError(f"use shape references unknown object id: {obj_id}")
    params = dict(shape.params)
    params["href"] = id_map[obj_id]
    del params["href_obj_id"]
    return Shape("use", params)


def optimizer_objects_to_svg(objects, fills: dict[int, str] | None = None) -> list[str]:
    """Serialize optimizer objects with object-id based <use> references resolved."""
    from .optimizer.vector_region import leaves

    objects = leaves(objects)
    ordered = sorted(objects, key=lambda obj: (float(obj.z), int(obj.id)))
    id_map = {int(obj.id): f"s{idx}" for idx, obj in enumerate(ordered)}
    body: list[str] = []
    for obj in ordered:
        fill = fills.get(obj.id, "") if fills is not None else getattr(obj.fill, "hex", "")
        assert obj.current is not None
        shape = resolve_use_shape(obj.current, id_map)
        body.append(shape_to_svg(shape, fill, id_map[int(obj.id)]))
    return body


def mirror_use(ref_id: str, axis: Axis) -> str:
    """Mirror element `ref_id` about the vertical axis via a reflection matrix."""
    return f'<use href="#{ref_id}" transform="matrix(-1 0 0 1 {_fmt(2 * axis.x)} 0)"/>'


def _ellipse_path_d(cx: float, cy: float, rx: float, ry: float) -> str:
    """Closed ellipse as four cubic-bézier quarter arcs (M/C only)."""
    ox, oy = rx * _KAPPA, ry * _KAPPA
    f = _fmt
    return (
        f"M{f(cx + rx)} {f(cy)} "
        f"C{f(cx + rx)} {f(cy + oy)} {f(cx + ox)} {f(cy + ry)} {f(cx)} {f(cy + ry)} "
        f"C{f(cx - ox)} {f(cy + ry)} {f(cx - rx)} {f(cy + oy)} {f(cx - rx)} {f(cy)} "
        f"C{f(cx - rx)} {f(cy - oy)} {f(cx - ox)} {f(cy - ry)} {f(cx)} {f(cy - ry)} "
        f"C{f(cx + ox)} {f(cy - ry)} {f(cx + rx)} {f(cy - oy)} {f(cx + rx)} {f(cy)} Z"
    )


def _annulus_path_d(cx: float, cy: float, r_outer: float, r_inner: float) -> str:
    """Two concentric circle subpaths (outer + inner); under even-odd fill the
    inner subpath cuts the hole."""
    return _ellipse_path_d(cx, cy, r_outer, r_outer) + " " + _ellipse_path_d(cx, cy, r_inner, r_inner)


def _rounded_rect_path_d(x: float, y: float, w: float, h: float, rx: float, ry: float) -> str:
    """Emit a rounded rectangle with quadratic corner arcs for geometric use."""
    rx = max(0.0, min(float(rx), float(w) / 2.0))
    ry = max(0.0, min(float(ry), float(h) / 2.0))
    if rx == 0.0 or ry == 0.0:
        f = _fmt
        return f"M{f(x)} {f(y)} L{f(x + w)} {f(y)} L{f(x + w)} {f(y + h)} L{f(x)} {f(y + h)} Z"
    f = _fmt
    return (
        f"M{f(x + rx)} {f(y)} L{f(x + w - rx)} {f(y)} "
        f"Q{f(x + w)} {f(y)} {f(x + w)} {f(y + ry)} "
        f"L{f(x + w)} {f(y + h - ry)} Q{f(x + w)} {f(y + h)} {f(x + w - rx)} {f(y + h)} "
        f"L{f(x + rx)} {f(y + h)} Q{f(x)} {f(y + h)} {f(x)} {f(y + h - ry)} "
        f"L{f(x)} {f(y + ry)} Q{f(x)} {f(y)} {f(x + rx)} {f(y)} Z"
    )


def shape_to_path_d(shape: Shape) -> str:
    """Convert any fitted shape to a path `d` string using only M/L/C/Z, so a
    flattened SVG has no basic-shape elements and mirrors are a plain x-reflection."""
    p = shape.params
    if shape.kind == "path":
        return p["d"]
    if shape.kind == "use":
        if "d" in p:
            return transform_path_d(p["d"], p["transform"])
        raise ValueError("cannot convert use shape to path data without source geometry")
    if shape.kind == "circle":
        return _ellipse_path_d(p["cx"], p["cy"], p["r"], p["r"])
    if shape.kind == "ellipse":
        return _ellipse_path_d(p["cx"], p["cy"], p["rx"], p["ry"])
    if shape.kind == "rect":
        x, y, w, hh = p["x"], p["y"], p["w"], p["h"]
        if "rx" in p or "ry" in p:
            return _rounded_rect_path_d(x, y, w, hh, p.get("rx", 0.0), p.get("ry", p.get("rx", 0.0)))
        f = _fmt
        return f"M{f(x)} {f(y)} L{f(x + w)} {f(y)} L{f(x + w)} {f(y + hh)} L{f(x)} {f(y + hh)} Z"
    if shape.kind == "polygon":
        pts = p["points"]
        f = _fmt
        body = " ".join(f"L{f(x)} {f(y)}" for x, y in pts[1:])
        return f"M{f(pts[0][0])} {f(pts[0][1])} {body} Z"
    if shape.kind == "annulus":
        return _annulus_path_d(p["cx"], p["cy"], p["r_outer"], p["r_inner"])
    raise ValueError(f"unknown shape kind: {shape.kind}")


def reflect_path_d(d: str, axis_x: float) -> str:
    """Reflect every x-coordinate of an absolute M/L/C/Q/Z path about x = axis_x.

    Assumes one explicit command per coordinate set (no implicit-repeat tokens like
    ``M0 0 10 0``) and no arc ``A`` flags to invert — which is exactly what this
    package's emitters produce. A per-coordinate x flip is then exact."""
    toks = _PATH_TOKEN.findall(d)
    out: list[str] = []
    i = 0
    while i < len(toks):
        cmd = toks[i]
        out.append(cmd)
        i += 1
        n = _COORD_COUNT[cmd]
        coords = toks[i:i + n]
        i += n
        for j, c in enumerate(coords):
            v = 2 * axis_x - float(c) if j % 2 == 0 else float(c)
            out.append(_fmt(v))
    return " ".join(out)


def apply_affine_point(bake: tuple[float, float, float, float, float, float],
                       x: float, y: float) -> tuple[float, float]:
    """Apply an SVG affine (a, b, c, d, e, f) to a point: (a*x + c*y + e, b*x + d*y + f)."""
    a, b, c, d, e, f = bake
    return (a * x + c * y + e, b * x + d * y + f)


def transform_path_d(d: str, m: tuple[float, float, float, float, float, float]) -> str:
    """Apply the SVG affine `m = (a, b, c, d, e, f)` to every point of an absolute
    path, so the transform can be *baked* into the geometry instead of carried on a
    wrapping `<g transform>`. Handles M/L/C/Q/Z exactly; for an elliptical-arc `A`
    only the endpoint moves and the x-axis-rotation advances by the affine's
    rotation. Reflections reverse arc orientation, so they also flip the sweep flag."""
    a, b, c, dd, e, f = m
    flips_orientation = (a * dd - b * c) < 0.0

    def pt(x: float, y: float) -> tuple[float, float]:
        return apply_affine_point(m, x, y)

    toks = _AFFINE_TOKEN.findall(d)
    out: list[str] = []
    i = 0
    while i < len(toks):
        cmd = toks[i]
        out.append(cmd)
        i += 1
        if cmd == "Z":
            continue
        if cmd == "A":
            arc_tokens = list(toks[i:i + 7])
            rx, ry, xrot, large, sweep, x, y = arc_tokens
            i += 7
            nx, ny = pt(float(x), float(y))
            theta = math.radians(float(xrot))
            axis_x = a * math.cos(theta) + c * math.sin(theta)
            axis_y = b * math.cos(theta) + dd * math.sin(theta)
            transformed_xrot = (math.degrees(math.atan2(axis_y, axis_x)) + 360.0) % 360.0
            if flips_orientation:
                sweep = "0" if int(float(sweep)) else "1"
            out += [rx, ry, _fmt(transformed_xrot), large, sweep, _fmt(nx), _fmt(ny)]
            continue
        n = _COORD_COUNT[cmd]
        nums = [float(t) for t in toks[i:i + n]]
        i += n
        for k in range(0, n, 2):
            nx, ny = pt(nums[k], nums[k + 1])
            out += [_fmt(nx), _fmt(ny)]
    return " ".join(out)


def path_svg(d: str, fill: str, fill_rule: str | None = None) -> str:
    rule = f' fill-rule="{fill_rule}"' if fill_rule else ""
    return f'<path fill="{fill}"{rule} d="{d}"/>'


def _gradient_stops(stops: list[tuple[float, str]]) -> str:
    return "".join(f'<stop offset="{_fmt(o)}" stop-color="{c}"/>' for o, c in stops)


def linear_gradient_def(elem_id: str, x1: float, y1: float, x2: float, y2: float,
                        stops: list[tuple[float, str]]) -> str:
    """A <linearGradient> in userSpaceOnUse coords (absolute px; no gradientTransform,
    survives --flatten)."""
    return (f'<linearGradient id="{elem_id}" gradientUnits="userSpaceOnUse" '
            f'x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}">'
            f'{_gradient_stops(stops)}</linearGradient>')


def radial_gradient_def(elem_id: str, cx: float, cy: float, r: float,
                        stops: list[tuple[float, str]], *,
                        focal_point: tuple[float, float] | None = None) -> str:
    """A <radialGradient> in userSpaceOnUse coords."""
    focal = "" if focal_point is None else f' fx="{_fmt(focal_point[0])}" fy="{_fmt(focal_point[1])}"'
    return (f'<radialGradient id="{elem_id}" gradientUnits="userSpaceOnUse" '
            f'cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}"{focal}>'
            f'{_gradient_stops(stops)}</radialGradient>')


def pattern_image_def(elem_id: str, x: float, y: float, w: float, h: float,
                      png_b64: str, transform: tuple | None = None) -> str:
    """A <pattern> paint server holding one stretched <image> spanning the bbox
    (preserveAspectRatio='none' => bilinear stretch). userSpaceOnUse + absolute
    coords so it survives --flatten; `transform` (an SVG affine a,b,c,d,e,f) is
    emitted as patternTransform to map the pattern frame to a baked frame."""
    pt = ""
    if transform is not None:
        a, b, c, d, e, f = transform
        pt = (f' patternTransform="matrix({_fmt(a)} {_fmt(b)} {_fmt(c)} '
              f'{_fmt(d)} {_fmt(e)} {_fmt(f)})"')
    return (f'<pattern id="{elem_id}" patternUnits="userSpaceOnUse"{pt} '
            f'x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}">'
            f'<image href="data:image/png;base64,{png_b64}" '
            f'x="0" y="0" width="{_fmt(w)}" height="{_fmt(h)}" '
            f'preserveAspectRatio="none"/></pattern>')


def fill_rule_for(geometry: Shape) -> str | None:
    """SVG fill-rule for a geometry: an explicit params['fill_rule'] if present, else
    'evenodd' for an annulus (its two same-winding circles only read as a ring under
    even-odd), else None."""
    return geometry.params.get("fill_rule", "evenodd" if geometry.kind == "annulus" else None)


def resolve_fill(fill, defs: list[str], *, geometry: dict | None = None,
                 transform: tuple | None = None) -> str:
    """Resolve a Fill to an SVG fill attribute. FlatFill -> its hex. Gradient/raster
    fill -> register a <def> (id g{len(defs)}, minted BEFORE the append) and return
    url(#id). `geometry` overrides the gradient/raster coords (used when the caller
    baked them); `transform` is the raster patternTransform (gradients ignore it)."""
    if isinstance(fill, FlatFill):
        return fill.hex
    gid = f"g{len(defs)}"
    g = geometry if geometry is not None else fill.geometry
    if isinstance(fill, RasterFill):
        defs.append(pattern_image_def(gid, g["x"], g["y"], g["w"], g["h"],
                                      fill.png_b64, transform))
    elif isinstance(fill, LinearGradientFill):
        defs.append(linear_gradient_def(gid, g["x1"], g["y1"], g["x2"], g["y2"], fill.stops))
    else:
        focal = (
            (g.get("fx", g["cx"]), g.get("fy", g["cy"]))
            if "fx" in g or "fy" in g else None
        )
        defs.append(radial_gradient_def(gid, g["cx"], g["cy"], g["r"], fill.stops, focal_point=focal))
    return f"url(#{gid})"


def render_svg_doc(width: int, height: int, body: list[str], defs: list[str] | None = None) -> str:
    defs_block = f'  <defs>{"".join(defs)}</defs>\n  ' if defs else "  "
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n'
        + defs_block
        + "\n  ".join(body)
        + "\n</svg>\n"
    )
