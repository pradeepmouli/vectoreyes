"""Canonical closed SVG path geometry for the drawing engine.

The engine keeps ``M`` as the stable origin of each path, but does not retain
``Z`` as a command: every :class:`ClosedPath` is closed by invariant and SVG
serialization emits ``Z``. Commands own their endpoint coordinates. Anchors
are separate, optional semantic annotations over the commands that participate
in a meaningful feature, so geometry has no duplicated point state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias
import re


Point: TypeAlias = tuple[float, float]
AnchorKind: TypeAlias = Literal["corner", "inflection", "extremum", "junction"]
_SVG_PATH_TOKEN = re.compile(r"[MLQCAZ]|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _point(value: Point) -> Point:
    x, y = float(value[0]), float(value[1])
    if not (isfinite(x) and isfinite(y)):
        raise ValueError("path points must be finite")
    return x, y


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class PathCommand:
    """Base command with an addressable endpoint in a closed path."""

    id: str
    end: Point

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("path command IDs must not be empty")
        object.__setattr__(self, "end", _point(self.end))


@dataclass(frozen=True)
class MoveCommand(PathCommand):
    """The stable, explicit starting point of one closed path."""


@dataclass(frozen=True)
class LineCommand(PathCommand):
    """A straight segment ending at ``end``."""


@dataclass(frozen=True)
class QuadraticCommand(PathCommand):
    """A quadratic Bézier segment ending at ``end``."""

    control: Point

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "control", _point(self.control))


@dataclass(frozen=True)
class CubicCommand(PathCommand):
    """A cubic Bézier segment ending at ``end``."""

    control_1: Point
    control_2: Point

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "control_1", _point(self.control_1))
        object.__setattr__(self, "control_2", _point(self.control_2))


@dataclass(frozen=True)
class ArcCommand(PathCommand):
    """An SVG elliptical-arc segment ending at ``end``."""

    rx: float
    ry: float
    rotation_degrees: float
    large_arc: bool
    sweep: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        if not all(isfinite(float(value)) for value in (self.rx, self.ry, self.rotation_degrees)):
            raise ValueError("arc parameters must be finite")
        if self.rx < 0 or self.ry < 0:
            raise ValueError("arc radii must be non-negative")


DrawableCommand: TypeAlias = LineCommand | QuadraticCommand | CubicCommand | ArcCommand


@dataclass(frozen=True)
class Anchor:
    """Semantic annotation over the commands that form one feature.

    A corner can own each command of its rounded span; an inflection anchor can
    own the command(s) that meet at its split. Constraints such as ``snap`` and
    ``stitch`` are operations, not anchor kinds.
    """

    id: str
    command_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self) is Anchor:
            raise TypeError("anchors must use a concrete anchor variant")
        if not self.id:
            raise ValueError("anchor IDs must not be empty")
        if not self.command_ids:
            raise ValueError("anchors must reference at least one command")
        if len(set(self.command_ids)) != len(self.command_ids):
            raise ValueError("anchor command IDs must be unique")

    @property
    def kind(self) -> AnchorKind:
        raise NotImplementedError


@dataclass(frozen=True)
class CornerAnchor(Anchor):
    """A sharp or rounded corner; a fitted radius is optional."""

    radius: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.radius is not None and (not isfinite(float(self.radius)) or self.radius < 0):
            raise ValueError("corner radius must be finite and non-negative")

    @property
    def kind(self) -> Literal["corner"]:
        return "corner"


@dataclass(frozen=True)
class InflectionAnchor(Anchor):
    """A fitted curve inflection that bounds separately fitted spans."""

    @property
    def kind(self) -> Literal["inflection"]:
        return "inflection"


@dataclass(frozen=True)
class ExtremumAnchor(Anchor):
    """A local path extremum retained by curve simplification."""

    @property
    def kind(self) -> Literal["extremum"]:
        return "extremum"


@dataclass(frozen=True)
class JunctionAnchor(Anchor):
    """A path junction paired with another junction in the drawing."""

    reference_junction_id: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.reference_junction_id:
            raise ValueError("junction anchors require a reference junction ID")
        if self.reference_junction_id == self.id:
            raise ValueError("junction anchors cannot reference themselves")

    @property
    def kind(self) -> Literal["junction"]:
        return "junction"


@dataclass(frozen=True)
class ClosedPath:
    """A closed SVG subpath with an explicit ``M`` and no internal ``Z``."""

    id: str
    commands: tuple[PathCommand, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("path IDs must not be empty")
        if len(self.commands) < 3:
            raise ValueError("closed paths require M plus at least two drawable commands")
        if not isinstance(self.commands[0], MoveCommand):
            raise ValueError("the first closed-path command must be MoveCommand")
        if any(isinstance(command, MoveCommand) for command in self.commands[1:]):
            raise ValueError("closed paths may contain only one MoveCommand")
        if any(
            not isinstance(command, (LineCommand, QuadraticCommand, CubicCommand, ArcCommand))
            for command in self.commands[1:]
        ):
            raise ValueError("closed paths may contain only drawable command variants after MoveCommand")
        if len({command.id for command in self.commands}) != len(self.commands):
            raise ValueError("path command IDs must be unique within a path")

    @property
    def start(self) -> Point:
        return self.commands[0].end

    @property
    def drawable_commands(self) -> tuple[DrawableCommand, ...]:
        return tuple(self.commands[1:])  # type: ignore[return-value]

    def to_svg_d(self) -> str:
        """Serialize this closed path, emitting ``Z`` from the closure invariant."""
        parts = [f"M{_fmt(self.start[0])} {_fmt(self.start[1])}"]
        for command in self.drawable_commands:
            if isinstance(command, LineCommand):
                parts.append(f"L{_fmt(command.end[0])} {_fmt(command.end[1])}")
            elif isinstance(command, QuadraticCommand):
                parts.append(
                    f"Q{_fmt(command.control[0])} {_fmt(command.control[1])} "
                    f"{_fmt(command.end[0])} {_fmt(command.end[1])}"
                )
            elif isinstance(command, CubicCommand):
                parts.append(
                    f"C{_fmt(command.control_1[0])} {_fmt(command.control_1[1])} "
                    f"{_fmt(command.control_2[0])} {_fmt(command.control_2[1])} "
                    f"{_fmt(command.end[0])} {_fmt(command.end[1])}"
                )
            else:
                parts.append(
                    f"A{_fmt(command.rx)} {_fmt(command.ry)} {_fmt(command.rotation_degrees)} "
                    f"{int(command.large_arc)} {int(command.sweep)} "
                    f"{_fmt(command.end[0])} {_fmt(command.end[1])}"
                )
        return " ".join([*parts, "Z"])


@dataclass(frozen=True)
class PathGeometry:
    """One closed path plus non-geometric command-feature annotations."""

    path: ClosedPath
    anchors: tuple[Anchor, ...] = ()

    def __post_init__(self) -> None:
        if len({anchor.id for anchor in self.anchors}) != len(self.anchors):
            raise ValueError("anchor IDs must be unique within geometry")
        command_ids = {
            command.id
            for command in self.path.commands
        }
        if len(command_ids) != len(self.path.commands):
            raise ValueError("path command IDs must be unique within geometry")
        for anchor in self.anchors:
            unknown = set(anchor.command_ids) - command_ids
            if unknown:
                raise ValueError(f"anchor references unknown commands: {sorted(unknown)!r}")

    def endpoint(self, command_id: str) -> Point:
        """Resolve a command endpoint without exposing path-storage details."""
        for command in self.path.commands:
            if command.id == command_id:
                return command.end
        raise KeyError(command_id)

    def to_svg_d(self) -> str:
        """Serialize the closed path for SVG output."""
        return self.path.to_svg_d()


@dataclass(frozen=True)
class PathGeometryWithCutouts:
    """One outer path geometry with one or more explicitly subtractive paths."""

    outer: PathGeometry
    cutouts: tuple[PathGeometry, ...]

    def __post_init__(self) -> None:
        if not self.cutouts:
            raise ValueError("use PathGeometry when there are no cutouts")
        path_ids = [self.outer.path.id, *(cutout.path.id for cutout in self.cutouts)]
        if len(set(path_ids)) != len(path_ids):
            raise ValueError("outer and cutout path IDs must be unique")

    @property
    def fill_rule(self) -> Literal["evenodd"]:
        """Cutout membership is explicit rather than inferred from winding."""
        return "evenodd"

    def to_svg_d(self) -> str:
        """Serialize the outer path followed by its subtractive cutouts."""
        return " ".join([
            self.outer.to_svg_d(),
            *(cutout.to_svg_d() for cutout in self.cutouts),
        ])


@dataclass(frozen=True)
class SymmetryAxis:
    """An explicit reflection line in drawing coordinates."""

    point: Point
    angle_degrees: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _point(self.point))
        if not isfinite(float(self.angle_degrees)):
            raise ValueError("symmetry-axis angle must be finite")


@dataclass(frozen=True)
class PathGeometryWithSymmetry:
    """A path half and the axis used to construct its reflected counterpart."""

    half: PathGeometry
    axis: SymmetryAxis


@dataclass(frozen=True)
class AffineTransform:
    """The SVG-compatible affine matrix used by geometry clones."""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(float(value)) for value in (self.a, self.b, self.c, self.d, self.tx, self.ty)):
            raise ValueError("affine-transform coefficients must be finite")


@dataclass(frozen=True)
class CloneGeometry:
    """Geometry reused from another drawing region under affine transforms."""

    source_region_id: str
    transforms: tuple[AffineTransform, ...]

    def __post_init__(self) -> None:
        if not self.source_region_id:
            raise ValueError("clone geometries require a source region ID")
        if not self.transforms:
            raise ValueError("clone geometries require at least one transform")


class ShapePrimitive:
    """Base class for strongly typed primitive-shape payloads.

    Concrete payloads deliberately come next: this contract does not
    admit an untyped ``kind`` plus parameter dictionary as an intermediate
    representation.
    """

    def __init__(self) -> None:
        if type(self) is ShapePrimitive:
            raise TypeError("shape primitives must use a concrete primitive variant")


@dataclass(frozen=True)
class CirclePrimitive(ShapePrimitive):
    """A circle expressed by its center and non-negative radius."""

    center: Point
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center))
        if not isfinite(float(self.radius)) or self.radius < 0:
            raise ValueError("circle radius must be finite and non-negative")


@dataclass(frozen=True)
class EllipsePrimitive(ShapePrimitive):
    """An axis-aligned ellipse expressed by its center and radii."""

    center: Point
    radius_x: float
    radius_y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center))
        if not all(isfinite(float(value)) and value >= 0 for value in (self.radius_x, self.radius_y)):
            raise ValueError("ellipse radii must be finite and non-negative")


@dataclass(frozen=True)
class RectanglePrimitive(ShapePrimitive):
    """An axis-aligned rectangle expressed by its top-left corner and size."""

    origin: Point
    width: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _point(self.origin))
        if not all(isfinite(float(value)) and value >= 0 for value in (self.width, self.height)):
            raise ValueError("rectangle dimensions must be finite and non-negative")


@dataclass(frozen=True)
class RoundedRectanglePrimitive(RectanglePrimitive):
    """An axis-aligned rectangle with independently addressable corner radii."""

    radius_x: float
    radius_y: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if not all(isfinite(float(value)) and value >= 0 for value in (self.radius_x, self.radius_y)):
            raise ValueError("rounded-rectangle radii must be finite and non-negative")
        if self.radius_x > self.width / 2 or self.radius_y > self.height / 2:
            raise ValueError("rounded-rectangle radii must not exceed half the corresponding dimension")


@dataclass(frozen=True)
class PolygonPrimitive(ShapePrimitive):
    """A closed straight-edge primitive with at least three vertices."""

    vertices: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("polygon primitives require at least three vertices")
        object.__setattr__(self, "vertices", tuple(_point(vertex) for vertex in self.vertices))


@dataclass(frozen=True)
class PrimitiveGeometry:
    """Geometry represented by one strongly typed shape primitive."""

    primitive: ShapePrimitive

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, ShapePrimitive):
            raise TypeError("primitive geometry requires a ShapePrimitive")


Geometry: TypeAlias = (
    PathGeometry
    | PathGeometryWithCutouts
    | PathGeometryWithSymmetry
    | PrimitiveGeometry
    | CloneGeometry
)


def geometry_from_svg_d(
    d: str,
    *,
    region_id: str,
    fill_rule: Literal["nonzero", "evenodd"] = "nonzero",
) -> PathGeometry | PathGeometryWithCutouts:
    """Parse closed SVG path data into typed, version-scoped path commands.

    ``region_id`` scopes generated path and command addresses as
    ``{region_id}.p0.c0``.  A single subpath becomes ``PathGeometry``; multiple
    subpaths require ``evenodd`` and become one outer path plus cutouts.
    """
    if not region_id:
        raise ValueError("SVG geometry requires a non-empty region ID")
    tokens = _SVG_PATH_TOKEN.findall(d)
    if not tokens:
        raise ValueError("SVG path data must not be empty")
    arity = {"M": 2, "L": 2, "Q": 4, "C": 6, "A": 7, "Z": 0}
    subpaths: list[list[tuple[str, list[float]]]] = []
    current: list[tuple[str, list[float]]] = []
    i = 0
    while i < len(tokens):
        kind = tokens[i]
        if kind not in arity:
            raise ValueError(f"expected an SVG path command, got {kind!r}")
        if kind == "M" and current:
            subpaths.append(current)
            current = []
        i += 1
        values = [float(value) for value in tokens[i:i + arity[kind]]]
        if len(values) != arity[kind]:
            raise ValueError(f"incomplete SVG {kind} command")
        i += arity[kind]
        current.append((kind, values))
    if current:
        subpaths.append(current)
    paths = tuple(
        PathGeometry(path=_closed_path_from_svg_commands(commands, f"{region_id}.p{path_index}"))
        for path_index, commands in enumerate(subpaths)
    )
    if len(paths) == 1:
        return paths[0]
    if fill_rule != "evenodd":
        raise ValueError("multiple closed SVG subpaths require evenodd fill_rule")
    return PathGeometryWithCutouts(outer=paths[0], cutouts=paths[1:])


def _closed_path_from_svg_commands(
    raw_commands: list[tuple[str, list[float]]],
    path_id: str,
) -> ClosedPath:
    if not raw_commands or raw_commands[0][0] != "M" or raw_commands[-1][0] != "Z":
        raise ValueError("typed paths require one explicit M and terminal Z")
    commands: list[PathCommand] = []
    for index, (kind, values) in enumerate(raw_commands[:-1]):
        command_id = f"{path_id}.c{index}"
        if kind == "M":
            if index != 0:
                raise ValueError("closed SVG subpaths may contain only one M command")
            commands.append(MoveCommand(command_id, (values[0], values[1])))
        elif kind == "L":
            commands.append(LineCommand(command_id, (values[0], values[1])))
        elif kind == "Q":
            commands.append(QuadraticCommand(command_id, (values[2], values[3]), (values[0], values[1])))
        elif kind == "C":
            commands.append(
                CubicCommand(
                    command_id,
                    (values[4], values[5]),
                    (values[0], values[1]),
                    (values[2], values[3]),
                )
            )
        elif kind == "A":
            commands.append(
                ArcCommand(
                    command_id,
                    (values[5], values[6]),
                    values[0],
                    values[1],
                    values[2],
                    bool(values[3]),
                    bool(values[4]),
                )
            )
        else:
            raise ValueError(f"unsupported SVG command {kind!r}")
    return ClosedPath(path_id, tuple(commands))
