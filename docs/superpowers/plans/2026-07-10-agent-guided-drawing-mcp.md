# Agent-Guided Drawing MCP Implementation Plan

> **Architecture amendment (implemented):** the earlier `DrawingScene` /
> `RenderTarget` executor described below has been superseded. A drawing version
> caches only `tuple[VectorRegion, ...]`; `drawing_id` and `source_regions` are
> first-class `VectorRegion` fields, child handles are derived as `r1-<id>`, and
> SVG/report/preview are on-demand projections. The task history remains below
> for implementation provenance.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace logo-specific MCP tools with trace_drawing, which either one-shot auto-idealizes or retains an interactive trace for immutable branching refinements.

**Architecture:** A pure drawing_trace.py produces the stable trace artifact. drawing_state.py owns the session-scoped version tree. drawing_plan.py parses and validates agent input, and drawing_refine.py creates scenes from existing Shape, Fill, and SVG emitters. mcp_server.py only resolves images, injects session context, and maps domain results to MCP responses.

**Tech Stack:** Python 3.12+, numpy, scipy, scikit-image, Pillow, skia-python, FastMCP, Pydantic through vectoreyes[server], pytest.

## Global Constraints

- Remove idealize_logo and idealize_logo_data; rename render_idealized_logo to render_drawing.
- trace_drawing with refine="interactive" runs palette segmentation, contours, and pre-simplified path fitting only; no primitives, symmetry, merging, fill inference, or optimizer passes.
- trace_drawing with refine="auto" runs the existing automatic idealizer once and does not allocate drawing state.
- Trace options are exactly max_colors, min_region_size, trace_level, simplify_tolerance, curve_tolerance, and curve_type. Image crop, resize, and alpha options remain PreprocessOpts.
- Interactive state is in-memory, attached to Context.session, and expires after 30 idle minutes. Auto mode has no retained state or version.
- refine_drawing(plan) receives drawing_id and base_version inside the plan, never an image or duplicate ID argument.
- Root is v0. New child versions append the atomically assigned child index: v0.0, v0.1, v0.1.0.
- Preserve TraceEngine.trace(rgb, options) -> TraceResult as the replacement seam for a future native or third-party tracer.
- Do not stage unrelated existing working-tree changes.

---

## File structure

| File | Responsibility |
| --- | --- |
| src/vectoreyes/drawing_trace.py | Trace types, deterministic Python TraceEngine, command serialization, and region map. |
| src/vectoreyes/drawing_state.py | Session-scoped drawing store, idle expiry, and immutable version tree. |
| src/vectoreyes/drawing_plan.py | Plan dataclasses, parsing, and JSON-pointer validation errors. |
| src/vectoreyes/drawing_refine.py | Primitive/path fitting, fills, paint order, SVG, and report generation. |
| src/vectoreyes/mcp_server.py | Pydantic MCP inputs, auto/interactive dispatch, and trace_drawing/refine_drawing handlers. |
| tests/test_drawing_trace.py | Trace options, IDs, paths, and region-map tests. |
| tests/test_drawing_state.py | Session isolation, TTL, and version-tree tests. |
| tests/test_drawing_plan.py | DSL parsing and validation tests. |
| tests/test_drawing_refine.py | Scene execution, path operations, fills, and immutability tests. |
| tests/test_mcp_server.py | MCP schema and same-session end-to-end tests. |

### Task 1: Stable trace artifact and Python TraceEngine

**Files:**

- Create: src/vectoreyes/drawing_trace.py
- Create: tests/test_drawing_trace.py
- Modify: src/vectoreyes/pipeline.py only if attach_coverage_field needs a reusable public helper.

**Interfaces:**

- Consumes: extract_palette, quantize, segment, region_contours, fit_path, and render_svg_doc.
- Produces: TraceResult for Tasks 2, 4, and 5.

- [ ] **Step 1: Write failing trace tests**

~~~python
def test_trace_keeps_small_region_using_absolute_size_only():
    image = np.full((100, 100, 3), 255, np.uint8)
    image[10:90, 10:90] = (0, 110, 240)
    image[4:8, 4:8] = (255, 100, 0)

    result = PythonTraceEngine().trace(image, TraceOptions(min_region_size=16))

    assert [region.id for region in result.regions] == ["r1", "r2"]


def test_trace_commands_are_scoped_by_region_and_subpath():
    region = PythonTraceEngine().trace(_annulus_image(), TraceOptions()).regions[0]

    assert region.trace_path.commands[0].id == "r1.p0.c0"
    assert any(command.id.startswith("r1.p1.") for command in region.trace_path.commands)


def test_region_map_labels_every_region():
    result = PythonTraceEngine().trace(_two_region_image(), TraceOptions())

    assert all(f">{region.id}<" in result.region_map_svg for region in result.regions)
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_drawing_trace.py -v  
Expected: FAIL with ModuleNotFoundError for vectoreyes.drawing_trace.

- [ ] **Step 3: Write the minimal trace implementation**

~~~python
@dataclass(frozen=True)
class TraceOptions:
    max_colors: int = 16
    min_region_size: int = 16
    trace_level: Literal["pixel", "subpixel"] = "pixel"
    simplify_tolerance: float = 1.5
    curve_tolerance: float = 1.0
    curve_type: Literal["quadratic", "cubic"] = "quadratic"

@dataclass(frozen=True)
class TraceCommand:
    id: str
    command: Literal["M", "L", "Q", "C", "Z"]
    values: tuple[float, ...]

@dataclass(frozen=True)
class TracePath:
    d: str
    fill_rule: Literal["nonzero", "evenodd"]
    commands: tuple[TraceCommand, ...]

@dataclass(frozen=True)
class TraceRegion:
    id: str
    source_label: int
    color: str
    mask: np.ndarray = field(compare=False, repr=False)
    contours: tuple[np.ndarray, ...] = field(compare=False, repr=False)
    trace_path: TracePath
    effective_trace_level: Literal["pixel", "subpixel"]

@dataclass(frozen=True)
class TraceResult:
    width: int
    height: int
    options: TraceOptions
    regions: tuple[TraceRegion, ...]
    region_map_svg: str

    def to_public_dict(self) -> dict[str, object]: ...

class TraceEngine(Protocol):
    def trace(self, rgb: np.ndarray, options: TraceOptions) -> TraceResult: ...

class PythonTraceEngine:
    def trace(self, rgb: np.ndarray, options: TraceOptions) -> TraceResult: ...
~~~

Call segment(quantized, min_area=options.min_region_size) directly; do not call _segment_image because it applies relative filtering. For subpixel, reuse the coverage guard and report pixel for a guarded region. Sort with (-area, bbox_y, bbox_x, color, source_label) before assigning r1, r2, and so on. Fit each contour using simplify_tolerance, curve_tolerance, and curve_type; parse M/L/Q/C/Z output into rN.pM.cK IDs. Build the map from translucent region paths and centered text labels.

- [ ] **Step 4: Run the test to verify it passes**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_drawing_trace.py -v  
Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/vectoreyes/drawing_trace.py src/vectoreyes/pipeline.py tests/test_drawing_trace.py
git commit -m "feat: add stable drawing trace artifact"
~~~

### Task 2: Session-scoped drawings and branching versions

**Files:**

- Create: src/vectoreyes/drawing_state.py
- Create: tests/test_drawing_state.py

**Interfaces:**

- Consumes: TraceResult from Task 1 and DrawingScene from Task 4.
- Produces: DrawingStore.create, DrawingStore.get, and DrawingStore.append for Task 5.

- [ ] **Step 1: Write failing state tests**

~~~python
def test_store_branches_from_any_retained_version(fake_clock):
    store = DrawingStore(now=fake_clock)
    session = object()
    drawing = store.create(session, _trace())

    first = store.append(session, drawing.id, "v0", plan={}, scene=_scene())
    second = store.append(session, drawing.id, "v0", plan={}, scene=_scene())
    child = store.append(session, drawing.id, "v0.1", plan={}, scene=_scene())

    assert (first.id, second.id, child.id) == ("v0.0", "v0.1", "v0.1.0")


def test_store_rejects_cross_session_and_expired_drawings(fake_clock):
    store = DrawingStore(now=fake_clock, idle_ttl_seconds=1800)
    owner, other = object(), object()
    drawing = store.create(owner, _trace())

    with pytest.raises(DrawingNotFound):
        store.get(other, drawing.id, "v0")
    fake_clock.advance(1801)
    with pytest.raises(DrawingNotFound):
        store.get(owner, drawing.id, "v0")
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_drawing_state.py -v  
Expected: FAIL with ModuleNotFoundError for vectoreyes.drawing_state.

- [ ] **Step 3: Write the minimal state implementation**

~~~python
class DrawingNotFound(Exception):
    error_code = "DRAWING_NOT_FOUND"

@dataclass(frozen=True)
class DrawingVersion:
    id: str
    parent_id: str | None
    plan: Mapping[str, object] | None
    scene: object | None
    label: str | None

@dataclass
class DrawingState:
    id: str
    trace: TraceResult
    versions: dict[str, DrawingVersion]
    child_counts: dict[str, int]
    last_access: float

class DrawingStore:
    def __init__(self, *, idle_ttl_seconds: float = 1800,
                 now: Callable[[], float] = time.monotonic): ...
    def create(self, session: object, trace: TraceResult) -> DrawingState: ...
    def get(self, session: object, drawing_id: str,
            version_id: str) -> tuple[DrawingState, DrawingVersion]: ...
    def append(self, session: object, drawing_id: str, base_version: str,
               *, plan: Mapping[str, object], scene: object,
               label: str | None = None) -> DrawingVersion: ...
~~~

Use dict[object, dict[str, DrawingState]], generate drw_ IDs with secrets.token_urlsafe(18), and call _evict_expired before every public method. Root is immutable v0. get and append refresh the 30-minute sliding expiry. append increments only the selected parent’s child counter. Raise DrawingNotFound for every owner, drawing, or version miss.

- [ ] **Step 4: Run the test to verify it passes**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_drawing_state.py -v  
Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/vectoreyes/drawing_state.py tests/test_drawing_state.py
git commit -m "feat: store branching drawing versions"
~~~

### Task 3: Refinement DSL parser and validator

**Files:**

- Create: src/vectoreyes/drawing_plan.py
- Create: tests/test_drawing_plan.py

**Interfaces:**

- Consumes: TraceResult and a base DrawingScene.
- Produces: DrawingPlan, parse_plan, validate_plan, and PlanValidationError for Tasks 4–5.

- [ ] **Step 1: Write failing plan tests**

~~~python
def test_plan_accepts_path_group_fit_break_and_close():
    plan = parse_plan({
        "version": "vectoreyes.plan.v1", "drawing_id": "drw_x",
        "base_version": "v0", "ops": [{
            "op": "set_geometry", "target": "r1",
            "geometry": {"type": "path", "ops": [
                {"op": "group", "id": "s1",
                 "commands": ["r1.p0.c1", "r1.p0.c2"]},
                {"op": "fit", "target": "s1", "type": "quadratic"},
                {"op": "close"},
            ]},
        }],
    })

    validate_plan(plan, _trace(), _scene())


def test_plan_reports_pointer_for_non_contiguous_path_commands():
    plan = parse_plan(_path_plan(["r1.p0.c1", "r1.p0.c3"]))

    with pytest.raises(PlanValidationError,
                       match="/ops/0/geometry/ops/0/commands/1"):
        validate_plan(plan, _trace(), _scene())
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_drawing_plan.py -v  
Expected: FAIL with ModuleNotFoundError for vectoreyes.drawing_plan.

- [ ] **Step 3: Write the parser and validator**

~~~python
@dataclass(frozen=True)
class DrawingPlan:
    version: Literal["vectoreyes.plan.v1"]
    drawing_id: str
    base_version: str
    label: str | None
    ops: tuple[object, ...]

class PlanValidationError(ValueError):
    def __init__(self, pointer: str, message: str): ...

def parse_plan(payload: Mapping[str, object]) -> DrawingPlan: ...
def validate_plan(plan: DrawingPlan, trace: TraceResult,
                  scene: object) -> None: ...
~~~

Support scene-level merge, split, detect_primitives, detect_symmetry,
detect_clones, set_symmetry, clone, set_geometry, set_fill, and set_z_order.
`detect_*` is the existing automatic pass, globally or with an optional singular
target; the paired explicit operation carries the agent-supplied relationship.
Nested path operations are group, fit(line|quadratic|cubic|keep), simplify,
seams, break, and close. Reject unknown targets, duplicate IDs, non-finite
numbers, invalid fill fields, omitted or repeated z-order targets, and missing
path-operation dependencies. A path group may contain only contiguous
non-M/non-Z commands from one trace subpath. PlanValidationError must carry an
RFC 6901 JSON pointer rooted at /ops.

- [ ] **Step 4: Run the test to verify it passes**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_drawing_plan.py -v  
Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/vectoreyes/drawing_plan.py tests/test_drawing_plan.py
git commit -m "feat: validate drawing refinement plans"
~~~

### Task 4: Immutable refinement executor

**Files:**

- Create: src/vectoreyes/drawing_refine.py
- Create: tests/test_drawing_refine.py
- Modify: src/vectoreyes/fit.py only to add explicit circle, ellipse, rect, or polygon fit helpers missing from the current API.

**Interfaces:**

- Consumes: validated DrawingPlan, TraceResult, and base DrawingScene.
- Produces: DrawingScene(targets, svg, report) for Tasks 2 and 5.

- [ ] **Step 1: Write failing executor tests**

~~~python
def test_refine_forces_circle_and_flat_fill():
    trace = PythonTraceEngine().trace(_disk_image(), TraceOptions())
    scene = refine(trace, root_scene(trace),
                   parse_plan(_circle_plan("r1", "#FF6600")))

    assert '<circle id="r1"' in scene.svg
    assert 'fill="#FF6600"' in scene.svg


def test_path_refinement_keeps_trace_immutable():
    trace = PythonTraceEngine().trace(_leaf_image(), TraceOptions())
    original = trace.regions[0].trace_path.d

    refine(trace, root_scene(trace), parse_plan(_line_path_plan(trace)))

    assert trace.regions[0].trace_path.d == original


def test_refine_uses_declared_z_order():
    trace = PythonTraceEngine().trace(_two_region_image(), TraceOptions())
    scene = refine(trace, root_scene(trace), parse_plan(_z_order_plan(trace)))

    assert scene.svg.index('id="r2"') < scene.svg.index('id="r1"')
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_drawing_refine.py -v  
Expected: FAIL with ModuleNotFoundError for vectoreyes.drawing_refine.

- [ ] **Step 3: Write the scene executor**

~~~python
@dataclass(frozen=True)
class RenderTarget:
    id: str
    source_regions: tuple[str, ...]
    shape: Shape
    fill: Fill
    z: int

@dataclass(frozen=True)
class DrawingScene:
    targets: tuple[RenderTarget, ...]
    svg: str
    report: Mapping[str, object]

def root_scene(trace: TraceResult) -> DrawingScene: ...
def refine(trace: TraceResult, base: DrawingScene,
           plan: DrawingPlan) -> DrawingScene: ...
~~~

root_scene emits one path target and FlatFill per trace region. set_geometry calls a requested primitive fitter, never recognize_primitive; reject residual above 0.06 without silently substituting another geometry. For path operations, sample selected pre-simplified trace commands, fit the requested type, preserve keep commands, maintain tangent continuity unless break appears, and append Z only for close. Convert explicit fills to existing FlatFill, LinearGradientFill, RadialGradientFill, or RasterFill. Use set_z_order as final order and existing SVG/fill emitters for output. Report target source regions, geometry, fill kind, and residual. Never mutate TraceResult or base scene.

- [ ] **Step 4: Run the test to verify it passes**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_drawing_refine.py -v  
Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/vectoreyes/drawing_refine.py src/vectoreyes/fit.py tests/test_drawing_refine.py
git commit -m "feat: execute drawing refinement plans"
~~~

### Task 5: Stateful MCP tools

**Files:**

- Modify: src/vectoreyes/mcp_server.py
- Modify: tests/test_mcp_server.py

**Interfaces:**

- Consumes: Tasks 1–4 plus ImageRef, PreprocessOpts, _render_preview_png, Context.session, and the existing idealize helper.
- Produces: trace_drawing and refine_drawing; removes logo-specific MCP tools.

- [ ] **Step 1: Write failing MCP tests**

~~~python
def test_stdio_trace_then_create_sibling_versions():
    async def workflow():
        params = StdioServerParameters(command="uv", args=["run", "vectoreyes-mcp"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                traced = await session.call_tool(
                    "trace_drawing", {"image": {"data_uri": _png_uri()}})
                root = traced.structuredContent
                first = await session.call_tool(
                    "refine_drawing", {"plan": _circle_plan(
                        root["drawing_id"], "v0", "flat")})
                second = await session.call_tool(
                    "refine_drawing", {"plan": _circle_plan(
                        root["drawing_id"], "v0", "orange")})
                return root, first.structuredContent, second.structuredContent

    root, first, second = asyncio.run(workflow())

    assert root["version"] == "v0"
    assert (first["version"], second["version"]) == ("v0.0", "v0.1")


def test_refine_schema_exposes_plan_header_fields():
    tools = asyncio.run(mcp.list_tools())
    schema = {tool.name: tool.inputSchema for tool in tools}["refine_drawing"]
    defs = schema.get("$defs", {})
    plan = schema["properties"]["plan"]
    if "$ref" in plan:
        plan = defs[plan["$ref"].split("/")[-1]]

    assert {"version", "drawing_id", "base_version", "ops"} <= set(plan["properties"])


def test_trace_auto_returns_svg_without_live_drawing_state():
    result = asyncio.run(mcp.call_tool(
        "trace_drawing", {"image": {"data_uri": _png_uri()},
                          "options": {"refine": "auto"}}))

    assert result.structuredContent["svg"].startswith("<svg ")
    assert "drawing_id" not in result.structuredContent
    assert "idealize_logo" not in {tool.name for tool in asyncio.run(mcp.list_tools())}
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_mcp_server.py -k "trace_drawing or refine_drawing" -v  
Expected: FAIL because neither tool exists.

- [ ] **Step 3: Write MCP models and handlers**

~~~python
class TraceDrawingOptions(BaseModel):
    refine: Literal["auto", "interactive"]
    max_colors: int = Field(16, ge=2, le=256)
    min_region_size: int = Field(16, ge=1)
    trace_level: Literal["pixel", "subpixel"] = "pixel"
    simplify_tolerance: float = Field(1.5, gt=0)
    curve_tolerance: float = Field(1.0, gt=0)
    curve_type: Literal["quadratic", "cubic"] = "quadratic"
    preprocess: PreprocessOpts = Field(default_factory=PreprocessOpts)

class RefinementPlanInput(BaseModel):
    version: Literal["vectoreyes.plan.v1"]
    drawing_id: str = Field(min_length=1)
    base_version: str = Field(pattern=r"^v\d+(?:\.\d+)*$")
    label: str | None = Field(None, max_length=200)
    ops: list[dict[str, object]]

def trace_drawing(image: ImageRef, options: TraceDrawingOptions | None = None,
                  ctx: Context = ...) -> CallToolResult: ...
def refine_drawing(plan: RefinementPlanInput,
                   ctx: Context = ...) -> CallToolResult: ...
~~~

Create module-global DrawingStore and PythonTraceEngine. Delete idealize_logo, idealize_logo_data, idealize_logo_file, idealize_logo_bytes, and their models/tests; rename render_idealized_logo to render_drawing. trace_drawing always reuses resolve_image and preprocess_image. For refine="auto", map max_colors, trace_level, simplify_tolerance, curve_tolerance, curve_type, and min_region_size to Options, call idealize once, and return only SVG, preview, and diagnostics; do not create a DrawingState. Add min_region_size to Options and use it as _segment_image's absolute component floor while preserving the automatic pipeline's existing relative filter. For refine="interactive", trace, store under ctx.session, and return public trace data plus drawing_id/v0. refine_drawing parses and validates, gets the base version, executes refinement, appends the child version, creates a best-effort preview, and returns version, label, SVG, report, and preview. Translate DrawingNotFound to [DRAWING_NOT_FOUND] and PlanValidationError to [INVALID_PLAN] ToolError messages. trace_drawing declares openai/fileParams ["image"]; refine_drawing has no file parameters. Both tools use the existing widget URI and return identical structuredContent and JSON text.

- [ ] **Step 4: Run the test to verify it passes**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_mcp_server.py -k "trace_drawing or refine_drawing" -v  
Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/vectoreyes/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add trace and branching refine tools"
~~~

### Task 6: Documentation and end-to-end acceptance

**Files:**

- Modify: docs/mcp.md
- Modify: README.md
- Modify: tests/test_drawing_refine.py

**Interfaces:**

- Consumes: completed trace/refine workflow.
- Produces: documented agent workflow and a regression fixture for sibling alternatives.

- [ ] **Step 1: Write the failing acceptance test**

~~~python
def test_folded_logo_can_branch_and_emit_native_dots():
    trace = PythonTraceEngine().trace(_folded_logo_image(),
                                      TraceOptions(min_region_size=8))
    flat = refine(trace, root_scene(trace),
                  parse_plan(_flat_dot_plan("drw_test", "v0")))
    gradient = refine(trace, root_scene(trace),
                      parse_plan(_gradient_dot_plan("drw_test", "v0")))

    assert all(target.shape.kind == "path" for target in root_scene(trace).targets)
    assert '<circle id="r' in flat.svg
    assert "linearGradient" in gradient.svg
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_drawing_refine.py -k folded -v  
Expected: FAIL until Tasks 1–4 are complete.

- [ ] **Step 3: Document exact agent usage**

Add this exchange to docs/mcp.md and a concise MCP section to README.md:

~~~text
1. Call trace_drawing once with an image reference and trace options.
2. Inspect regions and region_map_svg; retain drawing_id and v0.
3. Send refine_drawing plans containing drawing_id, base_version, label, and ops.
4. Branch from any returned version while the originating MCP session remains active.
5. Re-trace after DRAWING_NOT_FOUND; state expires after 30 idle minutes.
~~~

Document every trace option, all four region operations, all four path operations, semantic group versus boolean union, v0.1.0 version paths, and the TraceEngine substitution boundary.

- [ ] **Step 4: Run acceptance and regression tests**

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_drawing_trace.py tests/test_drawing_state.py tests/test_drawing_plan.py tests/test_drawing_refine.py tests/test_mcp_server.py -q  
Expected: PASS.

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/optimizer tests/test_pipeline.py tests/test_fit.py tests/test_mcp_image.py -q  
Expected: PASS; auto mode preserves one-shot automatic idealization and secure image handling remains unchanged.

- [ ] **Step 5: Commit**

~~~bash
git add README.md docs/mcp.md tests/test_drawing_refine.py
git commit -m "docs: describe agent-guided drawing workflow"
~~~

## Plan self-review

- **Coverage:** Task 1 covers trace options, trace paths, deterministic IDs, region maps, and the engine seam. Task 2 covers session ownership, sliding expiry, and branching versions. Tasks 3–4 cover the full MVP DSL, immutable scenes, primitives, paths, fills, z-order, and reports. Task 5 covers the MCP contract; Task 6 covers documentation and end-to-end branching acceptance.
- **Out of scope:** CLI, portable traces, persistence, boolean union/split, clones, symmetry, congruency, ribbons, arcs, and automatic candidate selection are absent from every task.
- **Consistency:** TraceResult enters DrawingState; DrawingPlan validates before refine; DrawingScene becomes an immutable DrawingVersion; MCP handlers only map domain objects to protocol responses.
- **Placeholder scan:** No deferred implementation markers or unspecified error paths remain.
