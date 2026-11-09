# vectorEyes

![vectorEyes](src/vectoreyes/assets/vectoreyes-lockup.svg)

> Deterministic logo idealizer — turn a *rendered* (raster) logo into a clean,
> editable, symmetry and shape-preserving SVG.

Where conventional tracers (`vtracer`, `potrace`, `autotrace`) chase pixel edges
and emit anonymous `<path>` soup, **vectorEyes recognizes structure**: it
collapses anti-aliasing to a true palette, detects symmetry, and fits *ideal
primitives* (`<ellipse>`, `<rect>`, regular polygons) to regions — falling back
to fitted Bézier paths only where no primitive matches. The output SVG doubles as
a human-editable parametric model.

No model/LLM in the loop. Every stage is a known, deterministic algorithm.
The Python package and command-line tool are both named `vectoreyes`.
The MCP server command is `vectoreyes-mcp`; its tools use the `vectoreyes_`
prefix. Saved refinement plans for this build use `vectoreyes.plan.v1`.

### Agent plugin

For the agent-guided image → trace → plan refinement → visual evaluation
workflow, see [plugin packaging and installation](docs/plugins.md).
The shared bundle supports portable Agent Plugins, Codex and Claude Code;
its skills are generated from the same packaged instructions served by MCP.
The deterministic engine remains usable independently of an agent.

## Status

🌱 **v1 pipeline working.** The deterministic A0→B→C pipeline idealizes the
Daikonic mark end-to-end (SSIM 0.98 / mean ΔE 0.006 vs. source) into a 1.9 KB
structured SVG with exact bilateral symmetry. See the design spec and plan:
- spec: [`docs/superpowers/specs/2026-06-04-vectormark-logo-idealizer-design.md`](docs/superpowers/specs/2026-06-04-vectormark-logo-idealizer-design.md)
- plan: [`docs/superpowers/plans/2026-06-04-vectormark-v1.md`](docs/superpowers/plans/2026-06-04-vectormark-v1.md)

```bash
uv sync --extra dev
uv run vectoreyes path/to/logo.png -o out.svg
uv run pytest
```

### Variant matrix

Explore the geometric design space in one shot — `idealize` swept across an
`epsilon × max_error` grid:

```bash
uv run vectoreyes logo.png --variants                       # 3×3 grid -> ./logo-variants/
uv run vectoreyes logo.png --variants --out-dir ./looks/
uv run vectoreyes logo.png --variants --epsilons 0.5,2,4 --max-errors 0.5,2
```

Add `--axes` to draw each detected symmetry axis over the contact-sheet tiles —
a quick visual check of what the symmetry detector found.

Writes `variant-e<ε>-m<max_error>.svg` per cell and a `manifest.json` listing each
variant's params and the fitter strategies it used. With the `scoring` extra
installed it also renders an annotated `contact-sheet.png` (one tile per cell,
labelled with its strategy histogram).

MCP integration is available for AI clients that can call local stdio servers
(the server lives in the optional `server` extra; scored selection needs the
`scoring` extra — without it, selection falls back to the cascade pick):

```bash
uv sync --extra server --extra scoring
npm --prefix integrations/mcp-app install
npm --prefix integrations/mcp-app run build
uv run vectoreyes-mcp
```

See [`docs/mcp.md`](docs/mcp.md) for client configuration and hosted app notes.

## Pipeline (v1)

```
A0  color optimization   robust palette (OKLab, ΔE-merge); AA halo discarded as noise
B   trace + clean        vtracer per layer; drop background plate
C   idealize             symmetry axis (PCA + reflection gate) → fundamental domain
                         → contour → RDP → primitive-first / Bézier-fallback
                         → geometry-level mirror (<use>)
→   structured SVG       native shapes + <use>; optional --flatten to paths
```

v1 targets flat-color segmented marks. v1.1 adds gradient normalize/re-apply.

## Prototype → core

Built first in **Python** (numpy/scipy/scikit-image/shapely/Pillow + vtracer),
with a pure, IO-free pipeline so the eventual **Rust core** (npm via napi-rs,
PyPI via PyO3/maturin, WASM for browser) is a mechanical port.

## License

MIT
