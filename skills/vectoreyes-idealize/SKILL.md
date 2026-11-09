---
name: vectoreyes-idealize
description: >-
  Turn a raster logo image into a clean, editable, exactly-symmetric SVG using
  the vectoreyes CLI. Use when the user wants to vectorize, idealize, clean up,
  trace, or convert a logo/icon/emblem image (PNG/JPG) to SVG — including an
  image the agent just generated. Best on flat-color marks, app icons, and
  simple logos.
---

# Idealize a logo into SVG with vectoreyes

vectoreyes is a deterministic raster-logo → SVG idealizer. This skill takes a
raster image — one the user supplies, or one you just generated — and produces a
clean, editable, symmetric SVG by running the `vectoreyes` command-line tool.

The handoff is a **file path**, not image bytes: you save the raster to a file,
then point the CLI at it. (A path is short text you can pass between steps; raw
image bytes are not.)

## Step 1 — Decide how to run vectoreyes

vectoreyes is published on PyPI. Pick the invocation that fits the environment;
you will reuse it as `<VM>` in Step 3.

**Preferred — `uvx` (no install, ephemeral).** If `uv` is available
(`uv --version` succeeds), run vectoreyes with no persistent install — `uvx`
fetches and caches it on first use, later runs reuse the cache. **Prefer the
`scoring` extra**, which enables render-ΔE candidate scoring for the best output:

```bash
# <VM> = full, with scoring (preferred)
uvx --from "vectoreyes[scoring]" vectoreyes
# <VM> = minimal (no resvg-py; cascade-priority pick, warns once)
uvx vectoreyes
```

**Fallback — `pip install` (persistent).** If `uv` is not available but Python
≥ 3.12 and `pip` are, install once, then call `vectoreyes` directly:

```bash
pip install "vectoreyes[scoring]"   # or: pip install vectoreyes   (minimal)
# <VM> = vectoreyes
```

If the scoring extra can't be fetched (a locked-down sandbox with no `resvg-py`
wheel), drop `[scoring]` — vectoreyes still works, just with the cascade pick.

> Notes
> - Requires Python ≥ 3.12. vectoreyes's own wheel is pure-Python; its
>   dependencies and the `scoring` extra ship prebuilt manylinux/musllinux/
>   macOS/Windows wheels, so this works in any networked environment.
> - **No PyPI?** Install from the git source instead. Minimal:
>   `uvx --from "git+https://github.com/pradeepmouli/vectoreyes" vectoreyes`.
>   With scoring, use the PEP 508 form so the extra is carried:
>   `uvx --from "vectoreyes[scoring] @ git+https://github.com/pradeepmouli/vectoreyes" vectoreyes`
>   (same `"<spec> @ git+…"` form for `pip install`).

## Step 2 — Get the raster onto disk

You need a PNG/JPG file path. Two cases:

- **The user already gave you an image file** → use that path; skip to Step 3.
- **You are generating the logo** (image-gen tool / model) → generate it, then
  **save the bytes to a file**, e.g. `/tmp/vm-logo.png`. Some hosts already
  expose the generated image as a file in the working sandbox; if so, use that
  path. Otherwise write the bytes out explicitly before Step 3.

The CLI reads the file from disk, so the generator and `vectoreyes` must share a
filesystem — true when both run locally, or both run inside the same sandbox.

## Step 3 — Run vectoreyes

Use the `<VM>` invocation you chose in Step 1:

```bash
# preferred (uvx + scoring):
uvx --from "vectoreyes[scoring]" vectoreyes /tmp/vm-logo.png -o /tmp/vm-logo.svg --colors 16
# or, after a pip install:
vectoreyes /tmp/vm-logo.png -o /tmp/vm-logo.svg --colors 16
```

The SVG is written to `-o`; omit `-o` to print it to stdout (handy when you want
to read the SVG back to the user directly).

Useful flags (all optional):

| Flag | Default | Meaning |
|------|---------|---------|
| `-o, --output` | stdout | write the SVG to this path |
| `--colors` | 16 | maximum palette colors |
| `--epsilon` | 1.5 | primitive/polygon recognition tolerance, px |
| `--max-error` | 1.0 | Bézier fit tolerance, px |
| `--flatten` | off | emit plain paths instead of native primitives + `<use>` |
| `--no-symmetry` | off | disable symmetry detection |

Guidance: start with defaults. Raise `--colors` for richer marks; lower it to
force a flatter palette. Raise `--epsilon` if curves are coming out too busy.
Use `--flatten` only if the consumer can't handle `<use>`/native primitives.

## Step 3b — Explore looks across the geometry space (optional)

When the user wants to **compare options or pick a look**, `--variants` idealizes
the mark across an `epsilon × max_error` grid and writes the set to a directory —
no need to hand-tune flags one run at a time:

```bash
# 3×3 default grid -> ./<stem>-variants/
<VM> /tmp/vm-logo.png --variants --out-dir /tmp/vm-looks
# custom axes (any grid):
<VM> /tmp/vm-logo.png --variants --out-dir /tmp/vm-looks --epsilons 0.5,2,4 --max-errors 0.5,2
# also overlay the detected symmetry axis on each tile (a diagnostic view):
<VM> /tmp/vm-logo.png --variants --axes --out-dir /tmp/vm-looks
```

It writes one `variant-e<ε>-m<max_error>.svg` per cell, a `manifest.json` (each
variant's params **and the fitter strategies it actually used**), and — when the
`scoring` extra is installed — an annotated `contact-sheet.png`. Show the user the
contact sheet (or summarize the manifest); let them pick a cell; then deliver that
variant's SVG as the result.

The `contact-sheet.png` here is a **real artifact the tool produced** — that is
exactly what Step 4's "don't fabricate a contact sheet" rule does NOT forbid.
Show vectoreyes's real sheet; never invent one.

## Step 4 — Deliver the result

- **For a single run, the deliverable is the SVG itself** — report it and its
  output path, nothing more. vectoreyes emits one SVG string; do **not** fabricate
  optimization stats, file-size tables, color-swatch grids, "production ready"
  checklists, or a multi-panel contact sheet around it. Those are *invented*, not
  produced by the tool, and only add noise. (The real `contact-sheet.png` from
  `--variants` is a different thing — it's an actual tool output; show that.)
- **If you can't run vectoreyes, do not try to draw the SVG yourself.** Never
  hand-author, guess, or hallucinate an output. If the tool didn't actually run
  (not installed, no network, an error, or you only have the input image), say so
  and stop — a fabricated SVG is worse than none, and only vectoreyes's real
  output is meaningful here.
- To preview, open the file (`open /tmp/vm-logo.svg` on macOS,
  `xdg-open` on Linux) or read the SVG text back to the user — an `<svg>`
  document renders directly in any browser or markdown preview.

## Tips

- vectoreyes is **deterministic**: same input + flags → same SVG every time.
- It targets flat-color logos/icons/emblems. Photographic or heavily-gradient
  images are out of scope and will produce poor results.
- If you generated the image, keep the prompt simple and the mark flat — that is
  what idealizes cleanly.
