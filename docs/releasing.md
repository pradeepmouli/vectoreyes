# Releasing vectormark to PyPI

vectormark publishes via **PyPI Trusted Publishing** (OIDC) — GitHub Actions
proves the workflow's identity to PyPI directly, so there is **no API token** in
this repository. The release workflow lives at `.github/workflows/release.yml`
and runs on any `v*` tag.

## One-time setup (maintainer)

You only do this once, on pypi.org. It cannot be automated from here — it ties
your PyPI account to this repo.

1. Create the project's trusted publisher **before the first upload** using a
   *pending publisher* (PyPI lets you pre-register a project that does not exist
   yet):
   - Log in to <https://pypi.org> → **Account → Publishing → Add a pending publisher**.
   - **PyPI Project Name:** `vectormark`
   - **Owner:** `pradeepmouli`
   - **Repository name:** `vectormark`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`
2. In this GitHub repo: **Settings → Environments → New environment** → name it
   `pypi`. (Optional: add a required reviewer so a human approves each publish.)

That's it. The `id-token: write` permission and the `environment: pypi` binding
in the workflow match what you registered above.

## Cutting a release

1. Bump `version` in `pyproject.toml` (PyPI versions are immutable — a number can
   never be reused, so never re-tag an already-published version).
2. Commit on `master`.
3. Tag and push:
   ```bash
   git tag v0.0.1
   git push origin v0.0.1
   ```
4. The `Release` workflow builds the MCP review UI and sdist + wheel, verifies
   the installed wheel's widget outside the checkout, runs `twine check`, and
   publishes to PyPI. Watch it under the repo's **Actions** tab.

## Local build check (optional, before tagging)

```bash
npm --prefix integrations/mcp-app ci
npm --prefix integrations/mcp-app run build
uv build                 # -> platform/Python-specific wheel + .tar.gz
uvx twine check dist/*   # validate metadata renders on PyPI
```

VectorMark wheels compile the drawing engine, tracer, fitter, and optimizer with
Cython, including the boundary-distance kernel. CLI and MCP entry points, MCP
image resolution, and package initializers remain Python. Wheels are specific
to the build Python and platform. Source builds require a C compiler;
PEP 517 installs Cython and setuptools in the isolated build environment.
`uv sync --extra dev` compiles only the kernel for editable development; other
engine modules use their canonical Python sources so edits are immediately
visible. Release wheels compile those same sources, not a second implementation.
There is no separate Python fallback for the kernel. The MCP surface remains Python, and the
compiled widget HTML is included as package data as before. Native dependencies
such as Skia and the optional renderer also constrain supported platforms.

Verify release behavior against an installed wheel outside the checkout as well
as in editable mode. Native compilation can change runtime type enforcement,
so source-mode tests alone are not a release gate.

The built MCP App HTML is required when building a distribution. Hatch includes
it at `vectormark/assets/mcp-app.html` in the wheel and retains the frontend
source and built HTML in the source distribution. Missing HTML fails the build;
rebuild it after frontend changes. End users do not need Node.js.

The source distribution includes the Cython source and `hatch_build.py`, and
uses an explicit project-file allowlist so local graph
databases, nested worktrees, and `node_modules` are not published accidentally.
