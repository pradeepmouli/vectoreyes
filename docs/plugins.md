# vectorEyes agent plugin

One generated bundle supports the portable Agent Plugins format, Codex's
compatibility manifest, and Claude Code's manifest. All hosts receive the same
`designer-reconstruction` skill and references from the Python package. The
generator reads the version from `pyproject.toml`; do not edit generated files
under `integrations/plugin/`.

## Build and install

From this repository:

```sh
python scripts/build_agent_plugin.py --archive dist/vectoreyes-plugin.zip
python scripts/build_agent_plugin.py --check
claude plugin validate integrations/plugin
claude --plugin-dir ./integrations/plugin
```

For Codex, use the repository's `.agents/plugins/marketplace.json` entry in the
plugin browser, or import the generated ZIP where local plugin import is
supported. Installing/enabling is an explicit host action; building the ZIP
does not register or publish a plugin. The bundle includes both host manifests;
runtime compatibility still needs a smoke test in each supported host version.

The default bundle launches `vectoreyes-mcp` over stdio. Install a compatible
vectorEyes wheel with server/scoring extras into a dedicated environment and
make that environment's `vectoreyes-mcp` available on the host's PATH before
enabling. The plugin never silently downloads a runtime or changes Python
environments. It contains no platform-specific native binaries. Use the built
wheel matching the user's Python/OS/architecture; development-source installs
are a separate explicit choice.

To connect to an already running service instead:

```sh
python scripts/build_agent_plugin.py --output build/vectoreyes-http-plugin \
  --endpoint http://127.0.0.1:8000/mcp \
  --archive dist/vectoreyes-http-plugin.zip
```

Remote endpoints require HTTPS; loopback HTTP is allowed for local development.
Never package credentials, signed URLs, or a temporary private tunnel URL.
The generator translates portable `streamable-http` to the host compatibility
configuration's `http` transport. Choose one connection per workflow; drawing
handles from separate stdio and HTTP server processes are not interchangeable.
The local evaluation runner requires a reachable HTTP service; direct plugin
MCP calls can use stdio. An MCP connection is not supplied by the image model;
use whatever image-generation capability the host actually provides or accept
a user PNG.

## Workflow and UI

The plugin replaces duplicated agent instructions and manual option entry, not
the engine or the need for visual evidence. Skills drive whole-drawing review,
region worklists, bounded alternative fits and independent shape/fill verdicts.
MCP tools execute plans. The optional offline report provides artwork downloads,
maps, comparison crops, and selected-version parameters without requiring the
in-chat MCP widget. Keep the existing widget available until equivalent host
delivery/export and review flows have been verified; no widget is removed here.

The report distinguishes exact submitted requests, server-reported trace
options, per-action retained fit settings and output policy. Absent provenance
is labeled unavailable, never reconstructed from current defaults. A failed
trial has submitted parameters but no invented effective candidate settings.

## Distribution boundaries

`integrations/plugin/` is a generated, inspectable repository install target;
the ZIP is self-contained and has no symlinks to the source checkout. Run the
generator after changing the canonical skill and use `--check` to detect drift.
The obsolete `integrations/claude-plugin` pipeline has been removed.

The legacy Claude marketplace and Codex repository marketplace point to this
same generated directory. No public registry submission, installation, server
restart, or production deployment is performed by the build. Public ChatGPT
distribution requires a separately registered production MCP endpoint and its
submission/authentication process; a local stdio ZIP is not that registration.

Format references: [OpenAI plugin packaging](https://developers.openai.com/plugins/build/plugins),
[Claude manifest reference](https://code.claude.com/docs/en/plugins-reference).
