# Local MCP supervision on macOS

The local server and OpenAI tunnel run as two user LaunchAgents, independent of
the agent terminal session. Both start at login and restart after exit, with a
10-second restart throttle. They run while the user is logged in; this is not a
system service and cannot keep a sleeping Mac reachable.

Installed configuration:

- `~/Library/LaunchAgents/com.vectormark.mcp.plist`
- `~/Library/LaunchAgents/com.vectormark.tunnel.plist`
- Installed wheel runtime: `~/.local/share/vectormark/runtime`
- Logs: `~/Library/Logs/VectorMark/`

The server listens on `127.0.0.1:8000/mcp`. The tunnel uses the existing
`vectormark` profile in `~/.config/tunnel-client`. Its interactive zsh launch
loads the existing `CONTROL_PLANE_API_KEY` setting; no credential is copied
into the LaunchAgent or repository. Changing shell startup configuration may
affect the tunnel. Avoid printing its environment or credential-bearing logs.
The tunnel waits for the local MCP HTTP endpoint before starting, so a cold
server import at login does not race tunnel initialization.

## Status and restart

```sh
launchctl print gui/$(id -u)/com.vectormark.mcp
launchctl print gui/$(id -u)/com.vectormark.tunnel
curl --fail http://127.0.0.1:8080/readyz

launchctl kickstart -k gui/$(id -u)/com.vectormark.mcp
launchctl kickstart -k gui/$(id -u)/com.vectormark.tunnel
```

Readiness checks the tunnel; also verify the server through an MCP client
listing `trace_drawing`, `retrace_drawing`, `refine_drawing`,
`get_drawing_artifact`, and `render_drawing`.

Tool POST responses use JSON rather than SSE so large review panels and command
maps are not constrained by the SDK client's 1 MiB SSE event limit. The complete
tool result is preserved. This mode does not support mid-call client sampling or
elicitation; VectorMark does not currently use those callbacks.

To stop supervision, unload each job with `launchctl bootout
gui/$(id -u)/com.vectormark.mcp` (or `.tunnel`). Reload using `launchctl
bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.vectormark.mcp.plist`.

## Updating the installed build

```sh
npm --prefix integrations/mcp-app run build
uv build --wheel --python "$HOME/.local/share/vectormark/runtime/bin/python" \
  --out-dir "$HOME/.local/share/vectormark/builds"
wheel=$(ls -t "$HOME"/.local/share/vectormark/builds/vectormark-*.whl | head -n 1)
uv pip install --reinstall-package vectormark \
  --python "$HOME/.local/share/vectormark/runtime/bin/python" \
  "${wheel}[server,scoring]"
launchctl kickstart -k gui/$(id -u)/com.vectormark.mcp
launchctl kickstart -k gui/$(id -u)/com.vectormark.tunnel
```

Build with the same Python version and platform as the runtime: wheels now
contain a compiled drawing engine. This runtime is an installed wheel, not an
editable checkout.

## Exporting artifacts

In the drawing UI, select SVG, Plan JSON, Review panel, or another response file
and use **Export selected file**. The UI retrieves the selected version's
artifact contents and downloads them, rather than saving a `drawing://` URI.
Embedded-browser download behavior depends on the host permitting downloads.

Supervision is not drawing persistence: a server restart clears in-memory
drawing IDs and versions. Export important SVGs and plans before a planned
restart. Saved plans preserve declarations, but they are not a complete drawing
session backup and cannot be replayed against an expired drawing ID directly.
