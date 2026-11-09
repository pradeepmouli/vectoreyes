# Local MCP supervision on macOS

The local server and OpenAI tunnel run as two user LaunchAgents, independent of
the agent terminal session. Both start at login and restart after exit, with a
10-second restart throttle. They run while the user is logged in; this is not a
system service and cannot keep a sleeping Mac reachable.

New-install configuration:

- `~/Library/LaunchAgents/com.vectoreyes.mcp.plist`
- `~/Library/LaunchAgents/com.vectoreyes.tunnel.plist`
- Installed wheel runtime: `~/.local/share/vectoreyes/runtime`
- Logs: `~/Library/Logs/vectorEyes/`

The server listens on `127.0.0.1:8000/mcp`. Configure a `vectoreyes` tunnel profile
in `~/.config/tunnel-client` before using the new service labels. Its interactive zsh launch
loads the existing `CONTROL_PLANE_API_KEY` setting; no credential is copied
into the LaunchAgent or repository. Changing shell startup configuration may
affect the tunnel. Avoid printing its environment or credential-bearing logs.
The tunnel waits for the local MCP HTTP endpoint before starting, so a cold
server import at login does not race tunnel initialization.

The source rename does not migrate an installed service. Existing installations
retain their former runtime directory, LaunchAgent labels, and tunnel profile
until an explicit cutover. Export important drawings before switching servers:
the old process owns their live IDs.

For the rename cutover, run the new `com.vectoreyes.mcp` service on loopback
port `8001` alongside the old service on `8000`. Keep the existing tunnel
profile, tunnel ID, and LaunchAgent; change only its upstream URL to
`http://127.0.0.1:8001/mcp`. Restart the tunnel job, not the old drawing server.
Keep a private backup of the previous profile so the upstream change can be
rolled back. Clients must refresh the connector's tool list after the switch.

## Status and restart

```sh
launchctl print gui/$(id -u)/com.vectoreyes.mcp
launchctl print gui/$(id -u)/com.vectoreyes.tunnel
curl --fail http://127.0.0.1:8080/readyz

launchctl kickstart -k gui/$(id -u)/com.vectoreyes.mcp
launchctl kickstart -k gui/$(id -u)/com.vectoreyes.tunnel
```

Readiness checks the tunnel; also verify the server through an MCP client
listing `vectoreyes_trace_drawing`, `vectoreyes_retrace_drawing`, `vectoreyes_refine_drawing`, `vectoreyes_name_regions`,
`vectoreyes_get_drawing_artifact`, and `vectoreyes_render_drawing`.

Tool POST responses use JSON rather than SSE so large review panels and command
maps are not constrained by the SDK client's 1 MiB SSE event limit. The complete
tool result is preserved. This mode does not support mid-call client sampling or
elicitation; vectorEyes does not currently use those callbacks.

To stop supervision, unload each job with `launchctl bootout
gui/$(id -u)/com.vectoreyes.mcp` (or `.tunnel`). Reload using `launchctl
bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.vectoreyes.mcp.plist`.

## Updating the installed build

```sh
npm --prefix integrations/mcp-app run build
uv build --wheel --python "$HOME/.local/share/vectoreyes/runtime/bin/python" \
  --out-dir "$HOME/.local/share/vectoreyes/builds"
wheel=$(ls -t "$HOME"/.local/share/vectoreyes/builds/vectoreyes-*.whl | head -n 1)
uv pip install --reinstall-package vectoreyes \
  --python "$HOME/.local/share/vectoreyes/runtime/bin/python" \
  "${wheel}[server,scoring]"
launchctl kickstart -k gui/$(id -u)/com.vectoreyes.mcp
launchctl kickstart -k gui/$(id -u)/com.vectoreyes.tunnel
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
