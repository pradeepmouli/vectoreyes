# Local artwork UI preview

This host simulator renders the actual widget with saved evaluation artifacts.
It never submits conversions, plans, uploads, or messages to a live service.

From `integrations/mcp-app`, run these in separate terminals:

```sh
npm run dev
node dev/serve-fixtures.mjs /absolute/path/to/saved-radish-evaluation-bundle
```

The fixture server expects saved `versions/v0` and `versions/v1` artifact folders,
including `response.json`, SVG/map JSON and review PNGs. It only serves the
allowlisted artifacts from that directory over loopback. Open
`http://127.0.0.1:62747/dev/preview.html`.

`?scenario=error` exercises failed artifact retrieval; `?scenario=empty` exercises
the connected empty state; `?scenario=generating` shows the initial dot-grid
loading state. The preview labels its saved-artifact status clearly.
It does not simulate ChatGPT upload/download APIs or prove behavior in that host.

Run `npm test` for server-rendered component checks. `npm run test:browser` needs
Playwright available, or `PLAYWRIGHT_MODULE` pointing to its absolute `index.mjs`,
and a browser installation. `CHROME_PATH` optionally selects an existing Chrome
executable; `REVIEW_DIR` selects the screenshot directory. The browser test checks
real widget transitions, selected-version requests, edit messages, downloads,
the single-picture SVG carousel, retained per-revision drafts, mobile overflow, settings,
dark/reduced-motion, zoom, pending revisions, and error/empty states. Generation
uses real tool-input/result/cancel events; sending an edit request alone does not
claim the agent has started a conversion.

The full-size image carousel is the sole revision selection: one picture is
visible at a time, and horizontal swiping, arrows, or keyboard navigation move
between revisions. There is no thumbnail strip or second artwork preview.
Its parent-linked manifest drives
the branch path, and all review artifacts, settings evidence, edit targets, and
SVG exports use that selected revision. Review controls and optional rerun drafts
are absent from the handoff until Review & edit is opened. Rerun drafts are
explicitly separate from read-only source trace and revision recipe settings.

These dev files are not part of the single-file MCP widget build.
