# Simple badge: one circle and one curved ribbon

Suppose the source is a flat-color badge: a blue circle above an orange curved
ribbon. The circle should be round, the ribbon has intentional corner points,
and one smooth side should be one quadratic. These are observations to verify
against the actual source, not defaults to apply to every logo.

1. Submit `vectoreyes_trace_drawing` with the attached ImageRef and
   `{"refine":"auto","max_sampling_colors":4,"fit_strategy":"progressive_allow_lines"}`.
   Inspect v0 and the returned pass revisions; use whichever preserves the intended silhouette better.
2. Read the selected version's region map. Assume the circle is `r1` and ribbon is `r2` there.
   If the circle is already a recognized circle, do not fit it again. Otherwise
   use `primitive-plan.json`, substituting the real drawing ID and region ID.
3. Inspect the child. If a ribbon side remains unnecessarily fragmented, fetch
   its command map. Assume `r2.p0.c2` and `r2.p0.c6` bound that side. Use
   `curve-plan.json`, substituting the real child version and command IDs.
   The fit runs from the end of c2 to the end of c6: it replaces c3 through
   c6 inclusive, leaving c2 unchanged and preserving c6's endpoint. For a
   corner-to-corner side, name its entry and destination corner endpoints
   directly; do not select the command after the destination. Rounded corner
   spans must be excluded from the side.
4. Inspect the new review panel and difference. Reject a single-Q proposal if
   its shape mismatches the source or exceeds the action's error tolerance.
   A more complex side may require `strategy="progressive"` instead of a forced
   single command; do not simply loosen tolerance until a bad shape passes.
5. Retrieve the accepted SVG with `vectoreyes_get_drawing_artifact` and return it with the
   retained version. Describe any unresolved limitation.

The plans are schema-valid examples, not a recorded execution: live IDs,
addresses, and the visual evidence determine whether these operations apply.
They never supply invented SVG control points.
