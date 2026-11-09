import type { App, McpUiHostContext } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { StrictMode, useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { ArtworkHandoff, GeneratingIndicator } from "./ArtworkHandoff";
import type { Revision } from "./ArtworkHandoff";
import { readArtifact } from './artifactDelivery';
import vectorEyesIcon from "../../../src/vectoreyes/assets/vectoreyes-icon.svg?inline";
import vectorEyesLockup from "../../../src/vectoreyes/assets/vectoreyes-lockup.svg?inline";

// ─── Data shapes ──────────────────────────────────────────────────────────────

interface DiagnosticsInput {
  source_kind?: string;
  mime_type?: string;
  bytes?: number;
  sha256?: string;
  original_width?: number;
  original_height?: number;
}

interface DiagnosticsProcessed {
  width?: number;
  height?: number;
  cropped?: boolean;
  resized?: boolean;
  transparent?: boolean;
  quantized?: boolean;
}

interface DiagnosticsVectorEyes {
  colors?: number;
  flatten?: boolean;
  no_symmetry?: boolean;
  epsilon?: number;
  max_error?: number;
}

interface DiagnosticsOutput {
  svg_bytes?: number;
  element_count?: number;
  has_defs?: boolean;
  has_paths?: boolean;
  has_primitives?: boolean;
  has_symmetry?: boolean;
}

interface Diagnostics {
  input?: DiagnosticsInput;
  processed?: DiagnosticsProcessed;
  vectoreyes?: DiagnosticsVectorEyes;
  output?: DiagnosticsOutput;
  warnings?: string[];
}

/** Richer idealize_logo shape; also accepts the older flat render_idealized_logo shape. */
interface IdealizeLogoResult {
  svg: string;
  width: number;
  height: number;
  svg_bytes: number;
  preview_available?: boolean;
  /** Legacy field — accepted for backward compat but never displayed. */
  image_path?: string;
  diagnostics?: Diagnostics;
}

interface ImageRef {
  download_url?: string;
  file_id?: string;
  mime_type?: string;
  file_name?: string;
  url?: string;
  data_uri?: string;
  base64?: string;
}

interface TraceOptionsInput {
  preset?: string | null;
  refinement_passes: Record<string, boolean>;
  initial_stitch: boolean;
  background_color: string | null;
  background_tolerance: number;
  refine: "auto" | "none";
  max_sampling_colors: number | "auto";
  max_output_colors: number | null;
  allow_gradients: boolean;
  background_cutouts: "subtract" | "mask" | "off";
  min_region_size: number;
  max_hole_area: number;
  min_region_fraction: number;
  trace_level: "pixel" | "subpixel";
  simplify_tolerance: number;
  curve_tolerance: number;
  fit_strategy: "quadratic" | "progressive" | "progressive_allow_lines";
  corner_normalize: boolean;
  corner_rounding: boolean;
  support_weighted_corners: boolean;
  gradient_pretrace: boolean;
  remove_background: "auto" | "off" | "on";
  preprocess: {
    max_size_px: number;
    crop_to_content: boolean;
    preserve_transparency: boolean;
    quantize: boolean;
  };
}

interface TraceToolInput {
  /**
   * ChatGPT currently sends either a transferable ImageRef or a renderer-only
   * `/mnt/data/...` display path.  Keep this unknown at the boundary and
   * normalize it before ever calling the MCP server.
   */
  image?: unknown;
  image_data?: unknown;
  options?: Partial<TraceOptionsInput>;
}

interface TraceOptionSchema {
  "x-vectoreyes-presets"?: Record<string, { label: string; options: Partial<TraceOptionsInput> }>;
  enum?: string[];
  default?: unknown;
  description?: string;
  properties?: Record<string, TraceOptionSchema>;
}

interface DrawingArtifacts {
  report: string;
  svg: string;
  preview: string;
  review_panel: string;
  labeled_svg: string;
  path_map: string;
  command_map: string;
  raw_trace: string;
  plan: string;
  versions: string;
}

interface DrawingVersionResult {
  version: string;
  parent_version: string | null;
}

interface DrawingResult {
  drawing_id: string;
  version: string;
  parent_version: string | null;
  requested_base_version: string | null;
  baseline_version: string;
  outputs: DrawingVersionResult[];
  artifacts: DrawingArtifacts;
  report: { targets: unknown[]; groups: unknown[] };
  trace?: { width?: number; height?: number; options?: Record<string, unknown> };
  trace_options_schema?: Record<string, TraceOptionSchema>;
  trace_options?: TraceOptionsInput;
}

type WidgetResult = IdealizeLogoResult | DrawingResult;

type Theme = "default" | "daikonic";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function firstText(result: CallToolResult): string | undefined {
  const block = result.content?.find((item) => item.type === "text");
  return block?.type === "text" ? block.text : undefined;
}

function firstImageDataUri(result: CallToolResult | null): string | null {
  if (!result) return null;
  const block = result.content?.find((item) => item.type === "image");
  if (block?.type === "image" && block.data && block.mimeType) {
    return `data:${block.mimeType};base64,${block.data}`;
  }
  return null;
}

/**
 * Accepts both the richer idealize_logo shape (with optional diagnostics) and
 * the older flat render_idealized_logo shape. Defensive: missing fields are
 * tolerated — only `svg` is required.
 */
function parseStructuredResult(result: CallToolResult | null): WidgetResult | null {
  if (!result) return null;

  type DecodedResult = Partial<IdealizeLogoResult> & Partial<DrawingResult>;
  const structured = result.structuredContent as DecodedResult | undefined;
  if (structured?.drawing_id && structured.version && structured.artifacts) {
    return structured as DrawingResult;
  }
  if (structured?.svg) {
    return {
      svg: structured.svg,
      width: Number(structured.width ?? 0),
      height: Number(structured.height ?? 0),
      svg_bytes: Number(structured.svg_bytes ?? structured.svg.length),
      preview_available: structured.preview_available ?? false,
      image_path: structured.image_path,
      diagnostics: structured.diagnostics,
    };
  }

  const text = firstText(result);
  if (!text) return null;
  try {
    const parsed = JSON.parse(text) as DecodedResult;
    if (parsed.drawing_id && parsed.version && parsed.artifacts) {
      return parsed as DrawingResult;
    }
    if (parsed.svg) {
      return {
        svg: parsed.svg,
        width: Number(parsed.width ?? 0),
        height: Number(parsed.height ?? 0),
        svg_bytes: Number(parsed.svg_bytes ?? parsed.svg.length),
        preview_available: parsed.preview_available ?? false,
        image_path: parsed.image_path,
        diagnostics: parsed.diagnostics,
      };
    }
  } catch {
    return null;
  }
  return null;
}

function isDrawingResult(result: WidgetResult): result is DrawingResult {
  return "drawing_id" in result && "artifacts" in result;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

const SOURCE_KIND_LABELS: Record<string, string> = {
  platform_file: "Platform file",
  local_path: "Local file",
  url: "URL",
  data_uri: "Data URI",
  base64: "Base64",
};


// ─── Trace form and drawing artifact viewer ───────────────────────────────────

const DEFAULT_TRACE_OPTIONS: TraceOptionsInput = {
  refinement_passes: {}, // Defaults and pass names come from the engine's returned schema.
  initial_stitch: true,
  background_color: null,
  background_tolerance: 32,
  refine: "none",
  max_sampling_colors: 16,
  max_output_colors: null,
  allow_gradients: true,
  background_cutouts: "subtract",
  min_region_size: 16,
  max_hole_area: 128,
  min_region_fraction: 0.02,
  trace_level: "pixel",
  simplify_tolerance: 1.5,
  curve_tolerance: 1,
  fit_strategy: "quadratic",
  corner_normalize: false,
  corner_rounding: true,
  support_weighted_corners: false,
  gradient_pretrace: false,
  remove_background: "auto",
  preprocess: { max_size_px: 2048, crop_to_content: true, preserve_transparency: true, quantize: false },
};

const FALLBACK_ENUMS: Record<string, string[]> = {
  refine: ["auto", "none"],
  trace_level: ["pixel", "subpixel"],
  fit_strategy: ["quadratic", "progressive", "progressive_allow_lines"],
  remove_background: ["auto", "off", "on"],
};

type ArtifactKind = "svg" | "preview" | "review_panel" | "labeled_svg" | "path_map" | "command_map" | "raw_trace" | "plan" | "versions" | "report";

const ARTIFACTS: Array<{ kind: ArtifactKind; label: string }> = [
  { kind: "svg", label: "SVG" },
  { kind: "preview", label: "Preview PNG" },
  { kind: "review_panel", label: "Review panel" },
  { kind: "labeled_svg", label: "Region map SVG" },
  { kind: "path_map", label: "Path segment map SVG" },
  { kind: "command_map", label: "Command map SVG" },
  { kind: "raw_trace", label: "Raw trace JSON" },
  { kind: "report", label: "Geometry report JSON" },
  { kind: "plan", label: "Plan JSON" },
  { kind: "versions", label: "Version manifest" },
];

function enumChoices(schema: Record<string, TraceOptionSchema> | undefined, name: string): string[] {
  return schema?.[name]?.enum ?? FALLBACK_ENUMS[name] ?? [];
}

function dataUriForSvg(svg: string): string {
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

function imageRefFromToolInput(value: unknown): ImageRef | undefined {
  if (typeof value === "string") {
    return value.startsWith("data:image/png;") ? { data_uri: value } : undefined;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const candidate = value as Partial<ImageRef>;
  if (typeof candidate.download_url === "string") return typeof candidate.file_id === "string" && candidate.file_id ? { ...candidate, download_url: candidate.download_url } : undefined;
  if (typeof candidate.url === "string") return { ...candidate, url: candidate.url };
  if (typeof candidate.data_uri === "string" && candidate.data_uri.startsWith("data:image/png;")) return { ...candidate, data_uri: candidate.data_uri };
  if (typeof candidate.base64 === "string") return { ...candidate, base64: candidate.base64 };
  return undefined;
}

function sourceImageMessage(options: TraceOptionsInput): string {
  return [
    "Please rerun `vectoreyes_trace_drawing` using the original attached image and these settings from the vectorEyes widget.",
    "The widget received only a host-local display path, so it cannot safely resubmit the image itself.",
    "```json",
    JSON.stringify({ options }, null, 2),
    "```",
  ].join("\n");
}

function artifactSvg(result: CallToolResult | null): string | null {
  const structured = result?.structuredContent as { svg?: string } | undefined;
  if (structured?.svg) return structured.svg;
  const text = result ? firstText(result) : undefined;
  return text?.startsWith("<svg") ? text : null;
}

function TraceControls({
  app,
  initialInput,
  result,
  onResult,
  onError,
  recordedOptions,
  contextVersion,
  onRunning,
}: {
  app: App;
  initialInput: TraceToolInput;
  result: WidgetResult | null;
  onResult: (result: CallToolResult) => void;
  onError: (message: string | null) => void;
  recordedOptions?: Partial<TraceOptionsInput>;
  contextVersion: string;
  onRunning: (running: boolean) => void;
}) {
  const [form, setForm] = useState<TraceOptionsInput>(DEFAULT_TRACE_OPTIONS);
  const [sourceImage, setSourceImage] = useState<ImageRef | undefined>();
  const [sourceUnavailable, setSourceUnavailable] = useState(false);
  const [handoffSent, setHandoffSent] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const schema = result && isDrawingResult(result) ? result.trace_options_schema : undefined;

  useEffect(() => {
    const input = initialInput.image ?? initialInput.image_data;
    const image = imageRefFromToolInput(input);
    setSourceImage(image);
    setSourceUnavailable(Boolean(input) && !image);
    setHandoffSent(false);
  }, [initialInput]);

  const lastTraceOptions = recordedOptions;
  useEffect(() => {
    setForm({
      ...DEFAULT_TRACE_OPTIONS,
      ...initialInput.options,
      ...lastTraceOptions,
      preprocess: { ...DEFAULT_TRACE_OPTIONS.preprocess, ...initialInput.options?.preprocess, ...lastTraceOptions?.preprocess },
    });
  }, [initialInput, lastTraceOptions]);

  const setOption = <K extends keyof TraceOptionsInput>(key: K, value: TraceOptionsInput[K]) => {
    setForm((current) => ({ ...current, [key]: value }));
  };

  const runTrace = useCallback(async () => {
    if (!sourceImage) {
      onError("Trace a ChatGPT-attached image first; the widget reruns that same image reference.");
      return;
    }
    setIsRunning(true);
    onRunning(true);
    onError(null);
    try {
      onResult(await app.callServerTool({
        name: "vectoreyes_trace_drawing",
        arguments: { ...(sourceImage.download_url ? { image: sourceImage } : { image_data: sourceImage }), options: form },
      }));
    } catch (runError) {
      onError(runError instanceof Error ? runError.message : String(runError));
    } finally {
      setIsRunning(false);
      onRunning(false);
    }
  }, [app, form, onError, onResult, sourceImage, onRunning]);

  const requestHostRerun = useCallback(async () => {
    setIsRunning(true);
    onError(null);
    try {
      const response = await app.sendMessage({
        role: "user",
        content: [{ type: "text", text: sourceImageMessage(form) }],
      });
      if (response.isError) throw new Error("ChatGPT could not queue the trace request.");
      setHandoffSent(true);
    } catch (handoffError) {
      onError(handoffError instanceof Error ? handoffError.message : String(handoffError));
    } finally {
      setIsRunning(false);
    }
  }, [app, form, onError]);

  const enumSelect = (name: keyof Pick<TraceOptionsInput, "refine" | "trace_level" | "fit_strategy" | "remove_background">, label: string) => (
    <label className="field" key={name}>
      {label}
      <select value={form[name]} onChange={(event) => setOption(name, event.target.value as TraceOptionsInput[typeof name])}>
        {enumChoices(schema, name).map((choice) => <option key={choice} value={choice}>{choice}</option>)}
      </select>
    </label>
  );

  return (
    <section className="control-panel" aria-label="trace settings">
        <div className="settings-heading"><h3>Rerun from the source</h3><span role="status">{isRunning ? "Tracing…" : `Draft for ${contextVersion}`}</span></div>
      <p className="control-note">These are proposed settings for a new conversion, not edits to the selected revision. Recorded values seed the form; settings not recorded use defaults. Exact retained evidence is listed separately above.</p>

      <p className="control-note">
        {sourceImage
          ? "Reruns use the original attached image."
          : sourceUnavailable
            ? "ChatGPT supplied a display-only attachment path. Ask the agent to rerun it with these settings."
              : "Your agent passes the generated or attached image into vectorEyes. These controls are for optional reruns."}
      </p>

        <label className="field">Starting preset
          <select value={form.preset ?? ""} onChange={(event) => {
            const preset = event.target.value;
            const selected = schema?.preset?.["x-vectoreyes-presets"]?.[preset];
            if (!preset) { setOption("preset", null); return; }
            if (selected) setForm((current) => ({ ...DEFAULT_TRACE_OPTIONS, ...selected.options,
              preset, preprocess: current.preprocess }));
          }}>
            <option value="">Custom settings</option>
            {Object.entries(schema?.preset?.["x-vectoreyes-presets"] ?? {}).map(([name, preset]) =>
              <option key={name} value={name}>{preset.label}</option>)}
          </select>
        </label>
        <p className="control-note">Selecting a preset resets trace controls to that starting point. You can override every setting below.</p>
        <div className="select-grid">
        {enumSelect("trace_level", "Boundary")}
        {enumSelect("fit_strategy", "Path fitting")}
        {enumSelect("remove_background", "Background")}
      </div>

        <fieldset className="pass-checklist">
          <legend>Trace stages</legend>
          <div className="toggle-row">
            <label className="toggle"><input checked={form.preprocess.crop_to_content} type="checkbox" onChange={(event) => setForm((current) => ({ ...current, preprocess: { ...current.preprocess, crop_to_content: event.target.checked } }))} /> Crop to content</label>
            <label className="toggle"><input checked={form.gradient_pretrace} type="checkbox" onChange={(event) => setOption("gradient_pretrace", event.target.checked)} /> Gradient preprocessing</label>
            <label className="toggle"><input checked={form.preprocess.quantize} type="checkbox" onChange={(event) => setForm((current) => ({ ...current, preprocess: { ...current.preprocess, quantize: event.target.checked } }))} /> Pre-quantize colors</label>
            <label className="toggle"><input checked={form.initial_stitch} type="checkbox" onChange={(event) => setOption("initial_stitch", event.target.checked)} /> Initial stitch</label>
          </div>
          <p className="control-note">Selected stages run in the engine’s fixed order before the v0 trace is published.</p>
        </fieldset>

      <div className="number-grid">
        <label className="field">Max sampling colors
          <input disabled={form.max_sampling_colors === "auto"} min="2" max="256" type="number" value={form.max_sampling_colors === "auto" ? "" : form.max_sampling_colors}
            onChange={(event) => setOption("max_sampling_colors", Number(event.target.value) || 2)} />
        </label>
        <label className="toggle"> <input checked={form.max_sampling_colors === "auto"} type="checkbox"
          onChange={(event) => setOption("max_sampling_colors", event.target.checked ? "auto" : 16)} /> Auto palette </label>
        <label className="field">Min region px
          <input min="1" type="number" value={form.min_region_size} onChange={(event) => setOption("min_region_size", Number(event.target.value))} />
        </label>
      </div>

      <div className="number-grid">
        <label className="field">Max output colors (blank = unlimited)
          <input min="1" max="256" type="number" value={form.max_output_colors ?? ""}
            onChange={(event) => setOption("max_output_colors", event.target.value === "" ? null : Number(event.target.value))} />
        </label>
        <label className="toggle"><input checked={form.allow_gradients} type="checkbox"
          onChange={(event) => setOption("allow_gradients", event.target.checked)} /> Allow gradients</label>
        <label className="field">Background cutouts
          <select value={form.background_cutouts} onChange={(event) => setOption("background_cutouts", event.target.value as TraceOptionsInput["background_cutouts"])}>
            {["subtract", "mask", "off"].map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
      </div>

      <details className="trace-advanced">
        <summary>Advanced trace settings</summary>
          <div className="number-grid">
            <label className="field">Background color (blank = infer)
              <input type="text" placeholder="#ffffff" value={form.background_color ?? ""}
                pattern="#[0-9a-fA-F]{6}" onChange={(event) => setOption("background_color", event.target.value || null)} />
            </label>
            <label className="field">Background RGB tolerance
              <input type="number" min="0" max="442" value={form.background_tolerance}
                onChange={(event) => setOption("background_tolerance", Number(event.target.value))} />
            </label>
            <label className="field">Hole area
            <input min="0" type="number" value={form.max_hole_area} onChange={(event) => setOption("max_hole_area", Number(event.target.value))} />
          </label>
          <label className="field">Min fraction
            <input min="0" max="0.99" step="0.01" type="number" value={form.min_region_fraction} onChange={(event) => setOption("min_region_fraction", Number(event.target.value))} />
          </label>
          <label className="field">Simplify px
            <input min="0" step="0.1" type="number" value={form.simplify_tolerance} onChange={(event) => setOption("simplify_tolerance", Number(event.target.value))} />
          </label>
          <label className="field">Curve error
            <input min="0" step="0.1" type="number" value={form.curve_tolerance} onChange={(event) => setOption("curve_tolerance", Number(event.target.value))} />
          </label>
          <label className="field">Max image px
            <input min="1" type="number" value={form.preprocess.max_size_px} onChange={(event) => setForm((current) => ({ ...current, preprocess: { ...current.preprocess, max_size_px: Number(event.target.value) } }))} />
          </label>
        </div>
          <div className="toggle-row">
            <label className="toggle"><input checked={form.corner_rounding} type="checkbox" onChange={(event) => setOption("corner_rounding", event.target.checked)} /> Protect rounded corners in v0</label>
            <label className="toggle"><input checked={form.support_weighted_corners} type="checkbox" onChange={(event) => setOption("support_weighted_corners", event.target.checked)} /> Support-weighted corners</label>
          <label className="toggle"><input checked={form.preprocess.preserve_transparency} type="checkbox" onChange={(event) => setForm((current) => ({ ...current, preprocess: { ...current.preprocess, preserve_transparency: event.target.checked } }))} /> Preserve alpha</label>
        </div>
          </details>

            <fieldset className="pass-checklist">
              <legend>Refinement stages</legend>
              <label className="toggle"><input type="checkbox" checked={form.refine === "auto"} onChange={(event) => setOption("refine", event.target.checked ? "auto" : "none")} /> Run selected refinements</label>
              <p className="control-note">Each executed pass produces a revision, even if unchanged. Order is fixed; source fills are resolved last.</p>
            <div className="toggle-row">
              {Object.entries(schema?.refinement_passes?.properties ?? {}).map(([name, definition]) => (
                <label className="toggle" key={name} title={definition.description}>
                  <input type="checkbox" disabled={form.refine !== "auto"}
                    checked={form.refinement_passes?.[name] ?? Boolean(definition.default)}
                    onChange={(event) => setOption("refinement_passes", { ...form.refinement_passes, [name]: event.target.checked })} />
                  {name.replaceAll("_", " ")}
                </label>
              ))}
                <label className="toggle"><input checked={form.corner_normalize} disabled={form.refine !== "auto"} type="checkbox" onChange={(event) => setOption("corner_normalize", event.target.checked)} /> Normalize corners</label>
            </div>
            </fieldset>

        {lastTraceOptions && <details className="trace-advanced">
          <summary>Settings used to seed this draft</summary>
          <pre className="artifact-json">{JSON.stringify(lastTraceOptions, null, 2)}</pre>
        </details>}

      {sourceImage ? (
        <button className="run-button" disabled={isRunning} onClick={runTrace}>
          {isRunning ? "Tracing…" : "Rerun trace"}
        </button>
      ) : sourceUnavailable ? (
        <button className="run-button" disabled={isRunning || handoffSent} onClick={requestHostRerun}>
          {isRunning ? "Sending…" : handoffSent ? "Trace request sent" : "Ask agent to rerun"}
        </button>
      ) : (
        <button className="run-button" disabled>Rerun trace</button>
      )}
    </section>
  );
}

interface RevisionRecipeArtifact {
  parent_version?: string | null;
  effective_recipe?: { fields?: { settings?: { max_error?: number; epsilon?: number; fit_strategy?: TraceOptionsInput["fit_strategy"] } } };
}

function RevisionReview({ app, drawing, version, initialInput, read, onResult, onError, onRunning }: {
  app: App; drawing: DrawingResult; version: string; initialInput: TraceToolInput;
  read: (kind: ArtifactKind, version: string) => Promise<CallToolResult>;
  onResult: (result: CallToolResult) => void; onError: (message: string | null) => void; onRunning: (running: boolean) => void;
}) {
  const [evidence, setEvidence] = useState<{ report: unknown; trace: { options?: Partial<TraceOptionsInput> }; plan: RevisionRecipeArtifact } | null>(null);
  const [failed, setFailed] = useState(false);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let current = true;
    setFailed(false);
    Promise.all([read("report", version), read("raw_trace", version), read("plan", version)]).then(([report, trace, plan]) => {
      if (current) setEvidence({ report: report.structuredContent?.report, trace: trace.structuredContent?.trace as { options?: Partial<TraceOptionsInput> }, plan: plan.structuredContent?.plan as RevisionRecipeArtifact });
    }).catch(() => { if (current) setFailed(true); });
    return () => { current = false; };
  }, [read, version, retry]);
  const options = useMemo(() => {
    if (!evidence) return undefined;
    const settings = evidence.plan?.effective_recipe?.fields?.settings;
    return { ...(evidence.trace?.options as Partial<TraceOptionsInput>),
      ...(version === drawing.version ? drawing.trace_options : {}),
      ...(settings?.max_error !== undefined ? { curve_tolerance: settings.max_error } : {}),
      ...(settings?.epsilon !== undefined ? { simplify_tolerance: settings.epsilon } : {}),
      ...(settings?.fit_strategy ? { fit_strategy: settings.fit_strategy } : {}) };
  }, [evidence, drawing.trace_options, drawing.version, version]);
  return <>
    <details className="artifact-details"><summary>Diagnostics · {version}</summary>
      <code className="artifact-uri">{drawing.drawing_id} · {version} · Parent {evidence ? evidence.plan?.parent_version ?? "none" : "loading…"}</code>
      <p className="control-note">Geometry evidence for the selected revision. Shape and fill should be reviewed independently.</p>
      {evidence ? <pre className="artifact-json">{JSON.stringify(evidence.report, null, 2) ?? "No geometry report was returned."}</pre> : <p role="status">{failed ? "Review evidence could not be loaded." : `Loading evidence for ${version}…`}</p>}
    </details>
    <details className="artifact-details"><summary>Settings used · {version}</summary>
      <p className="control-note">Read-only evidence, not a proposed rerun. Source trace settings and this revision’s recipe settings are kept separate.</p>
      {evidence && <><h3>Source trace settings</h3><pre className="artifact-json">{JSON.stringify(evidence.trace?.options, null, 2) ?? "Not recorded."}</pre><h3>Revision recipe settings</h3><pre className="artifact-json">{JSON.stringify(evidence.plan?.effective_recipe?.fields?.settings, null, 2) ?? "Not recorded."}</pre></>}
    </details>
    {failed && <button className="secondary-button" onClick={() => setRetry((value) => value + 1)}>Retry review evidence</button>}
    {evidence && <details className="artifact-details"><summary>Rerun settings &amp; passes</summary><TraceControls app={app} initialInput={initialInput} result={drawing} recordedOptions={options} contextVersion={version} onResult={onResult} onError={onError} onRunning={onRunning} /></details>}
  </>;
}

function ArtifactViewer({ app, drawing, initialInput, generating, onResult, onError, onRunning }: { app: App; drawing: DrawingResult; initialInput: TraceToolInput; generating: boolean; onResult: (result: CallToolResult) => void; onError: (message: string | null) => void; onRunning: (running: boolean) => void }) {
  const [selected, setSelected] = useState<ArtifactKind>("svg");
  const [viewVersion, setViewVersion] = useState(drawing.version);
  const [loaded, setLoaded] = useState<{ key: string; result: CallToolResult } | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadFailed, setLoadFailed] = useState(false);
  const [retry, setRetry] = useState(0);
  const [exported, setExported] = useState<{ url: string; filename: string; version: string } | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const editRequest = drafts[viewVersion] ?? "";
  const [sending, setSending] = useState(false);
  const [sentVersions, setSentVersions] = useState<Record<string, boolean>>({});
  const sent = sentVersions[viewVersion] ?? false;
  const [exporting, setExporting] = useState(false);
  const [revisions, setRevisions] = useState<Revision[]>([]);
  const [manifestFailed, setManifestFailed] = useState(false);
  const cache = useMemo(() => new Map<string, Promise<CallToolResult>>(), [app, drawing.drawing_id]);
  const read = useCallback((kind: ArtifactKind, version: string) => {
    const key = `${version}:${kind}`;
    let pending = cache.get(key);
    if (!pending) {
      pending = readArtifact(args => app.callServerTool({ name: "vectoreyes_get_drawing_artifact", arguments: args }),
        { drawing_id: drawing.drawing_id, version, artifact: kind }).then((result) => {
          if (result.isError) throw new Error(firstText(result) ?? "The artifact could not be loaded.");
          return result;
        }).catch((error) => { cache.delete(key); throw error; });
      cache.set(key, pending);
    }
    return pending;
  }, [app, drawing.drawing_id, cache]);
  const loadPreview = useCallback(async (version: string) => {
    const result = await read(selected, version);
    const svg = artifactSvg(result);
    const src = svg ? dataUriForSvg(svg) : firstImageDataUri(result);
    if (!src) { cache.delete(`${version}:${selected}`); throw new Error("Revision preview unavailable."); }
    return src;
  }, [read, cache, selected]);
  useEffect(() => { setViewVersion(drawing.version); }, [drawing.version]);
  useEffect(() => {
    let current = true;
    setManifestFailed(false);
    // The manifest can grow, unlike immutable revision artifacts.
    readArtifact(args => app.callServerTool({ name: "vectoreyes_get_drawing_artifact", arguments: args }),
      { drawing_id: drawing.drawing_id, version: drawing.version, artifact: "versions" }).then((result) => {
        const items = result.structuredContent?.versions;
        if (result.isError || !Array.isArray(items) || !items.length || !items.every((item) => typeof item.version === "string" && (item.parent_version === null || typeof item.parent_version === "string"))) throw new Error("Invalid revision manifest.");
        if (current) setRevisions(items as Revision[]);
      }).catch(() => { if (current) setManifestFailed(true); });
    return () => { current = false; };
  }, [app, drawing.drawing_id, drawing.version, retry]);

  const fetchArtifact = useCallback((kind: ArtifactKind) => read(kind, viewVersion), [read, viewVersion]);
  const artifactKey = `${viewVersion}:${selected}`;

  useEffect(() => {
    let current = true;
    setExported(null);
    setLoaded(null);
    setLoading(true);
    setLoadFailed(false);
    onError(null);
    fetchArtifact(selected).then((result) => {
      if (!result.isError) { if (current) setLoaded({ key: artifactKey, result }); }
      else throw new Error(firstText(result) ?? "The artifact could not be loaded.");
    }).catch((loadError: unknown) => {
      if (current) { setLoadFailed(true); onError(loadError instanceof Error ? loadError.message : String(loadError)); }
    }).finally(() => { if (current) setLoading(false); });
    return () => { current = false; };
  }, [fetchArtifact, onError, selected, retry, artifactKey]);

  const currentArtifact = loaded?.key === artifactKey ? loaded.result : null;
  const svg = artifactSvg(currentArtifact);
  // SVG wins over incidental preview PNGs returned alongside a command map.
  const image = svg ? null : firstImageDataUri(currentArtifact);
  const versions = revisions.length ? revisions : drawing.outputs?.length ? drawing.outputs : [{ version: drawing.version, parent_version: drawing.parent_version }];

  const exportFile = async () => {
    const kind = "svg";
    setExporting(true);
    setExported(null);
    onError(null);
    try {
      const result = await fetchArtifact(kind);
      if (result.isError) throw new Error(firstText(result) ?? "The artifact could not be exported.");
      const svgContent = artifactSvg(result);
      if (!svgContent) throw new Error("This response did not contain a downloadable SVG.");
      const href = dataUriForSvg(svgContent);
      const filename = `${drawing.drawing_id}-${viewVersion}-svg.svg`;
      const host = (window as Window & { openai?: {
        uploadFile?: (file: File) => Promise<{ fileId: string }>;
        getFileDownloadUrl?: (input: { fileId: string }) => Promise<{ downloadUrl: string }>;
      } }).openai;
      if (host) {
        if (!host.uploadFile || !host.getFileDownloadUrl) {
          throw new Error("This ChatGPT host does not expose file export APIs. SVG text is still available through vectoreyes_get_drawing_artifact; no downloadable file was created.");
        }
        const blob = new Blob([svgContent], { type: "image/svg+xml" });
        const { fileId } = await host.uploadFile(new File([blob], filename, { type: blob.type }));
        if (!fileId) throw new Error("ChatGPT did not return an exported file ID.");
        const { downloadUrl } = await host.getFileDownloadUrl({ fileId });
        if (!downloadUrl || new URL(downloadUrl).protocol !== "https:") {
          throw new Error("ChatGPT did not return a valid file download URL.");
        }
        setExported({ url: downloadUrl, filename, version: viewVersion });
        await app.updateModelContext({ content: [{ type: "text", text: `vectorEyes exported ${filename} as ChatGPT file ${fileId}. Download: ${downloadUrl}` }] });
        return;
      }
      const link = document.createElement("a");
      link.href = href;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (exportError) {
      onError(exportError instanceof Error ? exportError.message : String(exportError));
    } finally {
      setExporting(false);
    }
  };

  const requestEdit = async () => {
    if (!editRequest.trim() || sending || sent) return;
    setSending(true); onError(null);
    try {
      const response = await app.sendMessage({ role: "user", content: [{ type: "text", text:
        `Revise vectorEyes drawing ${drawing.drawing_id}, starting from ${viewVersion}. Inspect its existing source, region and segment/feature artifacts, apply a versioned plan, and evaluate the result. Requested change: ${editRequest.trim()}` }] });
      if (response.isError) throw new Error("The host could not send the edit request. Please try again.");
      setSentVersions((current) => ({ ...current, [viewVersion]: true }));
    } catch (error) { onError(error instanceof Error ? error.message : String(error)); }
    finally { setSending(false); }
  };

  return <>
    <ArtworkHandoff version={viewVersion} revisions={versions} loadPreview={loadPreview} selected={selected} loading={loading || (!currentArtifact && !loadFailed)} unavailable={loadFailed} exporting={exporting} generating={generating}
      reviewOpen={reviewOpen} editRequest={editRequest} sending={sending} sent={sent}
      views={[{ kind: "svg", label: "Artwork" }, { kind: "review_panel", label: "Source & difference" }]}
      maps={[{ kind: "labeled_svg", label: "Regions" }, { kind: "path_map", label: "Segments & features" }, { kind: "command_map", label: "Commands" }]}
      onSelect={(kind) => setSelected(kind as ArtifactKind)}
      onVersionChange={setViewVersion} onDownloadSvg={() => void exportFile()}
      onToggleReview={() => { setReviewOpen(!reviewOpen); if (reviewOpen) setSelected("svg"); }}
      onEditRequestChange={(value) => { setDrafts((current) => ({ ...current, [viewVersion]: value })); setSentVersions((current) => ({ ...current, [viewVersion]: false })); }} onRequestEdit={() => void requestEdit()}
      reviewContent={<RevisionReview key={viewVersion} app={app} drawing={drawing} version={viewVersion} initialInput={initialInput} read={read} onResult={onResult} onError={onError} onRunning={onRunning} />}>
      {loading || (!currentArtifact && !loadFailed) ? <GeneratingIndicator label={`Loading ${ARTIFACTS.find((item) => item.kind === selected)?.label ?? selected}…`} />
        : image ? <img src={image} alt={`${selected} for ${viewVersion}`} /> : svg ? <img src={dataUriForSvg(svg)} alt={selected === "svg" ? `Vector artwork for ${viewVersion}` : `${selected} for ${viewVersion}`} />
          : <div className="loading-artifact"><p>No artifact was returned. Choose another view or retry.</p><button className="secondary-button" onClick={() => { cache.delete(artifactKey); setRetry((value) => value + 1); }}>Retry artifact</button></div>}
    </ArtworkHandoff>
    {exported?.version === viewVersion && <p role="status"><a href={exported.url} target="_blank" rel="noopener noreferrer" download={exported.filename}>Download {exported.filename}</a> (temporary link)</p>}
    {manifestFailed && <p className="control-note">Full revision history is unavailable. <button className="secondary-button" onClick={() => setRetry((value) => value + 1)}>Retry history</button></p>}
  </>;
}

function LegacyPreview({ result }: { result: IdealizeLogoResult }) {
  return <div className="preview-stage"><img src={dataUriForSvg(result.svg)} alt="Rendered SVG" /></div>;
}

function AppRoot() {
  const [toolInput, setToolInput] = useState<TraceToolInput>({});
  const [toolResult, setToolResult] = useState<CallToolResult | null>(null);
  const [hostContext, setHostContext] = useState<McpUiHostContext | undefined>();
  const [errorText, setErrorText] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  const { app, isConnected, error } = useApp({
    appInfo: { name: "vectoreyes", version: "0.2.1" }, capabilities: {},
    onAppCreated: (createdApp) => {
      createdApp.ontoolinput = (params) => {
        const input = (params.arguments ?? {}) as TraceToolInput;
        if (input.image !== undefined || input.image_data !== undefined || input.options !== undefined) setToolInput(input);
        if (Object.keys(input).length && !("artifact" in input)) setGenerating(true);
      };
      createdApp.ontoolresult = (result) => { setGenerating(false); setErrorText(null); setToolResult(result); };
      createdApp.ontoolcancelled = (params) => { setGenerating(false); setErrorText(params.reason || "Tool call was cancelled."); };
      createdApp.onerror = (appError) => { setGenerating(false); setErrorText(appError instanceof Error ? appError.message : String(appError)); };
      createdApp.onhostcontextchanged = (params) => setHostContext((current) => ({ ...current, ...params }));
      createdApp.onteardown = async () => ({});
    },
  });
  useEffect(() => { if (app) setHostContext(app.getHostContext()); }, [app]);
  const result = useMemo(() => parseStructuredResult(toolResult), [toolResult]);

  if (error) return <StatusPanel title="Connection failed" detail={error.message} />;
  if (!isConnected || !app) return <StatusPanel title="Connecting" detail="Waiting for the MCP Apps host." />;

  return <main className="app-shell" style={{ paddingTop: hostContext?.safeAreaInsets?.top, paddingRight: hostContext?.safeAreaInsets?.right, paddingBottom: hostContext?.safeAreaInsets?.bottom, paddingLeft: hostContext?.safeAreaInsets?.left }}>
    <header className="app-topbar"><img className="brand-id" src={vectorEyesLockup} width={142} height={21} alt="vectorEyes" /><span className="app-purpose">From image to editable artwork</span></header>
    <section className="preview-panel" aria-label="drawing response">
      {errorText && <div className="error-banner" role="alert">{errorText}</div>}
      {result ? isDrawingResult(result) ? <ArtifactViewer key={result.drawing_id} app={app} drawing={result} initialInput={toolInput} generating={generating} onResult={setToolResult} onError={setErrorText} onRunning={setGenerating} /> : <LegacyPreview result={result} /> : generating ? <div className="initial-generation"><h1>Creating your vector artwork</h1><GeneratingIndicator /></div> : <EmptyPreview />}
    </section>
  </main>;
}

function EmptyPreview() { return <div className="empty-preview"><img className="brand-icon" src={vectorEyesIcon} width={56} height={56} alt="" /><h1>Your next image, in vector.</h1><p>Your agent can pass a generated or attached image directly into vectorEyes. The artwork and SVG download will appear here.</p></div>; }
function StatusPanel({ title, detail }: { title: string; detail: string }) { return <main className="status-panel"><h1>{title}</h1><p>{detail}</p></main>; }

createRoot(document.getElementById("root")!).render(<StrictMode><AppRoot /></StrictMode>);
