import type { App, McpUiHostContext } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { StrictMode, useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

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

interface DiagnosticsVectormark {
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
  vectormark?: DiagnosticsVectormark;
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
  enum?: string[];
  default?: unknown;
  description?: string;
  properties?: Record<string, TraceOptionSchema>;
}

interface DrawingArtifacts {
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

// ─── Daikonic brand mark (trusted static SVG, rendered via JSX) ───────────────

function DaikonicMark({ size = 28 }: { size?: number }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 1254 1254"
      width={size}
      height={size}
      aria-hidden="true"
      style={{ flexShrink: 0, display: "block" }}
    >
      <path
        fill="#031D31"
        d="M622.5 360.5 L625.1 360.2 C625.87 360.09 626.47 359.37 626.43 358.59 Q625.75 343.98 627.5 329 Q628.45 250.49 670.5 189 Q679.51 178 688.5 167 Q740.36 114.5 809 102.5 Q821.75 101.39 835 101.5 Q858.06 103.11 872.5 122 Q876 129.79 878.5 138 Q882.17 172.19 870.5 202 Q860.07 226.77 843.5 249 Q791.62 308.08 721 325.5 Q701.08 332.4 680 336.5 Q678 337.5 676 338.5 Q655.27 341.77 639.5 357 Q638.61 358.84 638.58 359.35 C638.54 360.26 639.24 361.02 640.15 361.03 Q666.64 361.59 693 366.5 Q798.97 379.91 876.5 462 Q885.86 474.06 895.5 486 Q913.75 513.15 924.8 541.23 C925.19 542.21 925.23 543.81 924.9 544.8 L924.6 545.7 C924.27 546.69 923.15 547.5 922.1 547.5 L622.5 547.5 L322.9 547.5 C321.85 547.5 320.73 546.69 320.4 545.7 L320.1 544.8 C319.77 543.81 319.81 542.21 320.2 541.23 Q331.25 513.15 349.5 486 Q359.14 474.06 368.5 462 Q446.03 379.91 552 366.5 Q578.36 361.59 604.85 361.03 C605.76 361.02 606.46 360.26 606.42 359.35 Q606.39 358.84 605.5 357 Q589.73 341.77 569 338.5 Q567 337.5 565 336.5 Q543.92 332.4 524 325.5 Q453.38 308.08 401.5 249 Q384.93 226.77 374.5 202 Q362.83 172.19 366.5 138 Q369 129.79 372.5 122 Q386.94 103.11 410 101.5 Q423.25 101.39 436 102.5 Q504.64 114.5 556.5 167 Q565.49 178 574.5 189 Q616.55 250.49 617.5 329 Q619.25 343.98 618.57 358.59 C618.53 359.37 619.13 360.09 619.9 360.2 L622.5 360.5 Z"
      />
      <path
        fill="#1FAB9F"
        d="M622.5 578.5 L928.8 578.5 C931.12 578.5 934.2 579.95 935.69 581.73 Q936.51 582.73 937.5 588 Q939.82 606.08 942.5 624 Q943.77 667.77 936.5 708 Q935.31 710.31 934.76 710.6 C933.79 711.1 932.11 711.5 931.02 711.5 L622.5 711.5 L313.98 711.5 C312.89 711.5 311.21 711.1 310.24 710.6 Q309.69 710.31 308.5 708 Q301.23 667.77 302.5 624 Q305.18 606.08 307.5 588 Q308.49 582.73 309.31 581.73 C310.8 579.95 313.88 578.5 316.2 578.5 L622.5 578.5 Z"
      />
      <path
        fill="#FE830E"
        d="M622.5 743.5 L920.9 744.96 C925.1 744.98 927.74 748.32 926.8 752.41 Q913.08 812.26 879.5 872 Q877.75 873.75 877.4 874.1 C876.63 874.87 875.11 875.5 874.02 875.5 L622.5 875.5 L370.98 875.5 C369.89 875.5 368.37 874.87 367.6 874.1 Q367.25 873.75 365.5 872 Q331.92 812.26 318.2 752.41 C317.26 748.32 319.9 744.98 324.1 744.96 L622.5 743.5 Z"
      />
      <path
        fill="#ED3125"
        d="M622.5 907.5 L848.9 907.98 C853.1 907.99 854.51 910.76 852.05 914.16 Q846.69 921.58 836.5 935 Q786 991.62 723 1021.5 Q707.96 1030.4 693 1039.5 Q675.27 1052.08 660.5 1069 Q638.41 1098.85 631.5 1135 Q629.99 1137.85 629.37 1138.68 C628.61 1139.68 627.02 1140.77 625.8 1141.1 L622.5 1142 L619.2 1141.1 C617.98 1140.77 616.39 1139.68 615.63 1138.68 Q615.01 1137.85 613.5 1135 Q606.59 1098.85 584.5 1069 Q569.73 1052.08 552 1039.5 Q537.04 1030.4 522 1021.5 Q459 991.62 408.5 935 Q398.31 921.58 392.95 914.16 C390.49 910.76 391.9 907.99 396.1 907.98 L622.5 907.5 Z"
      />
    </svg>
  );
}

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
  preprocess: { max_size_px: 2048, preserve_transparency: true, quantize: false },
};

const FALLBACK_ENUMS: Record<string, string[]> = {
  refine: ["auto", "none"],
  trace_level: ["pixel", "subpixel"],
  fit_strategy: ["quadratic", "progressive", "progressive_allow_lines"],
  remove_background: ["auto", "off", "on"],
};

type ArtifactKind = "svg" | "preview" | "review_panel" | "labeled_svg" | "path_map" | "command_map" | "raw_trace" | "plan" | "versions";

const ARTIFACTS: Array<{ kind: ArtifactKind; label: string }> = [
  { kind: "svg", label: "SVG" },
  { kind: "preview", label: "Preview PNG" },
  { kind: "review_panel", label: "Review panel" },
  { kind: "labeled_svg", label: "Region map SVG" },
  { kind: "path_map", label: "Path segment map SVG" },
  { kind: "command_map", label: "Command map SVG" },
  { kind: "raw_trace", label: "Raw trace JSON" },
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
    "Please rerun `trace_drawing` using the original attached image and these settings from the VectorMark widget.",
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

function artifactJson(result: CallToolResult | null): string | null {
  const structured = result?.structuredContent as { trace?: unknown; plan?: unknown; versions?: unknown } | undefined;
  const value = structured?.trace ?? structured?.plan ?? structured?.versions;
  if (value) return JSON.stringify(value, null, 2);
  const text = result ? firstText(result) : undefined;
  if (!text) return null;
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

function versionLineage(drawing: DrawingResult): string {
  const fields = [
    `Requested base ${drawing.requested_base_version ?? "none"}`,
    `Actual parent ${drawing.parent_version ?? "none"}`,
    `Baseline ${drawing.baseline_version}`,
    `Current ${drawing.version}`,
  ];
  return fields.join(" · ");
}

function TraceControls({
  app,
  initialInput,
  result,
  onResult,
  onError,
}: {
  app: App;
  initialInput: TraceToolInput;
  result: WidgetResult | null;
  onResult: (result: CallToolResult) => void;
  onError: (message: string | null) => void;
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

  const lastTraceOptions = result && isDrawingResult(result) ? result.trace_options : undefined;
  useEffect(() => {
    setForm(lastTraceOptions ?? {
      ...DEFAULT_TRACE_OPTIONS,
      ...initialInput.options,
        preprocess: { ...DEFAULT_TRACE_OPTIONS.preprocess, ...initialInput.options?.preprocess },
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
    onError(null);
    try {
      onResult(await app.callServerTool({
        name: "trace_drawing",
        arguments: { ...(sourceImage.download_url ? { image: sourceImage } : { image_data: sourceImage }), options: form },
      }));
    } catch (runError) {
      onError(runError instanceof Error ? runError.message : String(runError));
    } finally {
      setIsRunning(false);
    }
  }, [app, form, onError, onResult, sourceImage]);

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
      <div className="brand-strip">
        <div className="brand-id">
          <DaikonicMark size={32} />
          <div><p className="eyebrow">vectormark</p><h1>Trace & refine</h1></div>
        </div>
        <span className="status-dot" aria-label={isRunning ? "Tracing" : "Ready"} />
      </div>

      <p className="control-note">
        {sourceImage
          ? "Reruns use the original attached image."
          : sourceUnavailable
            ? "ChatGPT supplied a display-only attachment path. Ask the agent to rerun it with these settings."
            : "Run trace_drawing on an attached image to enable reruns."}
      </p>

      <div className="select-grid">
        {enumSelect("refine", "Refine")}
        {enumSelect("trace_level", "Boundary")}
        {enumSelect("fit_strategy", "Path fitting")}
        {enumSelect("remove_background", "Background")}
      </div>

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
              <label className="toggle"><input checked={form.initial_stitch} type="checkbox" onChange={(event) => setOption("initial_stitch", event.target.checked)} /> Initial stitch in v0</label>
            <label className="toggle"><input checked={form.corner_normalize} type="checkbox" onChange={(event) => setOption("corner_normalize", event.target.checked)} /> Normalize corners in v1</label>
            <label className="toggle"><input checked={form.corner_rounding} type="checkbox" onChange={(event) => setOption("corner_rounding", event.target.checked)} /> Protect rounded corners in v0</label>
            <label className="toggle"><input checked={form.support_weighted_corners} type="checkbox" onChange={(event) => setOption("support_weighted_corners", event.target.checked)} /> Support-weighted corners</label>
            <label className="toggle"><input checked={form.gradient_pretrace} type="checkbox" onChange={(event) => setOption("gradient_pretrace", event.target.checked)} /> Gradient pretrace</label>
          <label className="toggle"><input checked={form.preprocess.preserve_transparency} type="checkbox" onChange={(event) => setForm((current) => ({ ...current, preprocess: { ...current.preprocess, preserve_transparency: event.target.checked } }))} /> Preserve alpha</label>
          <label className="toggle"><input checked={form.preprocess.quantize} type="checkbox" onChange={(event) => setForm((current) => ({ ...current, preprocess: { ...current.preprocess, quantize: event.target.checked } }))} /> Pre-quantize</label>
        </div>
          </details>

          <details className="trace-advanced">
            <summary>Automatic refinement passes</summary>
            <p className="control-note">Used only with refine=auto. Pass order is fixed; final source fills are always resolved.</p>
            <div className="toggle-row">
              {Object.entries(schema?.refinement_passes?.properties ?? {}).map(([name, definition]) => (
                <label className="toggle" key={name} title={definition.description}>
                  <input type="checkbox" disabled={form.refine !== "auto"}
                    checked={form.refinement_passes?.[name] ?? Boolean(definition.default)}
                    onChange={(event) => setOption("refinement_passes", { ...form.refinement_passes, [name]: event.target.checked })} />
                  {name.replaceAll("_", " ")}
                </label>
              ))}
            </div>
          </details>

        {lastTraceOptions && <details className="trace-advanced">
          <summary>Last trace parameters</summary>
          <pre className="artifact-json">{JSON.stringify(lastTraceOptions, null, 2)}</pre>
          {result && isDrawingResult(result) && <>
            <p className="control-note">Resolved trace diagnostics (including selected palette size):</p>
            <pre className="artifact-json">{JSON.stringify(result.trace?.options, null, 2)}</pre>
          </>}
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

function ArtifactViewer({ app, drawing, initialPanel, onError }: { app: App; drawing: DrawingResult; initialPanel: string | null; onError: (message: string | null) => void }) {
  const [selected, setSelected] = useState<ArtifactKind>("review_panel");
  const [loaded, setLoaded] = useState<CallToolResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [exported, setExported] = useState<{ url: string; filename: string } | null>(null);

  useEffect(() => { setSelected("review_panel"); setLoaded(null); setExported(null); }, [drawing.drawing_id, drawing.version]);

  const fetchArtifact = useCallback((kind: ArtifactKind) => app.callServerTool({
    name: "get_drawing_artifact",
    arguments: { drawing_id: drawing.drawing_id, version: drawing.version, artifact: kind },
  }), [app, drawing.drawing_id, drawing.version]);

  const load = useCallback(async (kind: ArtifactKind) => {
    setSelected(kind);
    setExported(null);
    setLoaded(null);
    setLoading(true);
    onError(null);
    try {
      setLoaded(await fetchArtifact(kind));
    } catch (loadError) {
      onError(loadError instanceof Error ? loadError.message : String(loadError));
    } finally {
      setLoading(false);
    }
  }, [fetchArtifact, onError]);

  const svg = artifactSvg(loaded);
  const image = firstImageDataUri(loaded) ?? (selected === "review_panel" ? initialPanel : null);
  const json = ["raw_trace", "plan", "versions"].includes(selected) ? artifactJson(loaded) : null;
  const uri = drawing.artifacts[selected];

  const exportFile = async () => {
    setLoading(true);
    setExported(null);
    onError(null);
    try {
      const result = await fetchArtifact(selected);
      setLoaded(result);
      const svgContent = artifactSvg(result);
      const imageContent = firstImageDataUri(result);
      const jsonContent = ["raw_trace", "plan", "versions"].includes(selected) ? artifactJson(result) : null;
      const extension = svgContent ? "svg" : imageContent ? "png" : jsonContent ? "json" : null;
      if (!extension) throw new Error("This response did not contain a downloadable artifact.");
      const href = svgContent ? dataUriForSvg(svgContent) : imageContent ?? `data:application/json;charset=utf-8,${encodeURIComponent(jsonContent!)}`;
      const filename = `${drawing.drawing_id}-${drawing.version}-${selected}.${extension}`;
      const host = (window as Window & { openai?: {
        uploadFile?: (file: File) => Promise<{ fileId: string }>;
        getFileDownloadUrl?: (input: { fileId: string }) => Promise<{ downloadUrl: string }>;
      } }).openai;
      if (host) {
        if (!host.uploadFile || !host.getFileDownloadUrl) {
          throw new Error("This ChatGPT host does not expose file export APIs. SVG text is still available through get_drawing_artifact; no downloadable file was created.");
        }
        const blob = svgContent
          ? new Blob([svgContent], { type: "image/svg+xml" })
          : imageContent
            ? new Blob([Uint8Array.from(atob(imageContent.split(",")[1]), (char) => char.charCodeAt(0))], { type: "image/png" })
            : new Blob([jsonContent!], { type: "application/json" });
        const { fileId } = await host.uploadFile(new File([blob], filename, { type: blob.type }));
        if (!fileId) throw new Error("ChatGPT did not return an exported file ID.");
        const { downloadUrl } = await host.getFileDownloadUrl({ fileId });
        if (!downloadUrl || new URL(downloadUrl).protocol !== "https:") {
          throw new Error("ChatGPT did not return a valid file download URL.");
        }
        setExported({ url: downloadUrl, filename });
        await app.updateModelContext({ content: [{ type: "text", text: `VectorMark exported ${filename} as ChatGPT file ${fileId}. Download: ${downloadUrl}` }] });
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
      setLoading(false);
    }
  };

  return <>
    <div className="artifact-header"><div><p className="eyebrow">Drawing {drawing.version}</p><h2>Response files</h2></div><code>{drawing.drawing_id}</code></div>
    <code className="artifact-uri" aria-label="version lineage">{versionLineage(drawing)}</code>
    <div className="artifact-list" role="group" aria-label="Response files">
        {ARTIFACTS.map(({ kind, label }) => <button key={kind} disabled={loading} className={`artifact-button${selected === kind ? " active" : ""}`} onClick={() => load(kind)}>{label}</button>)}
        <button className="artifact-button" disabled={loading} onClick={exportFile}>Export selected file</button>
    </div>
    {exported && <p><a href={exported.url} target="_blank" rel="noopener noreferrer" download={exported.filename}>Download {exported.filename}</a> (temporary link)</p>}
    <code className="artifact-uri">{uri}</code>
    <div className="preview-stage">
      {loading ? <p>Loading {selected}…</p> : image ? <img src={image} alt={`${selected} response artifact`} /> : svg ? <img src={dataUriForSvg(svg)} alt={`${selected} response artifact`} /> : json ? <pre className="artifact-json">{json}</pre> : <p>Select a response file to load it through VectorMark.</p>}
    </div>
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

  const { app, isConnected, error } = useApp({
    appInfo: { name: "vectormark", version: "0.2.1" }, capabilities: {},
    onAppCreated: (createdApp) => {
      createdApp.ontoolinput = (params) => {
        const input = (params.arguments ?? {}) as TraceToolInput;
        if (input.image !== undefined || input.image_data !== undefined || input.options !== undefined) setToolInput(input);
      };
      createdApp.ontoolresult = (result) => { setErrorText(null); setToolResult(result); };
      createdApp.ontoolcancelled = (params) => setErrorText(params.reason || "Tool call was cancelled.");
      createdApp.onerror = (appError) => setErrorText(appError instanceof Error ? appError.message : String(appError));
      createdApp.onhostcontextchanged = (params) => setHostContext((current) => ({ ...current, ...params }));
      createdApp.onteardown = async () => ({});
    },
  });
  useEffect(() => { if (app) setHostContext(app.getHostContext()); }, [app]);
  const result = useMemo(() => parseStructuredResult(toolResult), [toolResult]);
  const panel = useMemo(() => firstImageDataUri(toolResult), [toolResult]);

  if (error) return <StatusPanel title="Connection failed" detail={error.message} />;
  if (!isConnected || !app) return <StatusPanel title="Connecting" detail="Waiting for the MCP Apps host." />;

  return <main className="app-shell" style={{ paddingTop: hostContext?.safeAreaInsets?.top, paddingRight: hostContext?.safeAreaInsets?.right, paddingBottom: hostContext?.safeAreaInsets?.bottom, paddingLeft: hostContext?.safeAreaInsets?.left }}>
    <TraceControls app={app} initialInput={toolInput} result={result} onResult={setToolResult} onError={setErrorText} />
    <section className="preview-panel" aria-label="drawing response">
      {errorText && <div className="error-banner">{errorText}</div>}
      {result ? isDrawingResult(result) ? <ArtifactViewer app={app} drawing={result} initialPanel={panel} onError={setErrorText} /> : <LegacyPreview result={result} /> : <EmptyPreview />}
    </section>
  </main>;
}

function EmptyPreview() { return <div className="empty-preview"><div className="empty-glyph" /><p>Trace an image to inspect its review panel and response files.</p></div>; }
function StatusPanel({ title, detail }: { title: string; detail: string }) { return <main className="status-panel"><h1>{title}</h1><p>{detail}</p></main>; }

createRoot(document.getElementById("root")!).render(<StrictMode><AppRoot /></StrictMode>);
