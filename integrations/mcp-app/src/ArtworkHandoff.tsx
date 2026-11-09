import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";

interface ArtifactOption { kind: string; label: string }
export interface Revision { version: string; parent_version: string | null; label?: string }

/** Parent links are authoritative; version identifiers are opaque. */
export function revisionBranch(revisions: Revision[], leaf: string): Revision[] {
  const byId = new Map(revisions.map((revision) => [revision.version, revision]));
  const branch: Revision[] = [];
  const visited = new Set<string>();
  let cursor = byId.get(leaf);
  while (cursor && !visited.has(cursor.version)) {
    branch.unshift(cursor);
    visited.add(cursor.version);
    cursor = cursor.parent_version ? byId.get(cursor.parent_version) : undefined;
  }
  return branch;
}

export function revisionLeaves(revisions: Revision[]): Revision[] {
  const parents = new Set(revisions.map((revision) => revision.parent_version));
  return revisions.filter((revision) => !parents.has(revision.version));
}

export function GeneratingIndicator({ label = "Creating SVG…", compact = false }: { label?: string; compact?: boolean }) {
  const field = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const node = field.current;
    if (!node) return;
    let visible = true;
    const update = () => { node.dataset.paused = String(!visible || document.hidden); };
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; update(); });
    observer.observe(node);
    document.addEventListener("visibilitychange", update);
    update();
    return () => { observer.disconnect(); document.removeEventListener("visibilitychange", update); };
  }, []);
  return <div ref={field} className={`generating-indicator${compact ? " compact" : ""}`} role="status">
    <div className="dot-grid" aria-hidden="true">
      {Array.from({ length: 96 }, (_, index) => <i key={index} style={{ "--dot-delay": `${((index % 12) + Math.floor(index / 12)) * -0.11}s` } as CSSProperties} />)}
    </div>
    <span>{label}</span>
  </div>;
}

function RevisionSlide({ revision, active, loadPreview, children, position, total, stageClass, stageStyle, loading }: {
  revision: Revision; active: boolean; loadPreview: (version: string) => Promise<string>; children: ReactNode;
  position: number; total: number; stageClass: string; stageStyle: CSSProperties; loading: boolean;
}) {
  const slide = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(active);
  const [image, setImage] = useState<{ loader: typeof loadPreview; src: string } | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const node = slide.current;
    if (!node) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setVisible(true); observer.disconnect(); }
    }, { root: node.closest(".revision-carousel"), rootMargin: "0px 100%" });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible || active) return;
    let current = true;
    setFailed(false);
    loadPreview(revision.version).then((src) => { if (current) setImage({ loader: loadPreview, src }); }).catch(() => { if (current) setFailed(true); });
    return () => { current = false; };
  }, [visible, active, loadPreview, revision.version]);
  return <div ref={slide} className="revision-slide" role="group" aria-roledescription="slide" aria-label={`Revision ${revision.version}, ${position + 1} of ${total}`} aria-hidden={!active} data-version={revision.version}>
    <section className={stageClass} style={stageStyle} aria-label={active ? `Selected artwork artifact for ${revision.version}` : undefined} aria-busy={active && loading}>
      {active ? children : image?.loader === loadPreview ? <img src={image.src} alt="" /> : <p>{failed ? "Preview unavailable" : "Loading revision…"}</p>}
    </section>
  </div>;
}

export interface ArtworkHandoffProps {
  version: string;
  revisions: Revision[];
  loadPreview: (version: string) => Promise<string>;
  selected: string;
  loading: boolean;
  exporting?: boolean;
  generating?: boolean;
  unavailable?: boolean;
  reviewOpen: boolean;
  editRequest: string;
  sending: boolean;
  sent: boolean;
  views: ArtifactOption[];
  maps: ArtifactOption[];
  children: ReactNode;
  reviewContent?: ReactNode;
  onSelect: (kind: string) => void;
  onVersionChange: (version: string) => void;
  onDownloadSvg: () => void;
  onToggleReview: () => void;
  onEditRequestChange: (value: string) => void;
  onRequestEdit: () => void;
}

export function ArtworkHandoff(props: ArtworkHandoffProps) {
  const { selected, loading, reviewOpen, revisions, version } = props;
  const leaves = revisionLeaves(revisions);
  const [leaf, setLeaf] = useState(version);
  const activeLeaf = leaves.some((item) => item.version === leaf) && revisionBranch(revisions, leaf).some((item) => item.version === version)
    ? leaf : leaves.find((item) => revisionBranch(revisions, item.version).some((entry) => entry.version === version))?.version ?? version;
  const branch = revisionBranch(revisions, activeLeaf);
  const position = branch.findIndex((item) => item.version === version);
  const [zoom, setZoom] = useState(1);
  const carousel = useRef<HTMLDivElement>(null);
  const carouselWidth = useRef(0);
  const currentPosition = useRef(position);
  currentPosition.current = position;
  const scrollSelection = useRef<string | null>(null);
  const userScrolling = useRef(false);
  const branchKey = branch.map((item) => item.version).join("\n");
  const previousBranch = useRef(branchKey);
  useLayoutEffect(() => {
    const node = carousel.current;
    if (!node || position < 0) return;
    carouselWidth.current = node.clientWidth;
    if (scrollSelection.current !== version || previousBranch.current !== branchKey) {
      userScrolling.current = false;
      node.scrollLeft = position * node.clientWidth;
    }
    scrollSelection.current = null;
    previousBranch.current = branchKey;
  }, [version, position, branchKey]);
  useEffect(() => {
    const node = carousel.current;
    if (!node) return;
    const observer = new ResizeObserver(() => {
      if (node.clientWidth !== carouselWidth.current) { userScrolling.current = false; carouselWidth.current = node.clientWidth; node.scrollLeft = Math.max(0, currentPosition.current) * node.clientWidth; }
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  const stageClass = `preview-stage${selected.endsWith("map") || selected === "labeled_svg" ? " map-stage" : ""}${reviewOpen && zoom > 1 ? " zoomed-stage" : ""}`;
  return <>
    <div className="artwork-heading">
      <div><h1>Your vector artwork</h1><p>{props.unavailable ? "Preview unavailable. Retry or choose another view." : "Ready to export. Yours to refine."}</p></div>
      <button className="primary-button download-svg" disabled={props.exporting} onClick={props.onDownloadSvg}>
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5" /></svg>
        {props.exporting ? "Exporting SVG…" : "Export SVG"}
      </button>
    </div>
    <div ref={carousel} className="revision-carousel" role="region" aria-roledescription="carousel" aria-label="Drawing revisions" tabIndex={0}
      onPointerDown={() => { userScrolling.current = true; }}
      onWheel={(event) => { if (event.deltaX || event.shiftKey) userScrolling.current = true; }}
      onScrollEnd={() => { userScrolling.current = false; }}
      onScroll={(event) => {
        const node = event.currentTarget;
        if (!node.clientWidth) return;
        // A resize is layout, not a user's request to change revision.
        if (node.clientWidth !== carouselWidth.current) {
          userScrolling.current = false;
          carouselWidth.current = node.clientWidth;
          node.scrollLeft = Math.max(0, currentPosition.current) * node.clientWidth;
          return;
        }
        if (!userScrolling.current) {
          const expected = Math.max(0, currentPosition.current) * node.clientWidth;
          if (Math.abs(node.scrollLeft - expected) > 1) node.scrollLeft = expected;
          return;
        }
        const next = branch[Math.round(node.scrollLeft / node.clientWidth)];
        if (next && next.version !== version) { scrollSelection.current = next.version; props.onVersionChange(next.version); }
      }}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget) return;
        const next = event.key === "ArrowLeft" ? position - 1 : event.key === "ArrowRight" ? position + 1 : -1;
        if (branch[next]) { event.preventDefault(); props.onVersionChange(branch[next].version); }
      }}>
      {branch.map((revision, index) => <RevisionSlide key={revision.version} revision={revision} active={revision.version === version} position={index} total={branch.length} loadPreview={props.loadPreview} stageClass={stageClass} stageStyle={{ "--review-zoom": zoom } as CSSProperties} loading={loading}>{props.children}</RevisionSlide>)}
    </div>
    <nav className="revision-navigation" aria-label="Revision controls">
      <div className="revision-heading"><span aria-live="polite"><strong>{version}</strong> · {branch[position]?.label ?? "Revision"} · {position + 1} of {branch.length}</span>
        {leaves.length > 1 && <label className="branch-select">Branch
          <select aria-label="Revision branch" value={activeLeaf} onChange={(event) => { setLeaf(event.target.value); props.onVersionChange(event.target.value); }}>
            {leaves.map((item) => <option key={item.version} value={item.version}>Ending at {item.version}{item.label ? ` · ${item.label}` : ""}</option>)}
          </select></label>}
        <div className="revision-arrows"><button type="button" aria-label="Previous revision" disabled={position <= 0} onClick={() => props.onVersionChange(branch[position - 1].version)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m14 6-6 6 6 6" /></svg></button><button type="button" aria-label="Next revision" disabled={position < 0 || position >= branch.length - 1} onClick={() => props.onVersionChange(branch[position + 1].version)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m10 6 6 6-6 6" /></svg></button></div>
      </div>
    </nav>
    {props.generating && <div className="generation-status"><GeneratingIndicator compact label="Creating next revision…" /></div>}
    <div className="review-entry"><span>{reviewOpen ? `Reviewing ${version} · all views follow this revision` : "Your SVG is ready. Review is optional."}</span><button className="secondary-button" aria-expanded={reviewOpen} aria-controls="review-workspace" onClick={props.onToggleReview}>{reviewOpen ? "Close review" : "Review & edit"}</button></div>
    <div id="review-workspace" className={`artwork-layout${reviewOpen ? " reviewing" : ""}`}>
      {reviewOpen && <div className="review-view-controls"><h2>Review &amp; edit</h2><div className="view-options" role="group" aria-label="Review view">
        {[...props.views, ...props.maps].map(({ kind, label }) => <button key={kind} type="button" aria-pressed={selected === kind} className={selected === kind ? "view-option selected" : "view-option"} onClick={() => props.onSelect(kind)}>{label}</button>)}
      </div><label className="review-zoom">View scale <select aria-label="Review zoom" value={zoom} onChange={(event) => setZoom(Number(event.target.value))}><option value={1}>Fit</option><option value={2}>2×</option><option value={4}>4×</option></select></label></div>}
      {reviewOpen && <aside className="review-sidebar" aria-label={`Edit revision ${version}`}>
        <h2>What would you change?</h2><p>Use the colored region and segment maps to identify the part. Your agent will create a new revision from {version}.</p>
        <div className="review-tools">
          <label className="field">What would you like to change?<textarea rows={4} value={props.editRequest} onChange={(event) => props.onEditRequestChange(event.target.value)} placeholder="For example: make the lower edge of the red tip one smooth curve." /></label>
          <button className="primary-button" disabled={!props.editRequest.trim() || props.sending || props.sent} onClick={props.onRequestEdit}>{props.sending ? "Sending request…" : props.sent ? "Request sent" : "Ask agent to revise"}</button>
          {props.sent && <p role="status">Your agent has the drawing and revision details.</p>}
        </div>
        <p className="review-note">Shape and fill are evaluated separately.</p>
      </aside>}
    </div>
    {reviewOpen && <div className="review-evidence">{props.reviewContent}</div>}
  </>;
}
