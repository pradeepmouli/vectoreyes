import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { registerHooks } from 'node:module';
import test from 'node:test';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

registerHooks({
  load(url, context, nextLoad) {
    if (url.endsWith('.tsx')) return { format: 'module', shortCircuit: true,
      source: ts.transpileModule(readFileSync(new URL(url), 'utf8'), {
        compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext },
      }).outputText };
    return nextLoad(url, context);
  },
});

const module = await import('../src/ArtworkHandoff.tsx').catch(() => ({}));
const base = {
  version: 'v2', revisions: [{ version: 'v0', parent_version: null }, { version: 'v1', parent_version: 'v0' }, { version: 'v2', parent_version: 'v1' }], loadPreview: async () => 'data:image/svg+xml,test', selected: 'svg', loading: false,
  reviewOpen: false, editRequest: '', sending: false, sent: false,
  views: [{ kind: 'svg', label: 'Artwork' }, { kind: 'review_panel', label: 'Source & difference' }],
  maps: [{ kind: 'labeled_svg', label: 'Regions' }, { kind: 'path_map', label: 'Segments & features' }],
  files: [{ kind: 'svg', label: 'SVG' }, { kind: 'plan', label: 'Plan JSON' }],
  onSelect() {}, onVersionChange() {}, onDownloadSvg() {}, onExportSelected() {},
  onToggleReview() {}, onEditRequestChange() {}, onRequestEdit() {},
};

function render(props = {}) {
  assert.equal(typeof module.ArtworkHandoff, 'function', 'ArtworkHandoff must provide the artwork-first view');
  return renderToStaticMarkup(React.createElement(module.ArtworkHandoff, { ...base, ...props },
    React.createElement('img', { src: 'data:image/svg+xml,test', alt: 'Selected vector artwork' })));
}

test('artwork precedes review and direct SVG export is always visible', () => {
  const html = render();
  assert.match(html, /Your vector artwork/);
  assert.match(html, /Export SVG/);
  assert.match(html, /Review &amp; edit/);
  assert.doesNotMatch(html, /textarea/);
  assert.doesNotMatch(html, /Source &amp; difference|Response file|Export selected file|Plan JSON/);
});

test('review exposes existing maps and an optional agent edit request', () => {
  const html = render({ reviewOpen: true });
  assert.match(html, /Segments &amp; features/);
  assert.match(html, /What would you like to change/);
  assert.match(html, /Ask agent to revise/);
});

test('revision selection is scoped to the current drawing and version', () => {
  const html = render();
  assert.match(html, /aria-label="Drawing revisions"/);
  assert.match(html, /aria-label="Revision v2, 3 of 3" aria-hidden="false"/);
  assert.equal((html.match(/<img /g) ?? []).length, 1, 'Only one selected picture, with no thumbnail strip or duplicate preview');
  assert.doesNotMatch(html, /revision-thumb|revision-strip/);
  assert.doesNotMatch(html, /accepted|approved|quality score/i);
});

test('empty edits cannot be submitted and progress is announced', () => {
  const html = render({ reviewOpen: true, loading: true });
  assert.match(html, /disabled=""[^>]*>Ask agent to revise/);
  assert.match(html, /aria-busy="true"/);
});

test('a failed artifact does not claim it is ready to download', () => {
  const html = render({ unavailable: true });
  assert.doesNotMatch(html, /Ready to download/);
  assert.match(html, /Preview unavailable/);
});

test('branch navigation follows actual parents, not version sorting', () => {
  const revisions = [{ version: 'root', parent_version: null }, { version: 'z', parent_version: 'root' }, { version: 'a', parent_version: 'z' }, { version: 'fork', parent_version: 'root' }];
  assert.deepEqual(module.revisionBranch(revisions, 'a').map((item) => item.version), ['root', 'z', 'a']);
  assert.deepEqual(module.revisionLeaves(revisions).map((item) => item.version), ['a', 'fork']);
  assert.match(render({ revisions, version: 'a' }), /aria-label="Revision branch"/);
  assert.match(render({ revisions, version: 'z' }), /Revision a, 3 of 3/);
  assert.equal(module.revisionBranch([{ version: 'loop', parent_version: 'loop' }], 'loop').length, 1);
});

test('review evidence is absent from handoff and scoped inside review', () => {
  const reviewContent = React.createElement('div', {}, 'Exact settings for v2');
  assert.doesNotMatch(render({ reviewContent }), /Exact settings/);
  assert.match(render({ reviewContent, reviewOpen: true }), /Exact settings for v2/);
});

test('generation keeps one retained artwork with an inline pending status', () => {
  const html = render({ generating: true });
  assert.match(html, /Selected vector artwork/);
  assert.match(html, /Creating next revision/);
  assert.match(html, /dot-grid/);
  assert.doesNotMatch(html, /\d+%/);
});
