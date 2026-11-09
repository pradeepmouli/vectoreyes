import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const bundle = process.argv[2];
if (!bundle) throw new Error('Pass a saved evaluation bundle directory. No live drawing is used.');
const kinds = new Set(['svg', 'preview', 'review_panel', 'labeled_svg', 'path_map', 'command_map', 'raw_trace', 'plan', 'versions', 'report']);
const root = resolve(bundle, 'versions');
const response = JSON.parse(await readFile(resolve(root, 'v1', 'response.json'), 'utf8'));
createServer(async (request, result) => {
  result.setHeader('Access-Control-Allow-Origin', 'http://127.0.0.1:62747');
  result.setHeader('Content-Type', 'application/json');
  try {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/response') return result.end(JSON.stringify(response));
    const version = url.searchParams.get('version');
    const kind = url.searchParams.get('kind');
    if (url.pathname !== '/artifact' || !['v0', 'v1'].includes(version) || !kinds.has(kind)) {
      result.statusCode = 404; return result.end('{}');
    }
    if (['review_panel', 'preview'].includes(kind)) {
      const data = await readFile(resolve(root, version, `${kind}.png`));
      return result.end(JSON.stringify({ content: [{ type: 'image', mimeType: 'image/png', data: data.toString('base64') }] }));
    }
    let data;
    if (kind === 'versions') {
      const versions = await Promise.all(['v0', 'v1'].map(async (id) => {
        const saved = JSON.parse(await readFile(resolve(root, id, 'response.json'), 'utf8'));
        return { version: saved.version, parent_version: saved.parent_version, label: id === 'v0' ? 'Trace' : 'Refine' };
      }));
      data = { mime_type: 'application/json', versions };
    } else if (kind === 'report') {
      const saved = JSON.parse(await readFile(resolve(root, version, 'response.json'), 'utf8'));
      data = { mime_type: 'application/json', report: saved.report };
    } else data = JSON.parse(await readFile(resolve(root, version, `${kind}.json`), 'utf8'));
    result.end(JSON.stringify({ content: [], structuredContent: data }));
  } catch {
    result.end(JSON.stringify({ isError: true, content: [{ type: 'text', text: 'Saved fixture artifact is unavailable.' }] }));
  }
}).listen(62746, '127.0.0.1', () => console.log('Saved-artifact fixture server: http://127.0.0.1:62746'));
