import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readArtifact } from '../src/artifactDelivery.ts';

test('artifact pages preserve exact export bytes and checksum', async () => {
  const original = { content: [], structuredContent: { svg: '<svg>🍊</svg>' } };
  const raw = new TextEncoder().encode(JSON.stringify(original));
  const digest = Buffer.from(await crypto.subtle.digest('SHA-256', raw)).toString('hex');
  const result = await readArtifact(async args => {
    const offset = args.offset ?? 0, end = Math.min(raw.length, offset + 7);
    return { content: [], structuredContent: { chunk: { encoding: 'base64-json-utf8', offset,
      next_offset: end < raw.length ? end : null, total_bytes: raw.length, sha256: digest,
      data: Buffer.from(raw.slice(offset,end)).toString('base64') } } };
  }, { artifact: 'svg' });
  assert.deepEqual(result, original);
});

test('changed artifact pages are rejected', async () => {
  await assert.rejects(readArtifact(async () => ({content: [], structuredContent: {chunk: {
    encoding: 'base64-json-utf8', offset: 0, next_offset: null, total_bytes: 1,
    sha256: 'invalid', data: 'eA=='
  }}}), {artifact:'svg'}), /checksum/);
});
