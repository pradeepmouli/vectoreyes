import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';

type Chunk = { encoding: string; offset: number; next_offset: number | null;
  total_bytes: number; sha256: string; data: string };

/** Reassemble exact artifact responses, never treat one page as an export. */
export async function readArtifact(
  call: (args: Record<string, unknown>) => Promise<CallToolResult>,
  args: Record<string, unknown>,
): Promise<CallToolResult> {
  let result = await call(args);
  const first = result.structuredContent?.chunk as Chunk | undefined;
  if (!first) return result;
  if (!Number.isSafeInteger(first.total_bytes) || first.total_bytes <= 0 || first.total_bytes > 128 * 1024 * 1024)
    throw new Error('Artifact exceeds the client assembly budget.');
  const raw = new Uint8Array(first.total_bytes);
  let offset = 0;
  while (true) {
    if (result.isError) throw new Error('Artifact page request failed.');
    const chunk = result.structuredContent?.chunk as Chunk | undefined;
    if (!chunk || chunk.encoding !== 'base64-json-utf8' || chunk.offset !== offset ||
        chunk.sha256 !== first.sha256 || chunk.total_bytes !== first.total_bytes)
      throw new Error('Artifact page identity changed during retrieval.');
    const part = Uint8Array.from(atob(chunk.data), c => c.charCodeAt(0));
    if (!part.length || offset + part.length > raw.length) throw new Error('Invalid artifact page length.');
    raw.set(part, offset); offset += part.length;
    if (chunk.next_offset === null) {
      if (offset !== raw.length) throw new Error('Incomplete artifact response.');
      break;
    }
    if (chunk.next_offset !== offset) throw new Error('Invalid artifact page offset.');
    result = await call({ ...args, offset });
  }
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', raw)),
    b => b.toString(16).padStart(2, '0')).join('');
  if (digest !== first.sha256) throw new Error('Artifact checksum mismatch.');
  return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(raw)) as CallToolResult;
}
