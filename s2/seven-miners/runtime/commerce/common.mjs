// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
export const ORIGIN = 'https://telegraph-s2-hashprice.margyn.workers.dev';
export const LIMITS = Object.freeze({ timeoutMs: 6000, bodyBytes: 1048576, events: 256, text: 128 });
export class Refusal extends Error {
  constructor(code, status = 422) { super(code); this.code = code; this.status = status; }
}
export function requireValue(condition, code, status) {
  if (!condition) throw new Refusal(code, status);
}
export function text(value, name) {
  requireValue(typeof value === 'string', `missing_${name}`);
  const normalized = value.normalize('NFC').trim();
  requireValue(normalized.length > 0 && normalized.length <= LIMITS.text && !/[\x00-\x1f\x7f]/u.test(normalized), `invalid_${name}`);
  return normalized;
}
export function timestamp(value, name) {
  requireValue(typeof value==='string' && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$/.test(value),`invalid_${name}`);
  const parsed=Date.parse(value);
  requireValue(Number.isFinite(parsed),`invalid_${name}`);
  return parsed;
}
export async function bounded(task, ms = LIMITS.timeoutMs) {
  let timer;
  try {
    return await Promise.race([task, new Promise((_, reject) => { timer = setTimeout(() => reject(new Refusal('upstream_timeout', 504)), ms); })]);
  } finally { clearTimeout(timer); }
}
export async function jsonBody(response) {
  requireValue(response.ok, 'upstream_http_error', 502);
  requireValue(response.headers.get('content-type')?.includes('json'), 'upstream_not_json', 502);
  const reader = response.body.getReader();
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      length += value.length;
      requireValue(length <= LIMITS.bodyBytes, 'upstream_too_large', 502);
      chunks.push(value);
    }
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
  catch { throw new Refusal('upstream_invalid_json', 502); }
}
export function reply(body, status = 200) {
  return Response.json(body, { status, headers: {
    'cache-control': 'no-store', 'x-content-type-options': 'nosniff',
    'content-security-policy': "default-src 'none'; frame-ancestors 'none'", 'referrer-policy': 'no-referrer',
  } });
}
export async function guarded(fn) {
  try { return reply(await fn()); }
  catch (error) { return reply({ error: error instanceof Refusal ? error.code : 'source_unavailable' }, error instanceof Refusal ? error.status : 503); }
}
export function params(request, allowed) {
  requireValue(request.method === 'GET', 'method_not_allowed', 405);
  const url = new URL(request.url);
  requireValue(url.href.length <= 2048, 'request_too_large', 413);
  for (const key of url.searchParams.keys()) {
    requireValue(allowed.includes(key) && url.searchParams.getAll(key).length === 1, 'unknown_or_duplicate_parameter');
  }
  return url.searchParams;
}
