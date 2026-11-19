// GLEIF LEI reference data: CC0-1.0, https://www.gleif.org/en/meta/lei-data-terms-of-use II.3.
// Original normalization code: SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright 2026 zkasuran. See LICENSE and DATA-SOURCES.md. No GLEIF endorsement.
export const LIMITS = Object.freeze({ inputBytes: 256, urlBytes: 2048, records: 100,
  upstreamBytes: 786432, outputBytes: 16384, timeoutMs: 8000, snapshotAgeMs: 48 * 3600000,
  futureSkewMs: 300000, addressLines: 8, fieldBytes: 2048 });
export const GLEIF = 'https://api.gleif.org/api/v1/lei-records';
const enc = new TextEncoder();
const aliases = Object.freeze({ 'united states': 'US', usa: 'US', 'united states of america': 'US',
  california: 'US-CA', delaware: 'US-DE', 'united kingdom': 'GB', uk: 'GB',
  norway: 'NO', france: 'FR', italy: 'IT', germany: 'DE', india: 'IN', sweden: 'SE' });
export class Refusal extends Error {
  constructor(code, status = 422) { super(code); this.code = code; this.status = status; }
}
function refuse(code, status) { throw new Refusal(code, status); }
function clean(value, max = LIMITS.inputBytes) {
  if (typeof value !== 'string' || !value.trim() || enc.encode(value).length > max ||
      /[\p{Cc}\p{Cf}]/u.test(value)) refuse('invalid_input');
  return value.normalize('NFC').trim().replace(/\s+/gu, ' ');
}
export function validLEI(value) {
  if (!/^[A-Z0-9]{18}[0-9]{2}$/.test(value)) return false;
  let rem = 0;
  for (const c of value) {
    const digits = c >= 'A' && c <= 'Z' ? String(c.charCodeAt(0) - 55) : c;
    for (const d of digits) rem = (rem * 10 + Number(d)) % 97;
  }
  return rem === 1;
}
export function requestPlan(company, jurisdiction) {
  company = clean(company); jurisdiction = clean(jurisdiction);
  const jur = aliases[jurisdiction.toLowerCase()] || jurisdiction.toUpperCase();
  if (!/^[A-Z]{2}(?:-[A-Z0-9]{1,3})?$/.test(jur)) refuse('unsupported_jurisdiction');
  let mode = 'name', value = company, authority = null;
  const candidate = company.replace(/^LEI\s*:\s*/i, '').toUpperCase();
  if (/^[A-Z0-9]{18}[0-9]{2}$/.test(candidate) || /^LEI\s*:/i.test(company)) {
    if (!validLEI(candidate)) refuse('invalid_lei');
    mode = 'lei'; value = candidate;
  } else {
    const registry = company.match(/^(RA[0-9]{6}):(.+)$/i);
    if (registry) { mode = 'registry'; authority = registry[1].toUpperCase(); value = clean(registry[2]); }
    else if (/^[0-9]+$/.test(company)) { mode = 'registry'; }
    else if (/[*?\\]/u.test(company)) refuse('wildcard_not_supported');
  }
  const url = new URL(mode === 'lei' ? `${GLEIF}/${value}` : GLEIF);
  if (mode !== 'lei') {
    url.searchParams.set(mode === 'registry' ? 'filter[entity.registeredAs]' : 'filter[entity.legalName]', value);
    // Country-wide lookup omits jurisdiction filtering to avoid treating a country as a subdivision.
    // The complete returned collection is checked below. Over-limit queries refuse.
    if (jur.includes('-')) url.searchParams.set('filter[entity.jurisdiction]', jur);
    url.searchParams.set('page[size]', String(LIMITS.records));
  }
  return { company, jurisdiction: jur, mode, value, authority, url: url.href };
}
function object(x) { return x !== null && typeof x === 'object' && !Array.isArray(x); }
function stamp(x) {
  if (typeof x !== 'string' || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z$/.test(x)) refuse('invalid_source_date', 502);
  const t = Date.parse(x); if (!Number.isFinite(t)) refuse('invalid_source_date', 502); return t;
}
function sourceString(x) {
  if (typeof x !== 'string' || !x.trim() || enc.encode(x).length > LIMITS.fieldBytes || /[\p{Cc}\p{Cf}]/u.test(x))
    refuse('invalid_source_field', 502);
  return x;
}
function matchesJur(actual, expected) {
  return actual === expected || (!expected.includes('-') && actual.startsWith(`${expected}-`));
}
function addressText(a) {
  if (!object(a) || !Array.isArray(a.addressLines) || a.addressLines.length > LIMITS.addressLines)
    refuse('invalid_source_address', 502);
  const parts = [a.mailRouting, a.addressNumber, a.addressNumberWithinBuilding, ...a.addressLines,
    a.city, a.region, a.country, a.postalCode].filter(x => x != null && x !== '');
  return parts.map(sourceString).join(', ') || null;
}
export function normalize(plan, doc, now = Date.now()) {
  if (!object(doc) || !object(doc.meta) || !object(doc.meta.goldenCopy)) refuse('invalid_source_shape', 502);
  const publication = doc.meta.goldenCopy.publishDate, published = stamp(publication);
  if (now - published > LIMITS.snapshotAgeMs || published - now > LIMITS.futureSkewMs) refuse('stale_source', 503);
  let records;
  if (plan.mode === 'lei') {
    if (!object(doc.data)) refuse('invalid_source_shape', 502); records = [doc.data];
  } else {
    const p = doc.meta.pagination;
    if (!Array.isArray(doc.data) || doc.data.length > LIMITS.records || !object(p) ||
        !Number.isSafeInteger(p.total) || p.total < 0 || p.currentPage !== 1 ||
        !Number.isSafeInteger(p.lastPage) || p.lastPage < 1) refuse('invalid_source_shape', 502);
    if (p.lastPage !== 1 || p.total !== doc.data.length || doc.links?.next) refuse('incomplete_collection', 422);
    records = doc.data;
  }
  const matches = [], seen = new Set();
  for (const rec of records) {
    if (!object(rec) || rec.type !== 'lei-records' || !validLEI(rec.id) ||
        !object(rec.attributes) || rec.attributes.lei !== rec.id ||
        !object(rec.attributes.entity)) refuse('invalid_source_record', 502);
    if (seen.has(rec.id)) refuse('duplicate_source_record', 502); seen.add(rec.id);
    const e = rec.attributes.entity;
    const name = sourceString(e.legalName?.name), jur = sourceString(e.jurisdiction);
    if (!/^[A-Z]{2}(?:-[A-Z0-9]{1,3})?$/.test(jur)) refuse('invalid_source_jurisdiction', 502);
    const exact = plan.mode === 'lei' ? rec.id === plan.value : plan.mode === 'registry' ?
      e.registeredAs === plan.value && (!plan.authority || e.registeredAt?.id === plan.authority) :
      clean(name, LIMITS.fieldBytes).toLowerCase() === plan.value.toLowerCase();
    if (exact && matchesJur(jur, plan.jurisdiction)) matches.push(rec);
  }
  if (!matches.length) refuse(records.length ? 'no_exact_identity_in_jurisdiction' : 'missing', 404);
  if (matches.length !== 1) refuse('ambiguous_identity', 409);
  const rec = matches[0], a = rec.attributes, e = a.entity, r = a.registration;
  // An invalid or superseded LEI does not prove the entity's state. Renewal never sets active.
  if (!object(r) || !['ISSUED', 'LAPSED', 'RETIRED', 'MERGED', 'PENDING_TRANSFER', 'PENDING_ARCHIVAL', 'TRANSFERRED'].includes(r.status))
    refuse('unusable_lei_record', 422);
  if (!['ACTIVE', 'INACTIVE'].includes(e.status)) refuse('unknown_entity_status', 422);
  const lastUpdate = sourceString(r.lastUpdateDate);
  if (stamp(lastUpdate) > published + LIMITS.futureSkewMs) refuse('inconsistent_source_date', 502);
  return { active: e.status === 'ACTIVE', legal_name: sourceString(e.legalName.name),
    registration_id: rec.id, jurisdiction: e.jurisdiction, registered_address: addressText(e.legalAddress),
    evidence: { lei: rec.id, authority: e.registeredAt?.id ?? null, registered_as: e.registeredAs ?? null,
      entity_status: e.status, lei_registration_status: r.status, last_update: lastUpdate,
      publication, fetched_at: new Date(now).toISOString(), source_url: plan.url,
      terms: 'GLEIF LEI reference data CC0-1.0; original normalization LicenseRef-zkasuran-SAND-1.0',
      scope: 'Entity status as reported in the Global LEI Index. No certification of national good standing or incorporation date.',
      attribution: 'Global Legal Entity Identifier Foundation (GLEIF). No endorsement.' } };
}
export async function boundedBytes(response, signal) {
  if (!response.ok) refuse(response.status === 404 ? 'missing' : 'upstream_failure', response.status === 404 ? 404 : 502);
  const type = response.headers.get('content-type') || '';
  if (!/^application\/(?:json|vnd\.api\+json)(?:;|$)/i.test(type)) refuse('upstream_content_type', 502);
  const length = response.headers.get('content-length');
  if (length && (!/^\d+$/.test(length) || Number(length) > LIMITS.upstreamBytes)) refuse('source_too_large', 502);
  if (!response.body) refuse('empty_source', 502);
  const reader = response.body.getReader(), chunks = []; let size = 0;
  const abort = () => { reader.cancel().catch(() => {}); };
  signal?.addEventListener('abort', abort, {once:true});
  try {
    if (signal?.aborted) refuse('upstream_timeout', 504);
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      size += value.length;
      if (size > LIMITS.upstreamBytes) refuse('source_too_large', 502);
      chunks.push(value);
    }
    if (signal?.aborted) refuse('upstream_timeout', 504);
    const bytes = new Uint8Array(size); let pos = 0;
    for (const c of chunks) { bytes.set(c, pos); pos += c.length; }
    return bytes;
  } finally {
    signal?.removeEventListener('abort', abort);
    await reader.cancel().catch(() => {}); reader.releaseLock();
  }
}
export async function boundedJSON(response, signal) {
  const bytes = await boundedBytes(response, signal);
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
  catch { refuse('invalid_source_json', 502); }
}
function reply(body, status) {
  const text = JSON.stringify(body);
  if (enc.encode(text).length > LIMITS.outputBytes) refuse('output_too_large', 502);
  return new Response(text, { status, headers: { 'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store', 'Content-Security-Policy': "default-src 'none'; frame-ancestors 'none'",
    'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer' } });
}
export async function handle(request, fetcher = fetch, now = Date.now(), timeoutMs = LIMITS.timeoutMs) {
  let timer, readerTask;
  const controller = new AbortController();
  try {
    if (request.method !== 'GET') refuse('method_not_allowed', 405);
    if (enc.encode(request.url).length > LIMITS.urlBytes) refuse('request_too_large', 414);
    const u = new URL(request.url);
    if (u.pathname !== '/company') refuse('path_not_found', 404);
    const keys = [...u.searchParams.keys()];
    if (keys.length !== 2 || new Set(keys).size !== 2 || !keys.every(k => ['company', 'jurisdiction'].includes(k))) refuse('invalid_parameters');
    const plan = requestPlan(u.searchParams.get('company'), u.searchParams.get('jurisdiction'));
    const work = async () => {
      const response = await fetcher(plan.url, { method: 'GET', redirect: 'error', signal: controller.signal,
        headers: { Accept: 'application/vnd.api+json', 'User-Agent': 'Telegraph-Corporate-Truth/1 (public verification; github.com/zkasuran)' },
        cache: 'no-store' });
      readerTask = response.body;
      return normalize(plan, await boundedJSON(response, controller.signal), now);
    };
    const timeout = new Promise((_, reject) => { timer = setTimeout(() => {
      controller.abort(); if (readerTask && !readerTask.locked) readerTask.cancel().catch(() => {});
      reject(new Refusal('upstream_timeout', 504));
    }, timeoutMs); });
    const body = await Promise.race([work(), timeout]);
    return reply(body, 200);
  } catch (e) {
    return reply({ error: { code: e instanceof Refusal ? e.code : 'upstream_failure',
      message: 'No usable reference. No Boolean answer is supplied.' } }, e instanceof Refusal ? e.status : 502);
  } finally { clearTimeout(timer); controller.abort(); }
}
export default { fetch: request => handle(request) };
