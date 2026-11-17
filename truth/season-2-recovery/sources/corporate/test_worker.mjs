// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const { handle, normalize, requestPlan, LIMITS, validLEI } = await import(process.env.CORPORATE_WORKER_MODULE || './worker.mjs');
const fixture = JSON.parse(await readFile(new URL('./evidence/public/valid-name.json', import.meta.url)));
const original = fixture.document;
const now = Date.parse(fixture.fetched_at);
const plan = requestPlan('Apple Inc.', 'US-CA');
const clone = () => structuredClone(original);
const upstream = body => async () => new Response(JSON.stringify(body), { headers: { 'content-type': 'application/vnd.api+json' } });
const req = (company = 'Apple Inc.', jurisdiction = 'US-CA') => new Request(`https://local-test.invalid/company?${new URLSearchParams({company, jurisdiction})}`);
async function denied(body, status, code, input = req()) {
  const result = await handle(input, upstream(body), now);
  const answer = await result.json();
  assert.equal(result.status, status); assert.equal(answer.error.code, code); assert.equal(Object.hasOwn(answer, 'active'), false);
}
test('fresh exact name maps to a Boolean and the same legal identity', () => {
  const answer = normalize(plan, clone(), now);
  assert.equal(answer.active, true); assert.equal(answer.registration_id, 'HWUPKR0MPOU8FGXBT394');
  assert.equal(answer.jurisdiction, 'US-CA'); assert.equal(Object.hasOwn(answer, 'incorporated'), false);
});
test('real inactive source is false and no renewal logic is used', async () => {
  const f = JSON.parse(await readFile(new URL('./evidence/public/inactive.json', import.meta.url)));
  assert.equal(normalize(requestPlan(f.request.company, f.request.jurisdiction), f.document, Date.parse(f.fetched_at)).active, false);
});
test('real LAPSED LEI with ACTIVE entity is true', async () => {
  const f = JSON.parse(await readFile(new URL('./evidence/public/lapsed-active.json', import.meta.url)));
  assert.equal(f.document.data.attributes.registration.status, 'LAPSED');
  assert.equal(normalize(requestPlan(f.request.company, f.request.jurisdiction), f.document, Date.parse(f.fetched_at)).active, true);
});
test('NFC and whitespace normalization preserves punctuation', async () => {
  const answer = normalize(requestPlan(' Apple  Inc. ', 'california'), clone(), now);
  assert.equal(answer.active, true);
  await denied(clone(), 404, 'no_exact_identity_in_jurisdiction', req('Apple Inc', 'US-CA'));
});
test('country accepts subdivision but a different subdivision fails', async () => {
  assert.equal(normalize(requestPlan('Apple Inc.', 'US'), clone(), now).active, true);
  await denied(clone(), 404, 'no_exact_identity_in_jurisdiction', req('Apple Inc.', 'US-DE'));
});
test('misrouted LEI record never answers', async () => {
  const doc = clone(); doc.data = doc.data[0];
  await denied(doc, 404, 'no_exact_identity_in_jurisdiction', req('815600AE2D204A011F75'));
});
test('bare numeric identifier and authority-qualified identifier bind exact registeredAs', () => {
  assert.equal(normalize(requestPlan('806592', 'US-CA'), clone(), now).active, true);
  assert.throws(() => normalize(requestPlan('RA000407:806592', 'US-CA'), clone(), now), /no_exact_identity/);
});
test('two exact identities are ambiguous even when statuses agree', async () => {
  const doc = clone(), another = structuredClone(doc.data[0]);
  another.id = '815600AE2D204A011F75'; another.attributes.lei = another.id;
  doc.data.push(another); doc.meta.pagination.total = 2;
  await denied(doc, 409, 'ambiguous_identity');
});
test('fresh public ambiguous mixed-status source refuses', async () => {
  const f = JSON.parse(await readFile(new URL('./evidence/public/ambiguous.json', import.meta.url)));
  const exact = f.document.data.filter(x => x.attributes.entity.legalName.name === 'VALE S.R.L.');
  assert.ok(new Set(exact.map(x => x.attributes.entity.status)).size > 1);
  assert.throws(() => normalize(requestPlan('VALE S.R.L.', 'IT'), f.document, Date.parse(f.fetched_at)), /ambiguous_identity/);
});
test('source failure and missing result are never false', async () => {
  const doc = clone(); doc.data = []; doc.meta.pagination.total = 0;
  await denied(doc, 404, 'missing');
  const r = await handle(req(), async () => new Response('{}', {status: 503}), now);
  assert.equal(r.status, 502); assert.equal(Object.hasOwn(await r.json(), 'active'), false);
});
test('stale publication refuses', async () => {
  const doc = clone(); doc.meta.goldenCopy.publishDate = new Date(now - LIMITS.snapshotAgeMs - 1).toISOString();
  await denied(doc, 503, 'stale_source');
});
test('future publication refuses', async () => {
  const doc = clone(); doc.meta.goldenCopy.publishDate = new Date(now + LIMITS.futureSkewMs + 1).toISOString();
  await denied(doc, 503, 'stale_source');
});
test('missing publication and missing pagination refuse', async () => {
  const doc = clone(); delete doc.meta.goldenCopy; await denied(doc, 502, 'invalid_source_shape');
  const d = clone(); delete d.meta.pagination; await denied(d, 502, 'invalid_source_shape');
});
test('partial page and next page link refuse', async () => {
  const d = clone(); d.meta.pagination.total = 101; d.meta.pagination.lastPage = 2;
  await denied(d, 422, 'incomplete_collection');
  const e = clone(); e.links.next = 'https://api.gleif.org/api/v1/lei-records?page=2';
  await denied(e, 422, 'incomplete_collection');
});
test('malformed collection total is not absence', async () => {
  const d = clone(); d.meta.pagination.total = '1'; await denied(d, 502, 'invalid_source_shape');
});
test('duplicate or mismatched LEIs refuse', async () => {
  const d = clone(); d.data.push(structuredClone(d.data[0])); d.meta.pagination.total = 2;
  await denied(d, 502, 'duplicate_source_record');
  const e = clone(); e.data[0].attributes.lei = '815600AE2D204A011F75'; await denied(e, 502, 'invalid_source_record');
});
test('unknown entity status and invalidated LEI record refuse', async () => {
  const d = clone(); d.data[0].attributes.entity.status = null; await denied(d, 422, 'unknown_entity_status');
  for (const status of ['ANNULLED', 'CANCELLED', 'DUPLICATE', 'PENDING_VALIDATION']) {
    const e = clone(); e.data[0].attributes.registration.status = status; await denied(e, 422, 'unusable_lei_record');
  }
});
test('address captures all routing fields and no language label', () => {
  const d = clone(); Object.assign(d.data[0].attributes.entity.legalAddress, {mailRouting:'Care of agent',addressNumber:'4',addressNumberWithinBuilding:'2A'});
  const out = normalize(plan, d, now).registered_address;
  assert.ok(out.startsWith('Care of agent, 4, 2A,')); assert.ok(!out.includes('language'));
});
test('schema and resource boundaries refuse oversized source fields', async () => {
  const d = clone(); d.data[0].attributes.entity.legalName.name = 'X'.repeat(LIMITS.fieldBytes + 1);
  await denied(d, 502, 'invalid_source_field');
  const e = clone(); e.data[0].attributes.entity.legalAddress.addressLines = Array(9).fill('X');
  await denied(e, 502, 'invalid_source_address');
});
test('bad content type, JSON and invalid UTF-8 refuse', async () => {
  for (const [body, type] of [['{}', 'text/html'], ['{', 'application/json'], [new Uint8Array([255]), 'application/json']]) {
    const r = await handle(req(), async () => new Response(body,{headers:{'content-type':type}}), now);
    assert.equal(r.status, 502); assert.equal(Object.hasOwn(await r.json(),'active'),false);
  }
});
test('stream size is bounded even without content length', async () => {
  const stream = new ReadableStream({start(c){c.enqueue(new Uint8Array(LIMITS.upstreamBytes + 1));c.close();}});
  const r = await handle(req(), async () => new Response(stream,{headers:{'content-type':'application/json'}}), now);
  assert.equal(r.status,502); assert.equal((await r.json()).error.code,'source_too_large');
});
test('declared source length is bounded', async () => {
  const r = await handle(req(), async () => new Response('{}',{headers:{'content-type':'application/json','content-length':String(LIMITS.upstreamBytes+1)}}),now);
  assert.equal(r.status,502); assert.equal((await r.json()).error.code,'source_too_large');
});
test('timeout covers both headers and body', async () => {
  const hung = async () => new Promise(() => {});
  const r = await handle(req(),hung,now,10); assert.equal(r.status,504);
  const stream = new ReadableStream({start(){}});
  const s = await handle(req(),async () => new Response(stream,{headers:{'content-type':'application/json'}}),now,10);
  assert.equal(s.status,504);
});
test('redirect mode is error and no user-selected host is used', async () => {
  let called;
  await handle(req(),async (url, options) => {called={url,options};return upstream(clone())();},now);
  assert.equal(new URL(called.url).hostname,'api.gleif.org'); assert.equal(called.options.redirect,'error');
  assert.equal(called.options.method,'GET');
});
test('missing, duplicate and foreign query parameters refuse before upstream', async () => {
  for (const suffix of ['', '?company=A', '?company=A&company=B&jurisdiction=IT', '?company=A&jurisdiction=IT&url=http://localhost']) {
    const r = await handle(new Request(`https://local-test.invalid/company${suffix}`), async()=>{assert.fail('no upstream');}, now);
    assert.equal(r.status,422);
  }
});
test('method and path gates', async () => {
  assert.equal((await handle(new Request('https://local-test.invalid/company',{method:'POST'}))).status,405);
  assert.equal((await handle(new Request('https://local-test.invalid/other'))).status,404);
});
test('control, wildcard, unknown alias and multibyte oversized input refuse', () => {
  for (const company of ['A\u0000B', 'A\u200bB', '*', 'A?', 'é'.repeat(129)]) assert.throws(()=>requestPlan(company,'IT'));
  assert.throws(()=>requestPlan('Apple Inc.','somewhere'));
});
test('LEI checksum and explicit malformed LEI refuse', () => {
  assert.equal(validLEI('HWUPKR0MPOU8FGXBT394'),true);
  assert.equal(validLEI('HWUPKR0MPOU8FGXBT395'),false);
  assert.throws(()=>requestPlan('LEI:wrong','US'));
});
test('seeded hostile byte strings never produce an uncontrolled exception or upstream call', async () => {
  let seed=7917;
  for(let n=0;n<128;n++) {
    seed=(Math.imul(seed,1664525)+1013904223)>>>0;
    const c=String.fromCharCode(seed%32);
    const r=await handle(req(`A${c}B`),async()=>assert.fail('invalid input reached upstream'),now);
    assert.equal(r.status,422);
  }
});
test('same observation and time produces identical bytes', () => {
  assert.equal(JSON.stringify(normalize(plan,clone(),now)),JSON.stringify(normalize(plan,clone(),now)));
});
