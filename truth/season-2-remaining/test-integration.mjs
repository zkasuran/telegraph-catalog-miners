// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createIntegration, NEW_PATHS } from './worker.mjs';
import { handleFinance, apyBps, abiWords, reserveFromRound, LIMITS as FIN } from './sources/finance/finance.mjs';
import { handle, validatorRecord, metarObservation, LIMITS as OPS } from './sources/operations/operations.mjs';
const read = path => readFileSync(new URL(path, import.meta.url), 'utf8');
const load = path => JSON.parse(read(path));
const req = path => new Request('https://unchanged.example' + path);
const observed = Date.parse(load('sources/operations/168/OBSERVATIONS.json').observed_at) / 1000;
const fixtureFetch = async url => {
  const u = new URL(url);
  if (u.hostname === 'api.met.no') return new Response(read(`sources/operations/fixtures/168/metar-${u.searchParams.get('icao')}.body`));
  assert.equal(u.hostname, 'lab.ethpandaops.io');
  const index = u.searchParams.get('validator_index_eq');
  return new Response(read(`sources/operations/fixtures/108/${u.pathname.endsWith('dim_validator') ? 'identity' : 'xatu'}-${index}.body`));
};

test('Only the four exact routes are mounted', async () => {
  assert.deepEqual(NEW_PATHS, ['/truth/v2/finance-reserve', '/truth/v2/finance-yield',
    '/truth/v2/operations/validator', '/truth/v2/operations/weather']);
  for (const [i, path] of NEW_PATHS.entries()) {
    const request = req(path + '?sentinel=untouched');
    const selected = [];
    const app = createIntegration({fetch() { throw new Error('wrong old route'); }},
      r => { assert.equal(r, request); selected.push('finance'); return new Response('finance'); },
      r => { assert.equal(r, request); selected.push('operations'); return new Response('operations'); });
    await app.fetch(request, {}, {});
    assert.deepEqual(selected, [i < 2 ? 'finance' : 'operations']);
  }
});

test('Old request identity, receiver, body, env and context stay identical', async () => {
  for (const path of ['/health', '/hashprice', '/truth/v1/company', '/truth/v1/balance', '/truth/v1/hashprice',
    '/truth/v2/operations/delivery', '/truth/v2/operations/shipment', '/truth/v2/finance-payment',
    '/truth/v2/finance-dam', ...NEW_PATHS.flatMap(p => [p + '/', p + '-extra', '/prefix' + p])]) {
    const request = new Request('https://old.example' + path + '?q=a%20b', {method: 'POST', body: 'exact body', headers: {'x-marker':'old'}});
    const env = {marker: 1}, context = {waitUntil() {}}, response = new Response('original');
    const original = { async fetch(r, e, c) {
      assert.equal(this, original); assert.equal(r, request); assert.equal(e, env); assert.equal(c, context);
      assert.equal(await r.text(), 'exact body'); assert.equal(r.headers.get('x-marker'), 'old'); return response;
    }};
    const app = createIntegration(original, () => { throw new Error('finance must not run'); }, () => { throw new Error('operations must not run'); });
    assert.equal(await app.fetch(request, env, context), response);
  }
});

test('Lookalike and randomized paths cannot fall through to broad operations suffix handlers', async () => {
  let hits = 0;
  const app = createIntegration({fetch() { hits++; return new Response('old'); }},
    () => { throw new Error('unexpected finance'); }, () => { throw new Error('unexpected operations'); });
  for (let i = 0; i < 1000; i++) await app.fetch(req(`/other-${i}/weather`), {}, {});
  assert.equal(hits, 1000);
});

test('400 independently calculated APY vectors match the unchanged module', () => {
  const vectors = load('sources/finance/arithmetic-vectors.json'); assert.equal(vectors.length, 400);
  for (const v of vectors) assert.equal(apyBps(BigInt(v.rate), BigInt(v.denominator)), v.expected);
  assert.equal(apyBps(0n, 10n ** 18n), '0.000000');
  assert.throws(() => apyBps(-1n, 10n ** 18n));
});

test('Genuine reserve observation reproduces with half-up micro-BTC conversion and stale controls', () => {
  const r = load('sources/finance/reference.json').reserve;
  const words = [r.round_id, r.answer_satoshis, r.updated_at, r.updated_at, r.answered_in_round].map(BigInt);
  assert.equal(reserveFromRound(words, BigInt(r.decimals), BigInt(r.ethereum_block.timestamp)), '116144645100');
  assert.throws(() => reserveFromRound(words, 18n, BigInt(r.ethereum_block.timestamp)));
  assert.throws(() => reserveFromRound(words, 8n, BigInt(r.updated_at) + BigInt(FIN.oracleAge + 1)));
  assert.throws(() => reserveFromRound([1n, 0n, 1n, 1n, 1n], 8n, 1n));
  for (let i = 0; i < 1000; i++) assert.throws(() => abiWords('0x' + 'z'.repeat(i)));
});

test('All source validator indices and MET locations replay through the exact wrapper', async () => {
  const app = createIntegration({fetch() { throw new Error('unmatched'); }}, handleFinance,
    request => handle(request, fixtureFetch, observed));
  for (const index of [1, 100000, 500000]) {
    const response = await app.fetch(req(`/truth/v2/operations/validator?validator_index=${index}`));
    assert.equal(response.status, 200);
    const body = await response.json(); assert.equal(body.validator_index, index);
    assert.equal(body.effective_balance, index === 1 ? 32000000000 : 0);
    assert.equal(body.source_kind, 'delayed finalized consensus data');
  }
  for (const [location, station, temperature] of [['London','EGLC',11], ['Karachi','OPKC',32], ['Tokyo','RJTT',18]]) {
    const response = await app.fetch(req(`/truth/v2/operations/weather?location=${location}`));
    assert.equal(response.status, 200); const body = await response.json();
    assert.equal(body.station, station); assert.equal(body.temperature_c, temperature);
    assert.equal(body.temp_c, temperature); assert.equal(body.observation_type, 'station observation');
  }
});

test('Explicit delayed epoch refuses stale, future, wrong-index and wrong-epoch source records', () => {
  const data = load('sources/operations/fixtures/108/xatu-1.body');
  const id = load('sources/operations/fixtures/108/identity-1.body');
  const base = validatorRecord(data, id, 1, observed, 1606824023);
  assert.equal(Date.parse(base.epoch_start_time) / 1000, 1606824023 + base.epoch * 384);
  assert.throws(() => validatorRecord(data, id, 1, observed + 7200, 1606824023));
  assert.throws(() => validatorRecord(data, id, 1, 0, 1606824023));
  for (const [field, value] of [['validator_index', 2], ['epoch', 1], ['effective_balance', null]]) {
    const bad = structuredClone(data); bad.fct_validator_balance[0][field] = value;
    assert.throws(() => validatorRecord(bad, id, 1, observed, 1606824023));
  }
  assert.throws(() => validatorRecord({}, id, 1, observed, 1606824023));
});

test('Observed-airport selection refuses forecast, wrong station, entities and incorrect report time', () => {
  const xml = read('sources/operations/fixtures/168/metar-EGLC.body');
  for (const bad of ['<!DOCTYPE x>' + xml, xml.replaceAll('meteorologicalAerodromeReport','terminalAerodromeForecast'),
    xml.replaceAll('180920Z','180921Z'), xml.replaceAll('EGLC ', 'OPKC ')]) {
    assert.throws(() => metarObservation(bad, 'London', observed));
  }
  assert.throws(() => metarObservation(xml, 'Tokyo', observed));
  assert.throws(() => metarObservation(xml, 'London', observed + 86400));
});

test('Invalid new-route inputs refuse before any network access through the wrapper', async () => {
  let calls = 0; const never = () => { calls++; throw new Error('must not fetch'); };
  const app = createIntegration({fetch() { throw new Error('wrong route'); }},
    r => handleFinance(r, never), r => handle(r, never, observed));
  for (const path of ['/truth/v2/finance-reserve?asset=USDC', '/truth/v2/finance-reserve?asset=WBTC&asset=WBTC',
    '/truth/v2/finance-yield?project=morpho&symbol=USDC&chain=Ethereum',
    '/truth/v2/finance-yield?project=aave-v3&symbol=USDC',
    '/truth/v2/operations/validator?validator_index=-1', '/truth/v2/operations/validator?validator_index=1&validator_index=2',
    '/truth/v2/operations/weather?location=Unknown', '/truth/v2/operations/weather?location=London&location=Tokyo']) {
    const response = await app.fetch(req(path)); assert.notEqual(response.status, 200);
    const body = await response.json();
    for (const field of ['effective_balance','temperature_c','attested_reserve','supply_apy']) assert.ok(!(field in body));
  }
  assert.equal(calls, 0);
});

test('Redirects, upstream errors, non-JSON and oversized responses never become scored values', async () => {
  for (const make of [() => new Response(null,{status:302}), () => new Response('bad',{status:503}),
    () => new Response('<html>',{headers:{'content-type':'text/html'}}),
    () => new Response('x'.repeat(FIN.responseBytes + 1),{headers:{'content-type':'application/json'}})]) {
    const app = createIntegration(undefined, r => handleFinance(r, async () => make()),
      r => handle(r, async () => make(), observed));
    for (const path of ['/truth/v2/finance-reserve?asset=WBTC','/truth/v2/operations/weather?location=London']) {
      const response = await app.fetch(req(path)); assert.ok(response.status >= 400);
      const body = await response.json(); assert.ok(!('attested_reserve' in body) && !('temperature_c' in body));
    }
  }
});

test('Hanging new-source network calls abort under both unchanged source timeouts', async () => {
  const promises = [
    ['finance', r => handleFinance(r, hanging), '/truth/v2/finance-reserve?asset=WBTC', FIN.timeoutMs],
    ['operations', r => handle(r, hanging, observed), '/truth/v2/operations/weather?location=London', OPS.timeoutMs],
  ];
  function hanging(_url, init) { return new Promise((_resolve, reject) => init.signal.addEventListener('abort',
    () => reject(new Error('bounded abort')), {once: true})); }
  await Promise.all(promises.map(async ([name, run, path, limit]) => {
    const start = performance.now(); const response = await run(req(path));
    assert.ok(response.status >= 400); const elapsed = performance.now() - start;
    assert.ok(elapsed >= limit - 100 && elapsed < limit + 3000, name);
  }));
});
