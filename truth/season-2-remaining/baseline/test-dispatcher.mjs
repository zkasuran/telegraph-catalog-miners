// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import test from 'node:test';
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { corporate, inactive, ambiguous, blocks, block, bitcoinRPC, accountRPC, originalTransport, originalFixture } from './test-fixtures.mjs';

let transport = () => assert.fail('unexpected network call');
const nativeFetch = globalThis.fetch;
globalThis.fetch = (...args) => transport(...args);
// Import the actual deploy entry after installing the test transport, because the
// hashprice module captures fetch at initialization. No source module is changed.
const { default: dispatcher } = await import('./worker.mjs');
const { createDispatcher, forwardPath } = await import('./dispatcher.mjs');
const { default: original } = await import('./sources/original/worker.mjs');
const coinbase = await import('./sources/original/coinbase.mjs');
const { workersFetch } = await import('./workers-fetch.mjs');
const env = Object.freeze({ sentinel: 'exact-env' });
const ctx = { promises: [], waitUntil(p) { this.promises.push(p); } };
const origin = 'https://chosen-host.example:8443';
const req = (path, opts) => new Request(origin + path, opts);
const observations = [];
async function call(name, path, opts) {
  const response = await dispatcher.fetch(req(path, opts), env, ctx);
  const body = await response.text();
  observations.push({ name, path, status: response.status, headers: Object.fromEntries(response.headers), body });
  return { response, body, value: JSON.parse(body) };
}
function protectedHeaders(response) {
  assert.equal(response.headers.get('cache-control'), 'no-store');
  assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
  assert.ok(response.headers.get('content-security-policy').includes("default-src 'none'"));
}

test('routing seam preserves original Request identity, env, ctx, receiver and Response identity', async () => {
  const reply = new Response('sentinel', { status: 207, headers: { 'x-original': 'exact' } });
  const handlers = {};
  let input;
  for (const name of ['original', 'company', 'balance', 'hashprice']) {
    handlers[name] = { fetch(request, e, c) { assert.equal(this, handlers[name]);
      assert.equal(e, env); assert.equal(c, ctx); input = request; return reply; } };
  }
  const d = createDispatcher(handlers);
  for (const path of ['/health', '/__last', '/hashprice?block_height=840000', '/hashprice/%7Bblock_height%7D',
    '/unknown', '/truth/v1/company/', '/truth/v1/balance/42', '/truth/v1/hashprice/%7Bblock_height%7D',
    '/company', '/metric', '/truth/v1/health', '/TRUTH/v1/company']) {
    const request = req(path);
    assert.equal(await d.fetch(request, env, ctx), reply);
    assert.equal(input, request);
  }
  for (const [path, target] of [['company', '/company'], ['balance', '/metric'], ['hashprice', '/hashprice']]) {
    const request = req('/truth/v1/' + path + '?x=a%2Fb&x=c+d&empty=', { method: 'POST',
      body: new Uint8Array([0, 255, 13, 10, 32, 123, 125]),
      headers: { 'content-type': 'application/octet-stream', 'x-user': 'unchanged' }, redirect: 'manual' });
    const before = await request.clone().arrayBuffer();
    assert.equal(await d.fetch(request, env, ctx), reply);
    const u = new URL(input.url);
    assert.equal(u.origin, origin); assert.equal(u.pathname, target);
    assert.equal(u.search, '?x=a%2Fb&x=c+d&empty='); assert.equal(input.method, 'POST');
    assert.deepEqual(Object.fromEntries(input.headers), Object.fromEntries(request.headers));
    assert.equal(input.redirect, 'manual');
    assert.deepEqual(new Uint8Array(await input.arrayBuffer()), new Uint8Array(before));
  }
});

test('path forwarding retains stream bytes and caller cancellation', async () => {
  const abort = new AbortController();
  const request = req('/truth/v1/company?q=%26', { method: 'PUT', signal: abort.signal,
    body: new ReadableStream({ start(c) { c.enqueue(new Uint8Array([7, 0, 255])); c.close(); } }), duplex: 'half' });
  const forwarded = forwardPath(request, '/company');
  abort.abort(); assert.equal(forwarded.signal.aborted, true);
  assert.deepEqual(new Uint8Array(await forwarded.arrayBuffer()), new Uint8Array([7, 0, 255]));
});

test('scoped Workers transport preserves URL, method, headers, body and signal; redirects fail closed', async () => {
  const signal = new AbortController().signal;
  const init = { method: 'POST', body: 'exact body', headers: { 'x-user': 'exact' }, signal, redirect: 'error', cache: 'no-store' };
  const url = 'https://fixed-upstream.example/path?q=exact';
  transport = async (input, options) => {
    assert.equal(input, url); assert.deepEqual(options, { ...init, redirect: 'manual' });
    return Response.json({ ok: true });
  };
  assert.equal((await workersFetch(url, init)).status, 200);
  for (const status of [301, 302, 303, 307, 308]) {
    transport = async () => new Response('', { status, headers: { location: 'https://other.example/' } });
    await assert.rejects(workersFetch(url, init), /Redirect refused/);
  }
  const manual = { method: 'GET', redirect: 'manual' };
  transport = async (input, options) => { assert.equal(input, url); assert.equal(options, manual); return new Response('ok'); };
  assert.equal(await (await workersFetch(url, manual)).text(), 'ok');
});

test('company and balance through actual entry reject upstream redirects with no scored body', async () => {
  transport = async () => new Response('', { status: 302, headers: { location: 'https://untrusted.example/' } });
  const company = await call('company-redirect-refusal', '/truth/v1/company?company=Apple%20Inc.&jurisdiction=US-CA');
  assert.equal(company.response.status, 502); assert.ok(!Object.hasOwn(company.value, 'active'));
  const balance = await call('balance-redirect-refusal', '/truth/v1/balance?address=0x0000000000000000000000000000000000000001&block=42');
  assert.equal(balance.response.status, 502); assert.ok(!Object.hasOwn(balance.value, 'balance_gwei'));
});

test('actual deploy entry company success, inactive false, ambiguous and failed source body contracts', async () => {
  for (const [name, fixture, want] of [['active', corporate, true], ['inactive', inactive, false]]) {
    transport = async (url, options) => {
      assert.equal(new URL(url).hostname, 'api.gleif.org'); assert.equal(options.method, 'GET');
      assert.equal(options.redirect, 'manual');
      return Response.json(fixture.document);
    };
    const q = new URLSearchParams({ company: fixture.request.company, jurisdiction: fixture.request.jurisdiction });
    const x = await call('company-' + name, '/truth/v1/company?' + q);
    assert.equal(x.response.status, 200, x.body); assert.equal(x.value.active, want);
    assert.equal(typeof x.value.active, 'boolean'); protectedHeaders(x.response);
  }
  transport = async () => Response.json(ambiguous.document);
  const multi = await call('company-ambiguous', '/truth/v1/company?company=VALE%20S.R.L.&jurisdiction=IT');
  assert.equal(multi.response.status, 409); assert.equal(multi.value.error.code, 'ambiguous_identity');
  assert.ok(!Object.hasOwn(multi.value, 'active'));
  transport = async () => new Response('unavailable', { status: 503 });
  const denied = await call('company-source-refusal', '/truth/v1/company?company=Apple%20Inc.&jurisdiction=US-CA');
  assert.equal(denied.response.status, 502); assert.ok(!Object.hasOwn(denied.value, 'active')); protectedHeaders(denied.response);
});

test('balance on all three chains preserves numeric wire precision beyond JS safe integer', async () => {
  for (const chain of ['ethereum', 'polygon', 'arbitrum']) {
    const calls = []; transport = accountRPC({ calls });
    const x = await call('balance-' + chain, '/truth/v1/balance?address=0x0000000000000000000000000000000000000001&block=42&chain=' + chain);
    assert.equal(x.response.status, 200, x.body); protectedHeaders(x.response);
    assert.match(x.body, /"balance_gwei":9007199254\.740993[,}]/);
    assert.equal(x.value.balance_gwei_exact, '9007199254.740993001');
    assert.equal(x.value.balance_wei, '9007199254740993001');
    assert.equal(x.value.block, '42'); assert.equal(x.value.chain, chain);
    assert.equal(new Set(calls.map(x => x.url)).size, 2); assert.equal(calls.length, 6);
  }
  transport = accountRPC({ wei: 100000000000000000000000000000n });
  const maximum = await call('balance-maximum', '/truth/v1/balance?address=0x0000000000000000000000000000000000000001&block=42');
  assert.match(maximum.body, /"balance_gwei":100000000000000000000\.000000[,}]/);
  transport = accountRPC({ fail: true });
  const denied = await call('balance-source-refusal', '/truth/v1/balance?address=0x0000000000000000000000000000000000000001&block=42');
  assert.equal(denied.response.status, 502); assert.ok(!Object.hasOwn(denied.value, 'balance_gwei')); protectedHeaders(denied.response);
});

test('hashprice through actual entry verifies known height and exact six-decimal number lexeme', async () => {
  transport = bitcoinRPC();
  const x = await call('hashprice-success', '/truth/v1/hashprice?block_height=840000');
  assert.equal(x.response.status, 200, x.body); protectedHeaders(x.response);
  assert.match(x.body, /"hashprice_sats_per_ph_day":948924\.848225[,}]/);
  assert.equal(x.value.block_height, 840000); assert.equal(x.value.block_hash, block.canonical_hash);
  assert.equal(x.value.verification.coinbase_merkle_inclusion, true);
  transport = bitcoinRPC(block, (call, value) => call.method === 'getblockstats' ? { ...value, totalfee: value.totalfee + 1 } : value);
  const denied = await call('hashprice-source-refusal', '/truth/v1/hashprice?block_height=840000');
  assert.equal(denied.response.status, 503);
  assert.equal(denied.value.error.code, 'underclaimed_or_inconsistent_reward');
  assert.ok(!Object.hasOwn(denied.value, 'hashprice_sats_per_ph_day')); protectedHeaders(denied.response);
});

test('hashprice RPC diagnostics through the actual dispatcher do not leak remote text or retry missing data', async () => {
  const ordinary = bitcoinRPC();
  const calls = [];
  transport = async (url, options) => {
    const request = JSON.parse(options.body); calls.push(request);
    if (request.method === 'getrawtransaction') return Response.json({ jsonrpc: '2.0', id: 1,
      error: { code: -5, message: 'REMOTE_CONTENT_MUST_NOT_LEAK', data: { detail: 'REMOTE_DATA_MUST_NOT_LEAK' } } });
    return ordinary(url, options);
  };
  const x = await call('hashprice-diagnostic-permanent', '/truth/v1/hashprice?block_height=840000');
  assert.equal(x.response.status, 503);
  assert.equal(x.value.error.method, 'getrawtransaction');
  assert.equal(x.value.error.rpc_error_code, -5);
  assert.equal(x.value.error.rpc_error_code_type, 'number');
  assert.equal(x.value.error.http_status, 200);
  assert.equal(x.value.error.attempts, 1);
  assert.equal(x.value.error.retry_exhausted, false);
  assert.deepEqual(Object.keys(x.value.error).sort(), ['code', 'message', 'method', 'rpc_error_code',
    'rpc_error_code_type', 'http_status', 'attempts', 'retry_exhausted', 'envelope_jsonrpc', 'envelope_id',
    'envelope_id_type', 'envelope_id_matches', 'result_present', 'error_present', 'top_level_type'].sort());
  assert.equal(x.value.error.envelope_jsonrpc, '2.0'); assert.equal(x.value.error.envelope_id, 1);
  assert.equal(x.value.error.envelope_id_matches, true); assert.equal(x.value.error.result_present, false);
  assert.equal(calls.filter(c => c.method === 'getrawtransaction').length, 1);
  assert.equal(x.value.upstream.retries, 0);
  assert.ok(!x.body.includes('REMOTE_'));
  assert.ok(!Object.hasOwn(x.value, 'hashprice_sats_per_ph_day'));
});

test('numeric warmup retries exact payload through the unchanged wrapper and preserves proof', async () => {
  const ordinary = bitcoinRPC();
  const selected = [];
  transport = async (url, options) => {
    const request = JSON.parse(options.body);
    if (request.method === 'getrawtransaction') {
      selected.push(options.body);
      if (selected.length === 1) return Response.json({ jsonrpc: '2.0', id: 1, error: { code: -28, message: 'REMOTE_WARMUP_MESSAGE' } });
    }
    return ordinary(url, options);
  };
  const x = await call('hashprice-warmup-recovered', '/truth/v1/hashprice?block_height=840000');
  assert.equal(x.response.status, 200, x.body);
  assert.equal(selected.length, 2); assert.equal(selected[0], selected[1]);
  assert.equal(x.value.hashprice_sats_per_ph_day, 948924.848225);
  assert.equal(x.value.verification.coinbase_merkle_inclusion, true);
  assert.equal(x.value.verification.canonical_height_before_and_after, true);
  assert.equal(x.value.upstream.retries, 1); assert.equal(x.value.upstream.subrequests, 9);
  assert.ok(!x.body.includes('REMOTE_')); protectedHeaders(x.response);
});

test('string warmup code is diagnosed but is never retried through the dispatcher', async () => {
  let count = 0;
  transport = async () => {
    count++;
    return Response.json({ jsonrpc: '2.0', id: 1, error: { code: '-28', message: 'REMOTE_STRING_CODE' } });
  };
  const x = await call('hashprice-string-code-refusal', '/truth/v1/hashprice?block_height=840000');
  assert.equal(x.response.status, 503); assert.equal(count, 1);
  assert.equal(x.value.error.method, 'getblockhash'); assert.equal(x.value.error.rpc_error_code, null);
  assert.equal(x.value.error.rpc_error_code_type, 'string'); assert.equal(x.value.error.attempts, 1);
  assert.equal(x.value.upstream.retries, 0); assert.ok(!x.body.includes('REMOTE_'));
});

test('documented markerless legacy replies pass the existing dispatcher without weakening proof', async () => {
  const ordinary = bitcoinRPC();
  transport = async (...args) => {
    const res = await ordinary(...args);
    const body = await res.json();
    return Response.json({ id: 1, result: body.result, error: null });
  };
  const x = await call('hashprice-markerless-success', '/truth/v1/hashprice?block_height=840000');
  assert.equal(x.response.status, 200, x.body);
  assert.equal(x.value.hashprice_sats_per_ph_day, 948924.848225);
  assert.equal(x.value.verification.coinbase_merkle_inclusion, true);
  assert.equal(x.value.upstream.subrequests, 8); assert.equal(x.value.upstream.retries, 0);
});

test('foreign legacy IDs are rejected and redacted through the dispatcher', async () => {
  transport = async () => Response.json({ id: 'FOREIGN_ID_MUST_NOT_LEAK', result: block.canonical_hash, error: null });
  const x = await call('hashprice-foreign-id-refusal', '/truth/v1/hashprice?block_height=840000');
  assert.equal(x.response.status, 503); assert.equal(x.value.error.method, 'getblockhash');
  assert.equal(x.value.error.envelope_id, null); assert.equal(x.value.error.envelope_id_matches, false);
  assert.equal(x.value.error.envelope_id_type, 'string'); assert.equal(x.value.error.http_status, 200);
  assert.equal(x.value.upstream.subrequests, 1); assert.equal(x.value.upstream.retries, 0);
  assert.ok(!x.body.includes('FOREIGN_ID')); assert.ok(!Object.hasOwn(x.value, 'hashprice_sats_per_ph_day'));
});

test('all truth paths refuse malformed selectors, foreign upstream hints and POST body without I/O', async () => {
  transport = () => assert.fail('invalid request reached upstream');
  for (const [path, scored] of [['company', 'active'], ['balance', 'balance_gwei'], ['hashprice', 'hashprice_sats_per_ph_day']]) {
    for (const suffix of ['', '?url=https://evil.example', '?block_height=bad']) {
      const x = await call(path + '-invalid', '/truth/v1/' + path + suffix);
      assert.ok(x.response.status >= 400); assert.ok(!Object.hasOwn(x.value, scored));
    }
    const post = await call(path + '-post', '/truth/v1/' + path, { method: 'POST', body: ' { "exact": 1 }\n' });
    assert.equal(post.response.status, 405); assert.ok(!Object.hasOwn(post.value, scored));
  }
});

test('every original route and original method contract survives separately from truth routing', async () => {
  transport = () => assert.fail('local original routes should not call upstream');
  const cases = [
    ['health', '/health', {}, 200], ['health-post', '/health', { method: 'POST', body: 'x' }, 200],
    ['recent', '/__last', {}, 200], ['invalid', '/hashprice?block_height=invalid', {}, 200],
    ['duplicate', '/hashprice?block_height=42&block_height=42', {}, 200],
    ['post', '/hashprice?block_height=840000', { method: 'POST', body: 'x' }, 200],
    ['options', '/hashprice', { method: 'OPTIONS' }, 200],
    ['literal-encoded', '/hashprice/%7Bblock_height%7D', {}, 200],
    ['literal-braces', '/hashprice/{block_height}', {}, 200],
    ['literal-options', '/hashprice/%7bblock_height%7d', { method: 'OPTIONS' }, 200],
    ['404', '/unknown', {}, 404], ['404-options', '/unknown', { method: 'OPTIONS' }, 404],
  ];
  for (const [name, path, options, status] of cases) {
    const x = await call('original-' + name, path, options); assert.equal(x.response.status, status); protectedHeaders(x.response);
    const direct = await original.fetch(req(path, options), env, ctx);
    assert.deepEqual(Object.fromEntries(x.response.headers), Object.fromEntries(direct.headers));
    const want = await direct.json();
    if (name === 'recent') assert.ok(Array.isArray(x.value.requests));
    else assert.deepEqual(x.value, want);
  }
});

test('original success fast path and raw fallback use original proof checks and preserve cache context', async () => {
  const writes = [];
  globalThis.caches = { default: { async match() { return undefined; }, async put(key, response) {
    writes.push({ key: key.url, entry: await response.json() });
  } } };
  for (const [fixture, fallback] of [[block, false], [blocks.find(x => x.height === 800000), true]]) {
    const calls = []; transport = originalTransport(fixture, { fallback, calls });
    const x = await call('original-success-' + (fallback ? 'fallback' : 'fast'), '/hashprice?block_height=' + fixture.height + '&legacy_extra=kept');
    assert.equal(x.response.status, 200, x.body); assert.equal(x.value.status, 'ok', x.body);
    assert.equal(x.value.verification.coinbase_serialized_txid_checked, true);
    assert.equal(x.value.verification.coinbase_merkle_proof_checked, true);
    assert.equal(x.value.block_height, fixture.height);
    assert.ok(calls.some(x => x.startsWith('https://mempool.space/')));
    assert.ok(calls.some(x => x.startsWith('https://blockstream.info/')));
    if (fallback) assert.ok(calls.some(x => x.endsWith('/hex')));
    else assert.equal(x.value.hashprice_sats_per_ph_day, 948924.848225);
  }
  await Promise.all(ctx.promises);
  assert.equal(writes.length, 2);
  for (const write of writes) assert.ok(write.key.startsWith(origin + '/__verified-cache/1.3.1/'));
  delete globalThis.caches;
});

test('unchanged original coinbase functions verify public wire facts and reject tampering', async () => {
  for (const fixture of blocks) {
    const s = originalFixture(fixture);
    assert.equal(await coinbase.coinbaseReward(s.raw, s.tx.txid), fixture.stats.subsidy + fixture.stats.totalfee);
    assert.equal(await coinbase.coinbaseReward(coinbase.serializedCoinbase(s.tx), s.tx.txid), fixture.stats.subsidy + fixture.stats.totalfee);
    await coinbase.verifyCoinbaseProof(s.proof, s.tx.txid, fixture.height, fixture.block.merkleroot);
    await assert.rejects(coinbase.coinbaseReward('00' + s.raw.slice(2), s.tx.txid));
    await assert.rejects(coinbase.verifyCoinbaseProof({ ...s.proof, pos: 1 }, s.tx.txid, fixture.height, fixture.block.merkleroot));
  }
  const fixture = blocks.find(x => x.height !== 840000 && x.height !== 800000);
  transport = originalTransport(fixture, { corrupt: true });
  const refused = await call('original-proof-refusal', '/hashprice?block_height=' + fixture.height);
  assert.equal(refused.response.status, 200); assert.equal(refused.value.status, 'unavailable');
  assert.equal(refused.value.hashprice_sats_per_ph_day, null);
});

test('persist exact body and header evidence for local dispatcher contracts', () => {
  globalThis.fetch = nativeFetch;
  writeFileSync(new URL('./.test-results/dispatcher-contracts.json', import.meta.url), JSON.stringify({
    simulated_transport: true, deployed: false, actual_entry_used: true, public_fixture_replay: true,
    observations,
  }, null, 2) + '\n');
});
