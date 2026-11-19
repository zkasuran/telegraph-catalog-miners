// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import test from 'node:test';
import assert from 'node:assert/strict';
import worker, { LIMITS, CHAINS, quantity, gwei, parseQuery, rpcBatch, observe, serialize, handle } from './worker.mjs';
const URL = 'https://fixture.example/metric?address=0x0000000000000000000000000000000000000001&block=42';
const HASH = '0x' + 'a'.repeat(64);
const HEADER = { number: '0x2a', hash: HASH, parentHash: '0x' + 'b'.repeat(64), stateRoot: '0x' + 'c'.repeat(64), timestamp: '0x100' };
const query = parseQuery(URL);
function fake({ change, fail } = {}) {
  const calls = [];
  const fetcher = async (url, opts) => {
    const request = JSON.parse(opts.body); calls.push({ url, request });
    if (fail) return fail(url, request, opts);
    const entries = request.map((item) => {
      let result = item.method === 'eth_chainId' ? '0x1' : item.method === 'eth_getBlockByNumber' ? { ...HEADER } : item.method === 'eth_getBalance' ? '0x20000000000001' : '0x7';
      if (change) result = change(result, item, url, calls.length);
      return { id: item.id, jsonrpc: '2.0', result };
    });
    return new Response(JSON.stringify(entries.reverse()), { headers: { 'content-type': 'application/json' } });
  };
  return { fetcher, calls };
}
test('defaults and strict canonical inputs', () => {
  assert.deepEqual(query, { address: '0x0000000000000000000000000000000000000001', block: 42, chain: 'ethereum', metric: 'balance' });
  assert.equal(parseQuery(URL + '&chain=polygon&metric=tx_count').chain, 'polygon');
  for (const block of ['-1', '1.0', '1e2', 'latest', '0x2a', '01', '', '1000000000001', '9007199254740993']) {
    assert.throws(() => parseQuery(URL.replace('block=42', `block=${block}`)), /invalid_block/);
  }
  for (const tail of ['&chain=base', '&chain=ETHEREUM', '&metric=transfers', '&block=42', '&metric=', '&chain=', '&provider=https://localhost']) {
    assert.throws(() => parseQuery(URL + tail));
  }
  assert.throws(() => parseQuery(URL.replace('0001', '000z')), /invalid_address/);
  assert.throws(() => parseQuery(URL + '&x=' + 'a'.repeat(512)), /url_too_large/);
  assert.equal(parseQuery(URL.replace('block=42', 'block=0')).block, 0);
  assert.equal(parseQuery(URL.replace('block=42', 'block=1000000000000')).block, 1000000000000);
});
test('lossless six-decimal Gwei boundaries and exact companions', () => {
  const vectors = [[0n, '0.000000'], [1n, '0.000000'], [499n, '0.000000'], [500n, '0.000001'], [999n, '0.000001'], [1000n, '0.000001'], [999999999n, '1.000000'], [1000000000n, '1.000000'], [9007199254740993001n, '9007199254.740993'], [100000000000000000000000000000n, '100000000000000000000.000000']];
  for (const [wei, scored] of vectors) assert.equal(gwei(wei).scored, scored);
  assert.throws(() => gwei(-1n)); assert.throws(() => gwei(LIMITS.weiMax + 1n));
  let seed = 19n;
  for (let i = 0; i < 1000; i++) {
    seed = (seed * 6364136223846793005n + 1442695040888963407n) % LIMITS.weiMax;
    const result = gwei(seed); const roundedWei = BigInt(result.scored.replace('.', '')) * 1000n;
    assert.ok(roundedWei - seed <= 500n && seed - roundedWei <= 499n);
    assert.equal(BigInt(result.exact.replace('.', '')), seed);
  }
});
test('bounded canonical hex quantities', () => {
  assert.equal(quantity('0x0'), 0n); assert.equal(quantity('0xabcdef'), 11259375n);
  for (const x of ['0x', '0x00', '10', '-1', 42, null, '0x' + 'f'.repeat(65), '0x1g']) assert.throws(() => quantity(x));
});
test('exact wire number and fresh hash-pinned state on every request', async () => {
  const fixture = fake();
  const sample = await observe(query, fixture.fetcher);
  const wire = serialize(sample);
  assert.match(wire, /"balance_gwei":9007199\.254741[,}]/);
  assert.equal(JSON.parse(wire).balance_wei, '9007199254740993');
  assert.equal(JSON.parse(wire).tx_count, 7);
  for (const call of fixture.calls.flatMap(c => c.request)) {
    if (['eth_getBalance', 'eth_getTransactionCount'].includes(call.method)) {
      assert.deepEqual(call.params, [query.address, { blockHash: HASH, requireCanonical: true }]);
    }
    assert.ok(!JSON.stringify(call.params).includes('latest'));
  }
  await observe(query, fixture.fetcher);
  assert.equal(fixture.calls.length, 12, 'no header or balance reuse');
});
test('wrong chain, height, header, state and reorg controls fail', async () => {
  const changes = [
    (r, c) => c.method === 'eth_chainId' ? '0x89' : r,
    (r, c) => c.method === 'eth_getBlockByNumber' ? { ...r, number: '0x2b' } : r,
    (r, c, u) => c.method === 'eth_getBlockByNumber' && u.includes('drpc') ? { ...r, stateRoot: '0x' + 'd'.repeat(64) } : r,
    (r, c, u) => c.method === 'eth_getBalance' && u.includes('drpc') ? '0x1' : r,
    (r, c, u) => c.method === 'eth_getTransactionCount' && u.includes('drpc') ? '0x8' : r,
    (r, c, u, n) => c.method === 'eth_getBlockByNumber' && n > 4 ? { ...r, hash: '0x' + 'e'.repeat(64) } : r,
    (r, c) => c.method === 'eth_getBlockByNumber' ? null : r,
    (r, c) => c.method === 'eth_getTransactionCount' ? '0xe8d4a51001' : r,
  ];
  for (const change of changes) await assert.rejects(observe(query, fake({ change }).fetcher));
});
test('malformed envelopes, errors, HTML, redirect and oversized responses fail', async () => {
  const cases = [
    () => new Response('[]', { headers: { 'content-type': 'application/json' } }),
    () => new Response('[{"id":1,"jsonrpc":"2.0","result":"0x1"},{"id":1,"jsonrpc":"2.0","result":"0x1"}]', { headers: { 'content-type': 'application/json' } }),
    () => new Response('[{"id":1,"jsonrpc":"2.0","error":{"code":4444,"message":"pruned"}},{"id":2,"jsonrpc":"2.0","result":null}]', { headers: { 'content-type': 'application/json' } }),
    () => new Response('<html>RPC home</html>', { headers: { 'content-type': 'text/html' } }),
    () => new Response('{}', { status: 429 }),
    () => new Response('{}', { status: 302 }),
    () => new Response('x'.repeat(LIMITS.rpcBytes + 1), { headers: { 'content-type': 'application/json' } }),
    () => new Response('[]', { headers: { 'content-type': 'application/json', 'content-length': String(LIMITS.rpcBytes + 1) } }),
    () => new Response('garbage', { headers: { 'content-type': 'application/json' } }),
  ];
  for (const fail of cases) await assert.rejects(observe(query, fake({ fail }).fetcher));
});
test('timeout applies to slow streamed body', async () => {
  const controller = new AbortController();
  const promise = rpcBatch(CHAINS.ethereum.services[0], [['eth_chainId', []]], controller.signal,
    async () => new Response(new ReadableStream({ start() {} }), { headers: { 'content-type': 'application/json' } }));
  setTimeout(() => controller.abort(), 30);
  await assert.rejects(promise, /rpc_timeout/);
});
test('real handler returns nonzero HTTP failures with no numeric placeholders', async () => {
  const ok = await handle(new Request(URL), fake().fetcher);
  assert.equal(ok.status, 200); assert.equal(ok.headers.get('cache-control'), 'no-store');
  const noBlock = await handle(new Request(URL.replace('&block=42', '')), fake().fetcher);
  assert.equal(noBlock.status, 400); assert.ok(!(await noBlock.text()).includes('balance_gwei'));
  const unavailable = await handle(new Request(URL), fake({ change: (r, c) => c.method === 'eth_getBlockByNumber' ? null : r }).fetcher);
  assert.equal(unavailable.status, 502); assert.ok(!(await unavailable.text()).includes('balance_gwei'));
  const post = await handle(new Request(URL, { method: 'POST' }), fake().fetcher);
  assert.equal(post.status, 405);
});
test('four-request isolate saturation refuses without another upstream call', async () => {
  let release;
  const waiting = new Promise(resolve => { release = resolve; });
  const upstream = fake(); let calls = 0;
  const stalled = async (...args) => { calls++; await waiting; return upstream.fetcher(...args); };
  const pending = Array.from({ length: LIMITS.maxInflight }, () => handle(new Request(URL), stalled));
  await new Promise(resolve => setTimeout(resolve, 10));
  const before = calls;
  const denied = await handle(new Request(URL), stalled);
  assert.equal(denied.status, 429); assert.equal(calls, before);
  release();
  for (const response of await Promise.all(pending)) assert.equal(response.status, 200);
});
test('Cloudflare fetch signature does not confuse env with the fetch transport', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = fake().fetcher;
  try {
    const response = await worker.fetch(new Request(URL), {}, { waitUntil() {} });
    assert.equal(response.status, 200);
  } finally { globalThis.fetch = original; }
});
