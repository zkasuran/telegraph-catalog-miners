// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createWorker, verifyBundle, parseHeight, GENESIS, BOUNDS } from './worker.mjs';
import { compactTarget, sixDecimals } from './core.mjs';
import { decodeCoinbase, Reader, unhex } from './proof.mjs';

const fixtures = JSON.parse(readFileSync(new URL('evidence/fixtures-public.json', import.meta.url)));
const f = fixtures.find(x => x.height === 840000);
const request = q => new Request('https://truth.example/hashprice' + q);
const packet = result => Response.json({ jsonrpc: '2.0', id: 1, result });
function transport(fixture, mutate = () => {}) {
  const calls = [];
  let heightReads = 0;
  const fetcher = async (url, options) => {
    assert.equal(url, 'https://bitcoin-rpc.publicnode.com');
    assert.equal(options.method, 'POST');
    assert.equal(options.redirect, 'manual');
    const call = JSON.parse(options.body);
    calls.push(call);
    let value;
    if (call.method === 'getblockhash') {
      value = call.params[0] === 0 ? GENESIS : (++heightReads === 1 ? fixture.canonical_hash : fixture.canonical_after);
    } else if (call.method === 'getblock') value = fixture.block;
    else if (call.method === 'getblockheader') value = fixture.header;
    else if (call.method === 'getblockstats') value = fixture.stats;
    else if (call.method === 'getrawtransaction') value = fixture.coinbase_hex;
    else if (call.method === 'gettxoutproof') value = fixture.proof_hex;
    else throw new Error('unexpected_method');
    value = structuredClone(value);
    mutate(call, value);
    return packet(value);
  };
  return { fetcher, calls };
}

test('all seven real block fixtures through actual descriptor-compatible GET route', async () => {
  const outputs = [];
  for (const fixture of fixtures) {
    const { fetcher, calls } = transport(fixture);
    const r = await createWorker(fetcher).fetch(request('?block_height=' + fixture.height));
    const text = await r.text();
    assert.equal(r.status, 200, text);
    assert.match(text, /"hashprice_sats_per_ph_day":\d+\.\d{6}[,}]/);
    const value = JSON.parse(text);
    assert.equal(value.block_height, fixture.height);
    assert.equal(value.block_hash, fixture.canonical_hash);
    assert.equal(value.verification.coinbase_merkle_inclusion, true);
    assert.equal(value.hashprice_usd_per_ph_day, null);
    assert.equal(value.source, 'https://bitcoin-rpc.publicnode.com');
    assert.equal(value.verification.independent_runtime_provider, false);
    assert.equal(calls.length, 8);
    for (const call of calls.filter(x => !['getblockhash'].includes(x.method))) {
      assert.ok(JSON.stringify(call.params).includes(fixture.canonical_hash));
    }
    delete value.fetched_at;
    const stableText = text.replace(/"fetched_at":"[^"]+",/, '');
    outputs.push({ height: fixture.height, response: value, response_text: stableText });
  }
  writeFileSync(new URL('evidence/replay-results.json', import.meta.url), JSON.stringify({ simulated_transport: true, real_chain_fixtures: true, results: outputs }, null, 2) + '\n');
});

test('height 840000 high fees remain in the gross reward', async () => {
  const value = await verifyBundle(f);
  assert.equal(value.hashprice_sats_per_ph_day, 948924.848225);
  assert.equal(value.block_reward_sats, 4075061499);
  assert.equal(value.subsidy_sats, 312500000);
  assert.equal(value.fees_sats, 3762561499);
  assert.notEqual(value.hashprice_sats_per_ph_day, 72769.212230);
});

test('missing, negative, fractional, exponent, hash and duplicate input reject before I/O', async () => {
  let count = 0;
  const worker = createWorker(() => { count++; throw new Error('unexpected'); });
  const invalid = ['', '?block_height=-1', '?block_height=1.2', '?block_height=1e6', '?block_height=0840000',
    '?block_height=100000001', '?block_height=NaN', '?block_height=Infinity', '?block_height=%20',
    '?block_height=' + f.canonical_hash, '?block_hash=' + f.canonical_hash,
    '?block_height=840000&block_height=840000', '?block_height=840000&url=https://evil.test',
    '?block_height={block_height}', '?block_height=%00', '?block_height=0x123'];
  for (let i = 0; i < 1000; i++) invalid.push('?block_height=' + i + '.0');
  for (const query of invalid) {
    const res = await worker.fetch(request(query));
    assert.equal(res.status, 400);
    assert.ok(!Object.hasOwn(await res.json(), 'hashprice_sats_per_ph_day'));
  }
  assert.equal(count, 0);
});

test('strict route, URL size and GET method', async () => {
  const worker = createWorker(() => { throw new Error('unexpected'); });
  for (const route of ['/hashprice/{block_height}', '/hashprice/%7Bblock_height%7D', '/other']) {
    assert.equal((await worker.fetch(new Request('https://truth.example' + route))).status, 404);
  }
  assert.equal((await worker.fetch(new Request('https://truth.example/hashprice', { method: 'POST' }))).status, 405);
  assert.equal((await worker.fetch(request('?block_height=' + '1'.repeat(600)))).status, 400);
  assert.equal(parseHeight(request('?block_height=0')), 0);
  assert.equal(parseHeight(request('?block_height=100000000')), 100000000);
});

const mutations = {
  'wrong mainnet genesis': x => { x.genesis = '0'.repeat(64); },
  'canonical height changed': x => { x.canonical_after = '0'.repeat(64); },
  'wrong block height': x => { x.block.height++; },
  'wrong block hash': x => { x.block.hash = '0'.repeat(64); },
  'orphaned block': x => { x.block.confirmations = -1; },
  'forged header byte': x => { x.header = '01' + x.header.slice(2); x.proof_hex = '01' + x.proof_hex.slice(2); },
  'proof of work invalid nonce': x => { x.block.nonce++; },
  'wrong merkle root': x => { x.block.merkleroot = '0'.repeat(64); },
  'wrong difficulty': x => { x.block.difficulty *= 2; },
  'negative compact target': x => { x.block.bits = '1d80ffff'; },
  'invalid fee sum': x => { x.stats.totalfee++; },
  'negative fees': x => { x.stats.totalfee = -1; },
  'string fee amount': x => { x.stats.totalfee = String(x.stats.totalfee); },
  'unsafe fee amount': x => { x.stats.totalfee = Number.MAX_SAFE_INTEGER + 1; },
  'wrong subsidy': x => { x.stats.subsidy *= 2; },
  'stats from another height': x => { x.stats.height++; },
  'stats from another block': x => { x.stats.blockhash = '0'.repeat(64); },
  'underclaimed reward': x => { x.stats.totalfee++; },
  'coinbase bytes tampered': x => { x.coinbase_hex = '00' + x.coinbase_hex.slice(2); },
  'coinbase witness flags': x => { x.coinbase_hex = x.coinbase_hex.slice(0, 10) + '02' + x.coinbase_hex.slice(12); },
  'coinbase bytes truncated': x => { x.coinbase_hex = x.coinbase_hex.slice(0, -2); },
  'coinbase trailing bytes': x => { x.coinbase_hex += '00'; },
  'coinbase hex uppercase': x => { x.coinbase_hex = x.coinbase_hex.toUpperCase(); },
  'coinbase position replaced': x => { x.block.tx[0] = x.block.tx[1]; },
  'proof header changed': x => { x.proof_hex = '01' + x.proof_hex.slice(2); },
  'proof sibling changed': x => { x.proof_hex = x.proof_hex.slice(0, 230) + 'ff' + x.proof_hex.slice(232); },
  'proof missing bits': x => { x.proof_hex = x.proof_hex.slice(0, -2); },
  'proof unused trailing data': x => { x.proof_hex += '00'; },
  'proof count mismatches block': x => { x.block.nTx--; },
};
for (const [name, mutate] of Object.entries(mutations)) {
  test('adversarial ' + name, async () => {
    const changed = structuredClone(f);
    mutate(changed);
    await assert.rejects(verifyBundle(changed));
  });
}

test('CompactSize rejects nonminimal representations and excessive counts', () => {
  assert.throws(() => new Reader(unhex('fdfc00')).compact(1000), /noncanonical/);
  assert.throws(() => new Reader(unhex('feffff0000')).compact(100000), /noncanonical/);
  assert.throws(() => new Reader(unhex('ff0000000001000000')).compact(100000), /count_limit/);
});

test('coinbase parser rejects all truncation lengths of a real serialized transaction', async () => {
  for (let length = 0; length < f.coinbase_hex.length; length += 2) {
    await assert.rejects(decodeCoinbase(f.coinbase_hex.slice(0, length), f.height));
  }
});

test('1000 deterministic fuzz byte strings cannot be accepted as a coinbase', async () => {
  let seed = 103;
  for (let i = 0; i < 1000; i++) {
    let text = '';
    for (let j = 0; j < i % 120; j++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      text += (seed & 255).toString(16).padStart(2, '0');
    }
    await assert.rejects(decodeCoinbase(text, 840000));
  }
});

test('compact target mainnet range and published small-exponent vectors', () => {
  assert.equal(compactTarget(0x01123456), 0x12n);
  assert.equal(compactTarget(0x02008000), 0x80n);
  assert.equal(compactTarget(0x05009234), 0x92340000n);
  for (const bits of [0, 0x01003456, 0x1d80ffff, 0x207fffff, 0x2300ffff, -1, 0x100000000, 1.2]) {
    assert.throws(() => compactTarget(bits));
  }
});

test('rounding properties, linearity and half-up boundaries across 5000 integer cases', () => {
  assert.equal(sixDecimals(1n, 2000000n), 0.000001);
  let state = 103n;
  for (let i = 0; i < 5000; i++) {
    state = (state * 1664525n + 1013904223n) % 4294967296n;
    const n = state % 1000000000n + 1n;
    const d = state % 999999n + 1n;
    const scaled = (n * 1000000n) / d;
    const rounded = BigInt(Math.round(sixDecimals(n, d) * 1000000));
    assert.ok(rounded === scaled || rounded === scaled + 1n);
    assert.equal(sixDecimals(n * 7n, d * 7n), sixDecimals(n, d));
    assert.ok(sixDecimals(n + 1n, d) >= sixDecimals(n, d));
  }
});

for (const [name, supply] of [
  ['HTTP 429', async () => new Response('limited', { status: 429 })],
  ['HTTP 302', async () => new Response(null, { status: 302 })],
  ['HTTP 500', async () => new Response('error', { status: 500 })],
  ['HTML masquerading as success', async () => new Response('<html>captcha</html>')],
  ['invalid JSON', async () => new Response('{', { headers: { 'Content-Type': 'application/json' } })],
  ['RPC error', async () => Response.json({ jsonrpc: '2.0', id: 1, error: { code: -8, message: 'not found' } })],
  ['wrong RPC id', async () => Response.json({ jsonrpc: '2.0', id: 2, result: GENESIS })],
  ['declared oversized body', async () => new Response('{}', { headers: { 'Content-Type': 'application/json', 'Content-Length': String(BOUNDS.response + 1) } })],
  ['stream oversized body', async () => new Response(' '.repeat(BOUNDS.response + 1), { headers: { 'Content-Type': 'application/json' } })],
]) {
  test('upstream ' + name + ' returns no scored field', async () => {
    const res = await createWorker(supply).fetch(request('?block_height=840000'));
    assert.equal(res.status, 503);
    assert.ok(!Object.hasOwn(await res.json(), 'hashprice_sats_per_ph_day'));
  });
}

test('reorganization between calculation and final canonical lookup fails closed', async () => {
  const changed = structuredClone(f);
  changed.canonical_after = '0'.repeat(64);
  const { fetcher } = transport(changed);
  const res = await createWorker(fetcher).fetch(request('?block_height=840000'));
  assert.equal(res.status, 503);
  assert.equal((await res.json()).error.code, 'canonical_hash_changed');
});

test('source stalls after response headers, deadline bounds total request', async () => {
  const supply = async (_, { signal }) => new Response(new ReadableStream({
    start(controller) { signal.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError'))); },
  }), { headers: { 'Content-Type': 'application/json' } });
  const start = Date.now();
  const res = await createWorker(supply).fetch(request('?block_height=840000'));
  assert.equal(res.status, 503);
  assert.equal((await res.json()).error.code, 'upstream_timeout');
  assert.ok(Date.now() - start < BOUNDS.deadlineMs + 1500);
});

test('in-flight request limit and failure release', async () => {
  let resolve;
  const worker = createWorker(() => new Promise(r => { resolve = r; }));
  const first = worker.fetch(request('?block_height=840000'));
  assert.equal((await worker.fetch(request('?block_height=840000'))).status, 429);
  resolve(new Response('bad', { status: 500 }));
  assert.equal((await first).status, 503);
  const next = worker.fetch(request('?block_height=840000'));
  resolve(new Response('bad', { status: 500 }));
  assert.equal((await next).status, 503);
});

test('security headers cover successful and failed responses', async () => {
  const { fetcher } = transport(f);
  const worker = createWorker(fetcher);
  for (const query of ['?block_height=840000', '?block_height=-1']) {
    const res = await worker.fetch(request(query));
    assert.equal(res.headers.get('cache-control'), 'no-store');
    assert.equal(res.headers.get('x-content-type-options'), 'nosniff');
    assert.equal(res.headers.get('content-security-policy'), "default-src 'none'; frame-ancestors 'none'");
  }
});
