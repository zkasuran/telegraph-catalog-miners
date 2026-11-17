// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createWorker, GENESIS, RETRY } from './worker.mjs';
const fixture = JSON.parse(readFileSync(new URL('evidence/fixtures-public.json', import.meta.url))).find(x => x.height === 840000);
const stages = ['genesis', 'canonical', 'getblock', 'getblockheader', 'getblockstats', 'getrawtransaction', 'gettxoutproof', 'final'];
const req = () => new Request('https://local.example/hashprice?block_height=840000');

function supplier(policy) {
  const calls = [];
  let canonicalDone = false;
  const transport = async (url, options) => {
    assert.equal(url, 'https://bitcoin-rpc.publicnode.com');
    const body = JSON.parse(options.body);
    const stage = body.method === 'getblockhash' ? body.params[0] === 0 ? 'genesis' : canonicalDone ? 'final' : 'canonical' : body.method;
    calls.push({ stage, body: options.body, signal: options.signal });
    const override = await policy(stage, calls);
    if (override) return override;
    let value;
    if (stage === 'genesis') value = GENESIS;
    else if (stage === 'canonical') { canonicalDone = true; value = fixture.canonical_hash; }
    else if (stage === 'final') value = fixture.canonical_after;
    else if (stage === 'getblock') value = fixture.block;
    else if (stage === 'getblockheader') value = fixture.header;
    else if (stage === 'getblockstats') value = fixture.stats;
    else if (stage === 'getrawtransaction') value = fixture.coinbase_hex;
    else if (stage === 'gettxoutproof') value = fixture.proof_hex;
    return Response.json({ jsonrpc: '2.0', id: 1, result: value });
  };
  return { transport, calls };
}
const errorResponse = code => Response.json({ jsonrpc: '2.0', id: 1, error: { code, message: 'REMOTE_CONTENT_MUST_NOT_LEAK', data: { key: 'REMOTE_DATA_MUST_NOT_LEAK' } } });

for (const target of stages) {
  test('warmup at ' + target + ' retries identical payload twice and preserves proof', async () => {
    const { transport, calls } = supplier(stage => stage === target && calls.filter(x => x.stage === target).length <= 2 ? errorResponse(-28) : null);
    const res = await createWorker(transport).fetch(req());
    const value = await res.json();
    assert.equal(res.status, 200);
    assert.equal(value.hashprice_sats_per_ph_day, 948924.848225);
    assert.equal(value.verification.coinbase_merkle_inclusion, true);
    assert.equal(value.verification.canonical_height_before_and_after, true);
    const selected = calls.filter(x => x.stage === target);
    assert.equal(selected.length, 3);
    assert.equal(new Set(selected.map(x => x.body)).size, 1);
    assert.equal(calls.length, 10);
    assert.equal(value.upstream.subrequests, 10);
    assert.equal(value.upstream.retries, 2);
  });
  for (const code of [-5, -1, -8, -32601, -32603, -32000, -32098, '-28', null]) {
    test('non-transient ' + JSON.stringify(code) + ' at ' + target + ' is not retried or echoed', async () => {
      const { transport, calls } = supplier(stage => stage === target ? errorResponse(code) : null);
      const res = await createWorker(transport).fetch(req());
      const text = await res.text();
      const value = JSON.parse(text);
      assert.equal(res.status, 503);
      assert.equal(calls.filter(x => x.stage === target).length, 1);
      assert.equal(value.error.method, target === 'genesis' || target === 'canonical' || target === 'final' ? 'getblockhash' : target);
      assert.equal(value.error.rpc_error_code, Number.isSafeInteger(code) ? code : null);
      assert.equal(value.error.attempts, 1);
      assert.equal(value.upstream.retries, 0);
      assert.ok(!Object.hasOwn(value, 'hashprice_sats_per_ph_day'));
      assert.ok(!text.includes('REMOTE_'));
    });
  }
}

test('persistent warmup stops at three attempts and reports exhausted budget', async () => {
  const { transport, calls } = supplier(stage => stage === 'getrawtransaction' ? errorResponse(-28) : null);
  const res = await createWorker(transport).fetch(req());
  const value = await res.json();
  assert.equal(res.status, 503);
  assert.equal(calls.filter(x => x.stage === 'getrawtransaction').length, RETRY.attemptsPerMethod);
  assert.equal(value.error.rpc_error_code, -28);
  assert.equal(value.error.retry_exhausted, true);
  assert.equal(value.error.attempts, 3);
});

test('concurrent warmups share four extra calls and cannot approach 50 subrequests', async () => {
  const targets = new Set(['getblockheader', 'getblockstats', 'getrawtransaction', 'gettxoutproof']);
  const { transport, calls } = supplier(stage => targets.has(stage) ? errorResponse(-28) : null);
  const res = await createWorker(transport).fetch(req());
  const value = await res.json();
  assert.equal(res.status, 503);
  assert.ok(value.upstream.retry_slots_used <= 4);
  assert.ok(value.upstream.retries <= 4);
  assert.ok(value.upstream.subrequests <= 12);
  assert.ok(calls.length <= 12);
  for (const stage of targets) assert.ok(calls.filter(x => x.stage === stage).length <= 3);
});

test('warmup followed by forged proof still fails, with no proof retry', async () => {
  const { transport, calls } = supplier(stage => {
    if (stage !== 'gettxoutproof') return null;
    if (calls.filter(x => x.stage === stage).length === 1) return errorResponse(-28);
    return Response.json({ jsonrpc: '2.0', id: 1, result: '00' });
  });
  const res = await createWorker(transport).fetch(req());
  const value = await res.json();
  assert.equal(res.status, 503);
  assert.equal(calls.filter(x => x.stage === 'gettxoutproof').length, 2);
  assert.ok(!Object.hasOwn(value, 'hashprice_sats_per_ph_day'));
  assert.ok(!calls.some(x => x.stage === 'final'));
});

test('same original 15-second deadline cancels a stall after retry', async () => {
  const { transport, calls } = supplier(stage => {
    if (stage !== 'getrawtransaction') return null;
    if (calls.filter(x => x.stage === stage).length === 1) return errorResponse(-28);
    const signal = calls.at(-1).signal;
    return new Response(new ReadableStream({ start(controller) {
      signal.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError')), { once: true });
    } }), { headers: { 'Content-Type': 'application/json' } });
  });
  const start = Date.now();
  const res = await createWorker(transport).fetch(req());
  const value = await res.json();
  assert.equal(res.status, 503);
  assert.equal(value.error.code, 'upstream_timeout');
  assert.equal(value.error.method, 'getrawtransaction');
  assert.equal(value.error.attempts, 2);
  assert.ok(Date.now() - start < 16500);
});

test('reorganization after final-lookup warmup is rejected without another retry', async () => {
  const { transport, calls } = supplier(stage => {
    if (stage !== 'final') return null;
    if (calls.filter(x => x.stage === stage).length === 1) return errorResponse(-28);
    return Response.json({ jsonrpc: '2.0', id: 1, result: '0'.repeat(64) });
  });
  const res = await createWorker(transport).fetch(req());
  const value = await res.json();
  assert.equal(res.status, 503);
  assert.equal(value.error.code, 'canonical_hash_changed');
  assert.equal(calls.filter(x => x.stage === 'final').length, 2);
  assert.ok(!Object.hasOwn(value, 'hashprice_sats_per_ph_day'));
});
