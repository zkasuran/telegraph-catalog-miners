// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createWorker, parseRpcEnvelope, envelopeMetadata, GENESIS } from './worker.mjs';
const fixtures = JSON.parse(readFileSync(new URL('evidence/fixtures-public.json', import.meta.url)));
const f = fixtures.find(x => x.height === 840000);
const legacy = result => ({ result, error: null, id: 1 });
const modern = result => ({ jsonrpc: '2.0', result, id: 1 });

test('documented modern and markerless legacy block-hash replies correlate exactly', () => {
  assert.equal(parseRpcEnvelope(modern(GENESIS), 'getblockhash'), GENESIS);
  assert.equal(parseRpcEnvelope(legacy(GENESIS), 'getblockhash'), GENESIS);
});

for (const [name, data] of [
  ['wrong numeric ID modern', { ...modern(GENESIS), id: 2 }],
  ['string lookalike ID modern', { ...modern(GENESIS), id: '1' }],
  ['wrong numeric ID legacy', { ...legacy(GENESIS), id: 2 }],
  ['string lookalike ID legacy', { ...legacy(GENESIS), id: '1' }],
  ['foreign string ID', { ...legacy(GENESIS), id: 'FOREIGN_ID_MUST_NOT_LEAK' }],
  ['null ID', { ...legacy(GENESIS), id: null }],
  ['missing ID', { result: GENESIS, error: null }],
  ['legacy missing error', { result: GENESIS, id: 1 }],
  ['legacy missing result', { error: null, id: 1 }],
  ['unknown marker', { ...legacy(GENESIS), jsonrpc: '1.0' }],
  ['null marker', { ...legacy(GENESIS), jsonrpc: null }],
  ['object marker', { ...legacy(GENESIS), jsonrpc: { secret: 'REMOTE_CONTENT' } }],
  ['modern both result and null error', { ...modern(GENESIS), error: null }],
  ['modern both result and RPC error', { ...modern(GENESIS), error: { code: -28, message: 'warming' } }],
  ['modern missing result and error', { jsonrpc: '2.0', id: 1 }],
  ['legacy both success and RPC error', { result: GENESIS, error: { code: -28, message: 'warming' }, id: 1 }],
  ['legacy null result and null error', legacy(null)],
  ['array masquerading as reply', [modern(GENESIS)]],
  ['null top level', null],
  ['string top level', 'REMOTE_CONTENT'],
  ['additional top-level field', { ...legacy(GENESIS), method: 'other' }],
]) {
  test('reject ' + name + ' with safe observed envelope metadata', () => {
    assert.throws(() => parseRpcEnvelope(data, 'getblockhash', 200), error => {
      assert.equal(error.http_status, 200);
      assert.equal(error.rpc_error_code, null);
      assert.equal(error.method, 'getblockhash');
      assert.ok(!JSON.stringify(error.envelope).includes('FOREIGN_ID'));
      assert.ok(!JSON.stringify(error.envelope).includes('REMOTE_CONTENT'));
      return true;
    });
  });
}

test('safe metadata distinguishes our numeric ID, its lookalike and a foreign ID', () => {
  assert.equal(envelopeMetadata(modern(GENESIS)).envelope_id, 1);
  assert.equal(envelopeMetadata({ ...modern(GENESIS), id: '1' }).envelope_id_type, 'string');
  assert.equal(envelopeMetadata({ ...modern(GENESIS), id: '1' }).envelope_id_matches, false);
  assert.equal(envelopeMetadata({ ...modern(GENESIS), id: 'private-account-id' }).envelope_id, null);
  assert.equal(envelopeMetadata(legacy(GENESIS)).envelope_jsonrpc, 'absent');
});

for (const [method, value] of [
  ['getblockhash', GENESIS], ['getblockheader', f.header], ['getrawtransaction', f.coinbase_hex],
  ['gettxoutproof', f.proof_hex], ['getblock', f.block], ['getblockstats', f.stats],
]) {
  test('legacy ' + method + ' has a method-specific result shape', () => {
    assert.deepEqual(parseRpcEnvelope(legacy(value), method), value);
    for (const bad of [null, 1, true, [], 'not-a-block', {}]) {
      assert.throws(() => parseRpcEnvelope(legacy(bad), method));
    }
  });
}

for (const [name, error] of [
  ['missing code', { message: 'warming' }], ['string code', { code: '-28', message: 'warming' }],
  ['missing message', { code: -28 }], ['wrong message type', { code: -28, message: {} }],
  ['array error', []], ['string error', 'warming'], ['unknown error field', { code: -28, message: 'warming', secret: 'no' }],
]) {
  test('malformed legacy warmup ' + name + ' cannot enter the retry branch', () => {
    assert.throws(() => parseRpcEnvelope({ result: null, error, id: 1 }, 'getblockhash'), failure => {
      assert.equal(failure.rpc_error_code, null);
      return true;
    });
  });
}

test('documented correlated legacy RPC error preserves numeric code, not remote content', () => {
  assert.throws(() => parseRpcEnvelope({ result: null, error: { code: -5, message: 'REMOTE_ERROR', data: { key: 'REMOTE_DATA' } }, id: 1 }, 'getrawtransaction'), failure => {
    assert.equal(failure.rpc_error_code, -5);
    assert.equal(failure.http_status, 200);
    assert.ok(!JSON.stringify(failure).includes('REMOTE_'));
    return true;
  });
});

test('all seven genuine block fixtures pass through documented legacy envelopes', async () => {
  for (const fixture of fixtures) {
    const calls = [];
    const transport = async (_, options) => {
      const request = JSON.parse(options.body);
      calls.push(request);
      let result;
      if (request.method === 'getblockhash') result = request.params[0] === 0 ? GENESIS : fixture.canonical_hash;
      else if (request.method === 'getblock') result = fixture.block;
      else if (request.method === 'getblockheader') result = fixture.header;
      else if (request.method === 'getblockstats') result = fixture.stats;
      else if (request.method === 'getrawtransaction') result = fixture.coinbase_hex;
      else if (request.method === 'gettxoutproof') result = fixture.proof_hex;
      return Response.json(legacy(result));
    };
    const res = await createWorker(transport).fetch(new Request('https://local.example/hashprice?block_height=' + fixture.height));
    const value = await res.json();
    assert.equal(res.status, 200);
    assert.equal(value.block_hash, fixture.canonical_hash);
    assert.equal(value.verification.coinbase_merkle_inclusion, true);
    assert.equal(calls.length, 8);
    if (fixture.height === 840000) assert.equal(value.hashprice_sats_per_ph_day, 948924.848225);
  }
});

test('original envelope failure path now retains HTTP and shape evidence', async () => {
  const worker = createWorker(async () => Response.json({ result: GENESIS, id: 1 }));
  const res = await worker.fetch(new Request('https://local.example/hashprice?block_height=840000'));
  const value = await res.json();
  assert.equal(res.status, 503);
  assert.equal(value.error.http_status, 200);
  assert.equal(value.error.envelope_jsonrpc, 'absent');
  assert.equal(value.error.envelope_id, 1);
  assert.equal(value.error.envelope_id_type, 'number');
  assert.equal(value.error.result_present, true);
  assert.equal(value.error.error_present, false);
  assert.equal(value.error.top_level_type, 'object');
  assert.equal(value.upstream.subrequests, 1);
  assert.equal(value.upstream.retries, 0);
});
