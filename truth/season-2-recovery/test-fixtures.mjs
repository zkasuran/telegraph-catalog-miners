// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Test-only transport adapters over the included PublicNode public consensus facts.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { Reader, unhex, hex } from './sources/hashprice/proof.mjs';
export const blocks = JSON.parse(readFileSync(new URL('./sources/hashprice/evidence/fixtures-public.json', import.meta.url)));
export const block = blocks.find(x => x.height === 840000);
export const corporate = JSON.parse(readFileSync(new URL('./sources/corporate/evidence/public/valid-name.json', import.meta.url)));
export const inactive = JSON.parse(readFileSync(new URL('./sources/corporate/evidence/public/inactive.json', import.meta.url)));
export const ambiguous = JSON.parse(readFileSync(new URL('./sources/corporate/evidence/public/ambiguous.json', import.meta.url)));
const json = value => Response.json(value);
const sha256d = bytes => createHash('sha256').update(createHash('sha256').update(bytes).digest()).digest();
const display = bytes => Buffer.from(bytes).reverse().toString('hex');

export function bitcoinRPC(fixture = block, change = (_call, value) => value) {
  let heightReads = 0;
  return async (url, options) => {
    assert.equal(url, 'https://bitcoin-rpc.publicnode.com');
    assert.equal(options.method, 'POST');
    const call = JSON.parse(options.body);
    const values = { getblock: fixture.block, getblockheader: fixture.header,
      getblockstats: fixture.stats, getrawtransaction: fixture.coinbase_hex, gettxoutproof: fixture.proof_hex };
    let value;
    if (call.method === 'getblockhash') {
      value = call.params[0] === 0 ? '000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f'
        : ++heightReads % 2 === 1 ? fixture.canonical_hash : fixture.canonical_after;
    } else {
      assert.ok(Object.hasOwn(values, call.method));
      assert.ok(JSON.stringify(call.params).includes(fixture.canonical_hash));
      value = values[call.method];
    }
    return json({ jsonrpc: '2.0', id: 1, result: change(call, structuredClone(value)) });
  };
}

export function accountRPC({ wei = 9007199254740993001n, fail = false, calls = [] } = {}) {
  const hash = '0x' + 'a'.repeat(64);
  return async (url, options) => {
    assert.ok(['https://ethereum-public.nodies.app', 'https://eth.drpc.org',
      'https://polygon-bor-rpc.publicnode.com', 'https://polygon.drpc.org',
      'https://arbitrum-one-rpc.publicnode.com', 'https://arbitrum-one-public.nodies.app'].includes(url));
    assert.equal(options.method, 'POST');
    assert.equal(options.redirect, 'manual');
    const request = JSON.parse(options.body); calls.push({ url, request });
    if (fail) return json(request.map(x => ({ jsonrpc: '2.0', id: x.id, error: { code: -32000, message: 'fixture unavailable' } })));
    return json(request.map(call => {
      let result;
      if (call.method === 'eth_chainId') result = url.includes('polygon') ? '0x89' : url.includes('arbitrum') ? '0xa4b1' : '0x1';
      else if (call.method === 'eth_getBlockByNumber') {
        assert.deepEqual(call.params, ['0x2a', false]);
        result = { number: '0x2a', hash, parentHash: '0x' + 'b'.repeat(64), stateRoot: '0x' + 'c'.repeat(64), timestamp: '0x100' };
      } else {
        assert.deepEqual(call.params, ['0x0000000000000000000000000000000000000001', { blockHash: hash, requireCanonical: true }]);
        assert.ok(['eth_getBalance', 'eth_getTransactionCount'].includes(call.method));
        result = call.method === 'eth_getBalance' ? '0x' + wei.toString(16) : '0x7';
      }
      return { jsonrpc: '2.0', id: call.id, result };
    }).reverse());
  };
}

// Esplora JSON and branch are derived from public raw Bitcoin wire facts.
// These adapters are test-only. Neither implementation module is rewritten.
export function originalFixture(f = block) {
  const r = new Reader(unhex(f.coinbase_hex));
  const version = Number(r.uint(4));
  if (r.bytes[r.index] === 0) r.take(2);
  assert.equal(r.compact(1), 1);
  const vin = [{ txid: display(r.take(32)), vout: Number(r.uint(4)),
    scriptsig: hex(r.take(r.compact(100))), sequence: Number(r.uint(4)), is_coinbase: true }];
  const vout = Array.from({ length: r.compact(1000) }, () => ({ value: Number(r.uint(8)), scriptpubkey: hex(r.take(r.compact(10000))) }));
  const pr = new Reader(unhex(f.proof_hex));
  assert.equal(hex(pr.take(80)), f.header);
  const count = Number(pr.uint(4));
  const hashes = Array.from({ length: pr.compact(64) }, () => Buffer.from(pr.take(32)));
  const flags = pr.take(pr.compact(32));
  let usedBits = 0, usedHashes = 0;
  const branch = [];
  const width = h => Math.ceil(count / 2 ** h);
  function walk(height, pos) {
    const flag = flags[usedBits >> 3] >> (usedBits++ & 7) & 1;
    if (height === 0 || !flag) return hashes[usedHashes++];
    const left = walk(height - 1, pos * 2);
    const right = pos * 2 + 1 < width(height - 1) ? walk(height - 1, pos * 2 + 1) : left;
    if (pos === 0) branch.push(display(right));
    return sha256d(Buffer.concat([left, right]));
  }
  let h = 0; while (width(h) > 1) h++;
  assert.equal(display(walk(h, 0)), f.block.merkleroot);
  assert.equal(usedHashes, hashes.length);
  const tx = { txid: f.block.tx[0], version, locktime: 0, vin, vout,
    status: { confirmed: true, block_height: f.height, block_hash: f.block.hash, block_time: f.block.time } };
  // Locktime is the final four wire bytes after any witness stack.
  tx.locktime = Buffer.from(unhex(f.coinbase_hex)).readUInt32LE(f.coinbase_hex.length / 2 - 4);
  return { block: { id: f.block.hash, height: f.height, version: f.block.version,
    timestamp: f.block.time, nonce: f.block.nonce, bits: parseInt(f.block.bits, 16), difficulty: f.block.difficulty,
    merkle_root: f.block.merkleroot, previousblockhash: f.block.previousblockhash, stale: 0,
    extras: { header: f.header, reward: f.stats.subsidy + f.stats.totalfee, totalFees: f.stats.totalfee } },
    tx, proof: { block_height: f.height, pos: 0, merkle: branch }, raw: f.coinbase_hex };
}

export function originalTransport(f = block, { fallback = false, corrupt = false, calls = [] } = {}) {
  const sample = originalFixture(f);
  return async (url) => {
    calls.push(url);
    const u = new URL(url);
    if (u.hostname === 'mempool.space') {
      assert.equal(u.pathname, '/api/v1/blocks/' + f.height);
      const b = structuredClone(sample.block); if (corrupt) b.extras.header = 'ff' + b.extras.header.slice(2);
      return json([b]);
    }
    assert.equal(u.hostname, 'blockstream.info');
    if (u.pathname.endsWith('/txs/0')) return fallback ? new Response('', { status: 413 }) : json([sample.tx]);
    if (u.pathname.endsWith('/merkle-proof')) return json(sample.proof);
    if (u.pathname === '/api/block-height/' + f.height) return new Response(f.block.hash);
    if (u.pathname.endsWith('/txid/0')) return new Response(f.block.tx[0]);
    if (u.pathname.endsWith('/hex')) return new Response(sample.raw);
    assert.fail('unexpected original transport URL ' + url);
  };
}
