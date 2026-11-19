// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {normalizeOperation, operationKey, validateAdoption} from './registrar_binding.mjs';
const release = JSON.parse(readFileSync(new URL('manifest.json', import.meta.url)));
const fx = release.families.FX_NOW;
const op = {chain_id: 11155111, campaign: release.version, family: 'FX_NOW',
  diamond: '0x72a70e949362670Bf8bceAd24eFC5b8f950aa08b', token: '0xcc194e1a843b658526d0d5c431f670028582ae17',
  owner: '0x8b224783FE5b3c52B7DB0cb9B1754f8812b75287', fee_address: '0x8b224783FE5b3c52B7DB0cb9B1754f8812b75287',
  facet: '0x5d31e420da5551f93bf4c16a6da8e9cc895363c1', facet_code_hash: '0x06b0c5a10e54593b521bb6f247b209d86ae4711c24894e52e3e448c556882464',
  family_id: '0x' + '1'.repeat(64), wasm_hash: fx.keccak, signed_hash: '0x' + '2'.repeat(64),
  nonce: '0', value: '0', bond: '5000000', selector: '0xc4feb8ed', calldata: '0xc4feb8ed00',
  binary_url: fx.binary_url, runnable_url: fx.runnable_url, release_proof_sha256: 'a'.repeat(64), deployment_proof_sha256: 'b'.repeat(64)};
const normalized = normalizeOperation(op);
const identity = {emitter: normalized.diamond, author: normalized.owner, family_id: op.family_id,
  wasm_hash: op.wasm_hash, binary_url: op.binary_url, family: op.family, fee_address: normalized.fee_address, runnable_url: op.runnable_url};
const block = '0x' + '3'.repeat(64);
const evidence = {deployment_proof_sha256: op.deployment_proof_sha256, now_ms: 1000, proof_valid_until_ms: 2000,
  facet: op.facet, facet_code_hash: op.facet_code_hash, broadcast_state: 'reconciled', abi_bound_to_current_facet: true,
  transaction: {chain_id: op.chain_id, hash: op.signed_hash, from: op.owner, to: op.diamond, nonce: op.nonce, value: '0', data: op.calldata},
  receipt: {status: 1, safe: true, canonical: true, transaction_hash: op.signed_hash, block_hash: block, block_number: '123', from: op.owner, to: op.diamond},
  safe_block_hash: block, safe_block_number: '123', events: [{...identity, id: '101'}],
  getter: {...identity, id: '101', active: true, bond: op.bond, block_hash: block, block_number: '123'}};
assert.equal(validateAdoption(op, op, evidence).state, 'prepared_transaction_confirmed');
assert.match(operationKey(op), /11155111:0x72a70/);
let negatives = 0;
for (const mutate of [e => e.transaction.to = '0x3624EB012B5E998BB57B6bE6C9Cfc9c4bc5eb99b',
  e => e.transaction.from = '0x' + '4'.repeat(40), e => e.transaction.nonce = '1', e => e.transaction.chain_id = 1,
  e => e.transaction.data = '0xdeadbeef', e => e.transaction.hash = '0x' + '4'.repeat(64),
  e => e.receipt.status = 0, e => e.receipt.canonical = false, e => e.safe_block_hash = '0x' + '4'.repeat(64),
  e => e.events[0].wasm_hash = '0x' + '4'.repeat(64), e => e.events[0].runnable_url += '/changed',
  e => e.events.push(e.events[0]), e => e.getter.bond = '0', e => e.getter.id = '102',
  e => e.broadcast_state = 'unknown', e => e.abi_bound_to_current_facet = false,
  e => e.deployment_proof_sha256 = 'c'.repeat(64), e => e.now_ms = 2001,
  e => e.facet_code_hash = '0x' + '4'.repeat(64)]) {
  const wrong = structuredClone(evidence); mutate(wrong);
  assert.throws(() => validateAdoption(op, op, wrong)); negatives++;
}
for (const change of [{token: '0x' + '4'.repeat(40)}, {selector: '0xdeadbeef'}, {nonce: '1'},
  {release_proof_sha256: 'c'.repeat(64)}, {wasm_hash: '0x' + '4'.repeat(64)}]) {
  assert.throws(() => validateAdoption(op, {...op, ...change}, evidence)); negatives++;
}
console.log(JSON.stringify({status: 'PASS', positive_adoption_control: 1, negative_binding_controls: negatives,
  prototype_decoded_records_only: true, actual_ABI_decoder: false, signer: false, network: false}));
