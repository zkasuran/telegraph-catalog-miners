// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
// Prototype pure validation for already decoded records. No signing or I/O.
import {fingerprint} from './planner.mjs';

const address = value => {
  if (typeof value !== 'string' || !/^0x[a-fA-F0-9]{40}$/.test(value)) throw new Error('invalid address');
  return value.toLowerCase();
};
const digest = value => {
  if (typeof value !== 'string' || !/^0x[a-fA-F0-9]{64}$/.test(value)) throw new Error('invalid digest');
  return value.toLowerCase();
};
const natural = value => {
  if (typeof value !== 'string' || !/^(0|[1-9][0-9]{0,77})$/.test(value)) throw new Error('decimal integer string required');
  if (BigInt(value) >= 1n << 256n) throw new Error('uint256 overflow');
  return value;
};
function same(a, b, label) {
  if (fingerprint(a) !== fingerprint(b)) throw new Error('binding mismatch: ' + label);
}

export function normalizeOperation(op) {
  if (op.chain_id !== 11155111 || op.campaign !== 'zkcmp-s2-20261118-1' ||
      typeof op.family !== 'string' || !/^[A-Z_]+$/.test(op.family)) throw new Error('wrong deployment scope');
  if (address(op.diamond) !== '0x72a70e949362670bf8bcead24efc5b8f950aa08b') throw new Error('wrong Diamond');
  if (address(op.token) !== '0xcc194e1a843b658526d0d5c431f670028582ae17') throw new Error('wrong test USDC');
  if (op.selector !== '0xc4feb8ed' || !/^0xc4feb8ed[0-9a-f]*$/.test(op.calldata) || op.calldata.length % 2 !== 0) throw new Error('wrong registration calldata');
  const result = {...op, diamond: address(op.diamond), token: address(op.token), owner: address(op.owner),
    fee_address: address(op.fee_address), facet: address(op.facet), facet_code_hash: digest(op.facet_code_hash),
    family_id: digest(op.family_id), wasm_hash: digest(op.wasm_hash), signed_hash: digest(op.signed_hash),
    nonce: natural(op.nonce), value: natural(op.value), bond: natural(op.bond)};
  for (const key of ['binary_url', 'runnable_url']) {
    const url = new URL(op[key]);
    if (url.protocol !== 'https:' || url.origin !== 'https://telegraph-fin.margyn.workers.dev' || url.search || url.hash) throw new Error('unexpected public URL');
  }
  if (op.binary_url !== `https://telegraph-fin.margyn.workers.dev/_zkcmp_s2_20261118/binary/${op.family}/${result.wasm_hash.slice(2)}.wasm` ||
      op.runnable_url !== `https://telegraph-fin.margyn.workers.dev/_zkcmp_s2_20261118/runnable/${op.family}`) throw new Error('artifact URL binding mismatch');
  if (op.value !== '0' || !/^[a-f0-9]{64}$/.test(op.release_proof_sha256) || !/^[a-f0-9]{64}$/.test(op.deployment_proof_sha256)) throw new Error('missing release/deployment binding');
  return result;
}

export function operationKey(op) {
  const value = normalizeOperation(op);
  return [value.chain_id, value.diamond, value.owner, value.family, value.wasm_hash].join(':');
}

export function validateAdoption(intent, stored, evidence) {
  const op = normalizeOperation(intent);
  same(op, normalizeOperation(stored), 'stored operation');
  same(evidence.deployment_proof_sha256, op.deployment_proof_sha256, 'deployment proof');
  if (!Number.isSafeInteger(evidence.now_ms) || !Number.isSafeInteger(evidence.proof_valid_until_ms) ||
      evidence.now_ms > evidence.proof_valid_until_ms) throw new Error('stale proof');
  same({facet: address(evidence.facet), facet_code_hash: digest(evidence.facet_code_hash)},
    {facet: op.facet, facet_code_hash: op.facet_code_hash}, 'facet');
  if (evidence.broadcast_state !== 'reconciled' || evidence.abi_bound_to_current_facet !== true) throw new Error('unresolved broadcast or ABI');
  const tx = evidence.transaction, receipt = evidence.receipt;
  same({chain_id: tx.chain_id, hash: digest(tx.hash), from: address(tx.from), to: address(tx.to), nonce: natural(tx.nonce), value: natural(tx.value), data: tx.data},
    {chain_id: op.chain_id, hash: op.signed_hash, from: op.owner, to: op.diamond, nonce: op.nonce, value: op.value, data: op.calldata}, 'transaction');
  if (receipt.status !== 1 || receipt.safe !== true || receipt.canonical !== true ||
      digest(receipt.transaction_hash) !== op.signed_hash || digest(receipt.block_hash) !== digest(evidence.safe_block_hash) ||
      natural(receipt.block_number) !== natural(evidence.safe_block_number) || address(receipt.from) !== op.owner || address(receipt.to) !== op.diamond) throw new Error('receipt binding');
  if (!Array.isArray(evidence.events) || evidence.events.length !== 1) throw new Error('exactly one decoded registration event required');
  const expected = {emitter: op.diamond, author: op.owner, family_id: op.family_id, wasm_hash: op.wasm_hash,
    binary_url: op.binary_url, family: op.family, fee_address: op.fee_address, runnable_url: op.runnable_url};
  const event = evidence.events[0];
  const {id, ...eventIdentity} = event;
  same(eventIdentity, expected, 'event');
  const getter = evidence.getter;
  same(getter, {...expected, id: natural(id), active: true, bond: op.bond,
    block_hash: evidence.safe_block_hash, block_number: evidence.safe_block_number}, 'safe getter');
  return {state: 'prepared_transaction_confirmed', operation_key: operationKey(op), registration_id: natural(event.id)};
}
