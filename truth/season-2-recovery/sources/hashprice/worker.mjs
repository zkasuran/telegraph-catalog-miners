// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Additional Telegraph operational grant: EVALUATION-PERMISSION.txt.
import { verifiedAnswer } from './core.mjs';
import { decodeCoinbase, verifyInclusion, requireValue } from './proof.mjs';

export const RPC = 'https://bitcoin-rpc.publicnode.com';
export const GENESIS = '000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f';
export const BOUNDS = Object.freeze({ response: 1048576, deadlineMs: 15000, url: 512, txCount: 30000 });
export const RETRY = Object.freeze({ attemptsPerMethod: 3, extraCallsPerRequest: 4, delaysMs: Object.freeze([200, 400]) });
const RPC_METHODS = ['getblockhash', 'getblock', 'getblockheader', 'getblockstats', 'getrawtransaction', 'gettxoutproof'];

class RpcFailure extends Error {
  constructor(reason, method, rpcCode = null, codeType = 'absent', httpStatus = null, envelope = {}) {
    super(reason);
    this.name = 'RpcFailure';
    this.method = RPC_METHODS.includes(method) ? method : null;
    this.rpc_error_code = Number.isSafeInteger(rpcCode) ? rpcCode : null;
    this.rpc_error_code_type = codeType;
    this.http_status = httpStatus;
    this.attempts = 1;
    this.retry_exhausted = false;
    this.envelope = envelope;
  }
}

const valueType = value => value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value;
const plainObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);

export function envelopeMetadata(data) {
  const own = key => plainObject(data) && Object.hasOwn(data, key);
  return {
    envelope_jsonrpc: own('jsonrpc') ? data.jsonrpc === null ? null
      : ['2.0', '1.0', '1.1'].includes(data.jsonrpc) ? data.jsonrpc : 'other' : 'absent',
    // Never echo a foreign ID. Only our public numeric ID or its string lookalike.
    envelope_id: own('id') && (data.id === 1 || data.id === '1') ? data.id : null,
    envelope_id_type: own('id') ? valueType(data.id) : 'absent',
    envelope_id_matches: own('id') && data.id === 1,
    result_present: own('result'), error_present: own('error'), top_level_type: valueType(data),
  };
}

function validMethodResult(method, value) {
  if (method === 'getblockhash') return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
  if (method === 'getblockheader') return typeof value === 'string' && /^[0-9a-f]{160}$/.test(value);
  if (method === 'getrawtransaction' || method === 'gettxoutproof') {
    return typeof value === 'string' && value.length > 0 && value.length % 2 === 0
      && value.length <= (method === 'getrawtransaction' ? 200000 : 8192) && /^[0-9a-f]+$/.test(value);
  }
  if (method === 'getblock') return plainObject(value) && typeof value.hash === 'string'
    && /^[0-9a-f]{64}$/.test(value.hash) && Number.isSafeInteger(value.height)
    && Array.isArray(value.tx) && value.tx.length > 0;
  if (method === 'getblockstats') return plainObject(value) && typeof value.blockhash === 'string'
    && /^[0-9a-f]{64}$/.test(value.blockhash) && Number.isSafeInteger(value.height)
    && Number.isSafeInteger(value.subsidy) && Number.isSafeInteger(value.totalfee);
  return false;
}

export function parseRpcEnvelope(data, method, httpStatus = 200) {
  const metadata = envelopeMetadata(data);
  const reject = () => { throw new RpcFailure('upstream_rpc_error', method, null,
    plainObject(data?.error) && Object.hasOwn(data.error, 'code') ? valueType(data.error.code) : 'absent', httpStatus, metadata); };
  if (!plainObject(data) || !Object.hasOwn(data, 'id') || data.id !== 1) reject();
  const modern = data.jsonrpc === '2.0';
  const legacy = !Object.hasOwn(data, 'jsonrpc');
  const resultPresent = Object.hasOwn(data, 'result');
  const errorPresent = Object.hasOwn(data, 'error');
  // [CORE-RPC-INTERFACE]: legacy has no marker and both result/error members.
  // Modern replies have exactly one of result/error. Never coerce the ID type.
  if ((!modern && !legacy) || Object.keys(data).some(key => !['jsonrpc', 'id', 'result', 'error'].includes(key))) reject();
  if (modern ? resultPresent === errorPresent : !resultPresent || !errorPresent) reject();
  const failed = errorPresent && data.error !== null;
  if (failed) {
    if (!plainObject(data.error) || !Number.isSafeInteger(data.error.code) || typeof data.error.message !== 'string'
      || Object.keys(data.error).some(key => !['code', 'message', 'data'].includes(key))
      || (legacy && data.result !== null)) reject();
    throw new RpcFailure('upstream_rpc_error', method, data.error.code, 'number', httpStatus, metadata);
  }
  if (!resultPresent || (legacy && data.error !== null) || !validMethodResult(method, data.result)) reject();
  return data.result;
}

function retryPause(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new DOMException('aborted', 'AbortError')); return; }
    const finish = () => { signal.removeEventListener('abort', abort); resolve(); };
    const timer = setTimeout(finish, ms);
    const abort = () => { clearTimeout(timer); signal.removeEventListener('abort', abort); reject(new DOMException('aborted', 'AbortError')); };
    signal.addEventListener('abort', abort, { once: true });
  });
}

function response(value, status = 200) {
  let body = JSON.stringify(value);
  // The JSON number lexeme itself carries all six decimals, including trailing zeroes.
  if (status === 200 && typeof value.hashprice_sats_per_ph_day === 'number') {
    body = body.replace(/("hashprice_sats_per_ph_day":)[0-9.]+/, '$1' + value.hashprice_sats_per_ph_day.toFixed(6));
  }
  return new Response(body, { status, headers: {
    'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store',
    'Content-Security-Policy': "default-src 'none'; frame-ancestors 'none'", 'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer', 'Access-Control-Allow-Origin': '*',
    'X-License': 'LicenseRef-zkasuran-SAND-1.0; additional Telegraph evaluation grant',
  } });
}

export function parseHeight(request) {
  requireValue(request.method === 'GET', 'method_not_allowed');
  requireValue(request.url.length <= BOUNDS.url, 'invalid_params');
  const url = new URL(request.url);
  requireValue(url.pathname === '/hashprice', 'not_found');
  const entries = [...url.searchParams];
  requireValue(entries.length === 1 && entries[0][0] === 'block_height', 'invalid_params');
  const text = entries[0][1];
  requireValue(/^(0|[1-9][0-9]{0,8})$/.test(text) && Number(text) <= 100000000, 'invalid_height');
  return Number(text);
}

async function rpcOnce(method, params, signal, transport = fetch) {
  let httpStatus = null;
  let metadata = { envelope_jsonrpc: 'unparsed', envelope_id: null, envelope_id_type: 'unparsed',
    envelope_id_matches: false, result_present: false, error_present: false, top_level_type: 'unparsed' };
  try {
  const res = await transport(RPC, { method: 'POST', redirect: 'manual', signal,
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'zkasuran-hashprice-truth/1.0' },
    body: JSON.stringify({ jsonrpc: '2.0', id: 1, method, params }) });
  httpStatus = res.status;
  if (res.status !== 200) throw new RpcFailure('upstream_http_error', method, null, 'absent', res.status);
  requireValue((res.headers.get('content-type') || '').includes('application/json'), 'upstream_content_type');
  requireValue(Number(res.headers.get('content-length') || 0) <= BOUNDS.response && res.body, 'upstream_too_large');
  const reader = res.body.getReader();
  const pieces = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.length;
      requireValue(size <= BOUNDS.response, 'upstream_too_large');
      pieces.push(value);
    }
  } finally { await reader.cancel().catch(() => {}); }
  const bytes = new Uint8Array(size);
  let pos = 0;
  for (const piece of pieces) { bytes.set(piece, pos); pos += piece.length; }
  const result = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  metadata = envelopeMetadata(result);
  return parseRpcEnvelope(result, method, res.status);
  } catch (cause) {
    if (cause instanceof RpcFailure) {
      if (cause.http_status === null) cause.http_status = httpStatus;
      if (Object.keys(cause.envelope).length === 0) cause.envelope = metadata;
      throw cause;
    }
    const reason = signal.aborted && cause.name === 'AbortError' ? 'upstream_timeout'
      : /^[a-z_]{1,64}$/.test(cause.message) ? cause.message : 'upstream_failure';
    throw new RpcFailure(reason, method, null, 'absent', httpStatus, metadata);
  }
}

async function rpc(method, params, signal, transport, state) {
  for (let attempt = 1; ; attempt++) {
    try {
      if (signal.aborted) throw new DOMException('aborted', 'AbortError');
      if (attempt > 1) state.retries++;
      state.subrequests++;
      return await rpcOnce(method, params, signal, transport);
    } catch (cause) {
      const reason = signal.aborted && cause.name === 'AbortError' ? 'upstream_timeout'
        : /^[a-z_]{1,64}$/.test(cause.message) ? cause.message : 'upstream_failure';
      const error = cause instanceof RpcFailure ? cause : new RpcFailure(reason, method);
      error.attempts = attempt;
      // [CORE-RPC-ERRORS]: -28 means still warming up. Internal, missing-data,
      // unknown gateway, HTTP and transport errors are not reclassified as transient.
      if (error.rpc_error_code !== -28 || error.rpc_error_code_type !== 'number' || signal.aborted) throw error;
      if (attempt >= RETRY.attemptsPerMethod || state.retry_slots_used >= RETRY.extraCallsPerRequest) {
        error.retry_exhausted = true;
        throw error;
      }
      state.retry_slots_used++;
      try { await retryPause(RETRY.delaysMs[attempt - 1], signal); }
      catch {
        const timeoutError = new RpcFailure('upstream_timeout', method);
        timeoutError.attempts = attempt;
        throw timeoutError;
      }
    }
  }
}

export async function verifyBundle(bundle) {
  const { height, block, stats, header, canonical_hash: hash, genesis, canonical_after: after } = bundle;
  requireValue(genesis === GENESIS, 'wrong_network');
  requireValue(Number.isSafeInteger(height) && height >= 0 && height <= 100000000, 'invalid_height');
  requireValue(typeof hash === 'string' && /^[0-9a-f]{64}$/.test(hash) && hash === after, 'canonical_hash_changed');
  requireValue(block && block.hash === hash && block.height === height && Number.isSafeInteger(block.confirmations)
    && block.confirmations > 0, 'not_canonical');
  requireValue(Array.isArray(block.tx) && block.tx.length > 0 && block.tx.length <= BOUNDS.txCount
    && block.nTx === block.tx.length && new Set(block.tx).size === block.tx.length
    && block.tx.every(x => typeof x === 'string' && /^[0-9a-f]{64}$/.test(x)), 'invalid_transaction_list');
  requireValue(stats && stats.blockhash === hash && stats.height === height, 'stats_identity_mismatch');
  requireValue(typeof block.bits === 'string' && /^[0-9a-f]{8}$/.test(block.bits), 'invalid_bits');
  const coinbase = await decodeCoinbase(bundle.coinbase_hex, height);
  requireValue(coinbase.txid === block.tx[0], 'coinbase_txid_mismatch');
  await verifyInclusion(bundle.proof_hex, header, coinbase.txid, block.nTx);
  const halvings = Math.floor(height / 210000);
  const subsidy = Number(halvings >= 64 ? 0n : 5000000000n >> BigInt(halvings));
  requireValue(stats.subsidy === subsidy && Number.isSafeInteger(stats.totalfee) && stats.totalfee >= 0, 'stats_reward_invalid');
  // An underclaimed coinbase is unavailable rather than a silent definition change.
  requireValue(coinbase.reward === subsidy + stats.totalfee, 'underclaimed_or_inconsistent_reward');
  const normalized = { id: hash, height, version: block.version, timestamp: block.time, nonce: block.nonce,
    bits: parseInt(block.bits, 16), difficulty: block.difficulty, merkle_root: block.merkleroot,
    previousblockhash: block.previousblockhash, stale: 0,
    extras: { header, reward: coinbase.reward, totalFees: stats.totalfee } };
  const value = await verifiedAnswer(normalized, height, RPC);
  return { ...value,
    attribution: 'Bitcoin mainnet consensus facts via Allnodes PublicNode. Free evaluation use under [PUBLICNODE].',
    coinbase_txid: coinbase.txid, coinbase_outputs: coinbase.outputs,
    verification: { block_header_sha256d: true, proof_of_work_target: true, difficulty_from_bits: true,
      mainnet_genesis: true, canonical_height_before_and_after: true, coinbase_txid: true,
      coinbase_merkle_inclusion: true, coinbase_bip34_height: height >= 227931,
      reward_subsidy_plus_recorded_fees: true, full_chain_validation: false, independent_runtime_provider: false },
    citations: ['T1', 'T3', 'T4', 'D1', 'CORE-POW', 'CORE-TARGET', 'CORE-REWARD', 'CORE-TX', 'CORE-MERKLE', 'PUBLICNODE'],
    license: 'LicenseRef-zkasuran-SAND-1.0; additional Telegraph evaluation grant in EVALUATION-PERMISSION.txt',
  };
}

export function createWorker(transport = fetch) {
  // No result memo: every response rechecks canonical height. One in-flight request per isolate.
  let busy = false;
  return { async fetch(request) {
    if (new URL(request.url).pathname === '/health' && request.method === 'GET') {
      return response({ status: 'ready', intent: 'MINING_HASHPRICE_VERIFY_GENERAL', upstream_checked: false });
    }
    let height;
    try { height = parseHeight(request); }
    catch (error) { return response({ status: 'unavailable', error: { code: error.message, message: 'Use GET /hashprice with exactly one plain integer block_height query parameter.' } }, error.message === 'method_not_allowed' ? 405 : error.message === 'not_found' ? 404 : 400); }
    if (busy) return response({ status: 'unavailable', error: { code: 'busy', message: 'One verification is already in progress in this isolate.' } }, 429);
    busy = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), BOUNDS.deadlineMs);
    const state = { subrequests: 0, retries: 0, retry_slots_used: 0 };
    try {
      const call = (method, params) => rpc(method, params, controller.signal, transport, state);
      const genesis = await call('getblockhash', [0]);
      requireValue(genesis === GENESIS, 'wrong_network');
      const hash = await call('getblockhash', [height]);
      requireValue(typeof hash === 'string' && /^[0-9a-f]{64}$/.test(hash), 'invalid_canonical_hash');
      const block = await call('getblock', [hash, 1]);
      requireValue(block && Array.isArray(block.tx) && block.tx.length > 0, 'missing_coinbase');
      const [header, stats, coinbase, proof] = await Promise.all([
        call('getblockheader', [hash, false]),
        call('getblockstats', [hash, ['height', 'blockhash', 'subsidy', 'totalfee']]),
        call('getrawtransaction', [block.tx[0], false, hash]),
        call('gettxoutproof', [[block.tx[0]], hash]),
      ]);
      const value = await verifyBundle({ height, genesis, canonical_hash: hash, block, header, stats,
        coinbase_hex: coinbase, proof_hex: proof, canonical_after: hash });
      const after = await call('getblockhash', [height]);
      requireValue(after === hash, 'canonical_hash_changed');
      return response({ ...value, upstream: state });
    } catch (error) {
      controller.abort();
      const code = controller.signal.aborted && error.name === 'AbortError' ? 'upstream_timeout'
        : /^[a-z_]{1,64}$/.test(error.message) ? error.message : 'upstream_failure';
      const metadata = error instanceof RpcFailure ? { method: error.method, rpc_error_code: error.rpc_error_code,
        rpc_error_code_type: error.rpc_error_code_type, http_status: error.http_status,
        attempts: error.attempts, retry_exhausted: error.retry_exhausted, ...error.envelope } : {};
      return response({ status: 'unavailable', block_height: height, error: { code, message: 'The requested block could not be verified. No reference value is returned.', ...metadata }, upstream: state }, 503);
    } finally { clearTimeout(timeout); busy = false; }
  } };
}
export default createWorker();
