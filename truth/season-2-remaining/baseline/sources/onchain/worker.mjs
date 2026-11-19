// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran. Read LICENSE and DATA-SOURCES.md.
// Protocol references: [RPC], [EIP1898], [NODE], [YAML] in PROVENANCE.json.
export const LIMITS = Object.freeze({
  urlBytes: 512, rpcBytes: 131072, rpcTimeoutMs: 3500, totalTimeoutMs: 12000,
  blockMax: 1000000000000, nonceMax: 1000000000000n,
  weiMax: 100000000000000000000000000000n, quantityHexDigits: 64,
  maxInflight: 4, maxRequestsPerMinutePerIsolate: 30,
});
export const CHAINS = Object.freeze({
  ethereum: { id: 1n, native: 'ETH', services: [
    { name: 'Nodies', url: 'https://ethereum-public.nodies.app' },
    { name: 'dRPC', url: 'https://eth.drpc.org' },
  ] },
  polygon: { id: 137n, native: 'POL', services: [
    { name: 'PublicNode (Allnodes)', url: 'https://polygon-bor-rpc.publicnode.com' },
    { name: 'dRPC', url: 'https://polygon.drpc.org' },
  ] },
  arbitrum: { id: 42161n, native: 'ETH', services: [
    { name: 'PublicNode (Allnodes)', url: 'https://arbitrum-one-rpc.publicnode.com' },
    { name: 'Nodies', url: 'https://arbitrum-one-public.nodies.app' },
  ] },
});
const ADDRESS = /^0x[0-9a-fA-F]{40}$/;
const HASH = /^0x[0-9a-fA-F]{64}$/;
const KEYS = new Set(['address', 'block', 'chain', 'metric']);
const HEADERS = {
  'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store',
  'content-security-policy': "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
  'x-content-type-options': 'nosniff', 'referrer-policy': 'no-referrer',
  'strict-transport-security': 'max-age=31536000',
  'permissions-policy': 'camera=(), microphone=(), geolocation=()',
  'x-license': 'LicenseRef-zkasuran-SAND-1.0',
};
export class Fault extends Error {
  constructor(code, status = 502) { super(code); this.code = code; this.status = status; }
}
export function quantity(value) {
  if (typeof value !== 'string' || value.length > LIMITS.quantityHexDigits + 2 ||
      !/^0x(?:0|[1-9a-fA-F][0-9a-fA-F]*)$/.test(value)) throw new Fault('invalid_rpc_quantity');
  return BigInt(value);
}
export function decimalUnits(value, places) {
  const scale = 10n ** BigInt(places);
  return `${value / scale}.${(value % scale).toString().padStart(places, '0')}`;
}
export function gwei(wei) {
  if (typeof wei !== 'bigint' || wei < 0n || wei > LIMITS.weiMax) throw new Fault('balance_out_of_range');
  // 1 micro-Gwei = 1000 wei. Round nearest, ties upward, using integers only.
  const micro = (wei + 500n) / 1000n;
  return { scored: decimalUnits(micro, 6), exact: decimalUnits(wei, 9), wei: wei.toString() };
}
export function parseQuery(urlText) {
  if (new TextEncoder().encode(urlText).length > LIMITS.urlBytes) throw new Fault('url_too_large', 414);
  const url = new URL(urlText);
  if (url.pathname !== '/metric') throw new Fault('not_found', 404);
  for (const key of url.searchParams.keys()) {
    if (!KEYS.has(key) || url.searchParams.getAll(key).length !== 1) throw new Fault('unknown_or_duplicate_parameter', 400);
  }
  const address = url.searchParams.get('address');
  const b = url.searchParams.get('block');
  if (!address || !ADDRESS.test(address)) throw new Fault('invalid_address', 400);
  if (b === null || !/^(0|[1-9][0-9]{0,12})$/.test(b)) throw new Fault('invalid_block', 400);
  const block = Number(b);
  if (!Number.isSafeInteger(block) || block > LIMITS.blockMax) throw new Fault('invalid_block', 400);
  const chain = url.searchParams.get('chain') ?? 'ethereum';
  if (!Object.hasOwn(CHAINS, chain)) throw new Fault('unsupported_chain', 400);
  const metric = url.searchParams.get('metric') ?? 'balance';
  if (!['balance', 'tx_count'].includes(metric)) throw new Fault('unsupported_metric', 400);
  return { address: address.toLowerCase(), block, chain, metric };
}
export async function boundedText(response, signal) {
  const length = response.headers.get('content-length');
  if (length && (!/^[0-9]+$/.test(length) || Number(length) > LIMITS.rpcBytes)) throw new Fault('rpc_response_too_large');
  if (!response.body) throw new Fault('empty_rpc_response');
  const reader = response.body.getReader();
  const chunks = []; let count = 0;
  const abort = () => { reader.cancel().catch(() => {}); };
  signal.addEventListener('abort', abort, { once: true });
  try {
    while (true) {
      signal.throwIfAborted();
      const { value, done } = await reader.read();
      signal.throwIfAborted();
      if (done) break;
      count += value.length;
      if (count > LIMITS.rpcBytes) throw new Fault('rpc_response_too_large');
      chunks.push(value);
    }
    const bytes = new Uint8Array(count); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } finally {
    signal.removeEventListener('abort', abort);
    await reader.cancel().catch(() => {});
  }
}
export async function rpcBatch(service, calls, signal, fetcher = fetch, clock = () => new Date().toISOString()) {
  const sub = new AbortController();
  const abort = () => sub.abort();
  signal.addEventListener('abort', abort, { once: true });
  if (signal.aborted) sub.abort();
  const timer = setTimeout(abort, LIMITS.rpcTimeoutMs);
  const requests = calls.map(([method, params], i) => ({ jsonrpc: '2.0', id: i + 1, method, params }));
  try {
    const response = await fetcher(service.url, {
      method: 'POST', headers: { 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify(requests), signal: sub.signal, redirect: 'error', cache: 'no-store',
    });
    if (!response.ok) throw new Fault(response.status === 429 ? 'rpc_rate_limited' : 'rpc_http_error');
    if (!(response.headers.get('content-type') ?? '').toLowerCase().includes('application/json')) throw new Fault('rpc_not_json');
    const body = JSON.parse(await boundedText(response, sub.signal));
    if (!Array.isArray(body) || body.length !== requests.length) throw new Fault('invalid_rpc_envelope');
    const seen = new Set(); const results = [];
    for (const entry of body) {
      if (!entry || entry.jsonrpc !== '2.0' || !Number.isInteger(entry.id) || entry.id < 1 ||
          entry.id > requests.length || seen.has(entry.id) || 'error' in entry || !('result' in entry)) throw new Fault('invalid_rpc_envelope');
      seen.add(entry.id); results[entry.id - 1] = entry.result;
    }
    return { results, observed_at: clock() };
  } catch (error) {
    if (sub.signal.aborted) throw new Fault('rpc_timeout');
    if (error instanceof Fault) throw error;
    throw new Fault('rpc_transport_or_parse_error');
  } finally {
    clearTimeout(timer); signal.removeEventListener('abort', abort);
  }
}
export function header(value, block) {
  if (!value || typeof value !== 'object') throw new Fault('block_unavailable');
  if (quantity(value.number) !== BigInt(block)) throw new Fault('wrong_block');
  if (!HASH.test(value.hash) || !HASH.test(value.stateRoot) || !HASH.test(value.parentHash)) throw new Fault('invalid_block_header');
  const timestamp = quantity(value.timestamp);
  if (timestamp > 253402300799n) throw new Fault('invalid_block_timestamp');
  return { number: value.number.toLowerCase(), hash: value.hash.toLowerCase(),
    state_root: value.stateRoot.toLowerCase(), parent_hash: value.parentHash.toLowerCase(),
    timestamp: timestamp.toString() };
}
export async function observe(query, fetcher = fetch, clock = () => new Date().toISOString()) {
  const chain = CHAINS[query.chain];
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), LIMITS.totalTimeoutMs);
  const tag = `0x${query.block.toString(16)}`;
  try {
    const anchors = await Promise.all(chain.services.map(async (service) => {
      const sample = await rpcBatch(service, [['eth_chainId', []], ['eth_getBlockByNumber', [tag, false]]], abort.signal, fetcher, clock);
      if (quantity(sample.results[0]) !== chain.id) throw new Fault('wrong_chain');
      return { header: header(sample.results[1], query.block), observed_at: sample.observed_at };
    }));
    if (JSON.stringify(anchors[0].header) !== JSON.stringify(anchors[1].header)) throw new Fault('header_disagreement');
    const anchor = anchors[0].header;
    const blockRef = { blockHash: anchor.hash, requireCanonical: true };
    const observations = await Promise.all(chain.services.map(async (service, i) => {
      const sample = await rpcBatch(service, [
        ['eth_getBalance', [query.address, blockRef]],
        ['eth_getTransactionCount', [query.address, blockRef]],
      ], abort.signal, fetcher, clock);
      const wei = quantity(sample.results[0]); const nonce = quantity(sample.results[1]);
      gwei(wei);
      if (nonce > LIMITS.nonceMax) throw new Fault('nonce_out_of_range');
      const recheck = await rpcBatch(service, [['eth_chainId', []], ['eth_getBlockByNumber', [tag, false]]], abort.signal, fetcher, clock);
      if (quantity(recheck.results[0]) !== chain.id || JSON.stringify(header(recheck.results[1], query.block)) !== JSON.stringify(anchor)) throw new Fault('reorg_or_chain_change');
      return { service: service.name, url: service.url, wei: wei.toString(), nonce: nonce.toString(),
        header_observed_at: anchors[i].observed_at, state_observed_at: sample.observed_at,
        rechecked_at: recheck.observed_at, state_params: [query.address, blockRef],
        methods: ['eth_getBalance', 'eth_getTransactionCount'], header: anchor };
    }));
    if (observations[0].wei !== observations[1].wei || observations[0].nonce !== observations[1].nonce) throw new Fault('state_disagreement');
    return { query, anchor, observations, balance: gwei(BigInt(observations[0].wei)), nonce: observations[0].nonce,
      served_at: clock() };
  } finally {
    clearTimeout(timer); abort.abort();
  }
}
export function serialize(sample) {
  const { query, balance } = sample;
  const result = {
    intent_id: 'ONCHAIN_METRIC_VERIFY_GENERAL', address: query.address, block: String(query.block),
    chain: query.chain, chain_id: Number(CHAINS[query.chain].id), metric: query.metric,
    balance_gwei: balance.scored, balance_gwei_exact: balance.exact, balance_wei: balance.wei,
    balance_rounding: 'nearest micro-Gwei; ties upward (1000 wei resolution)',
    tx_count: Number(sample.nonce), tx_count_semantics: 'Account nonce, not transfers or received transactions.',
    block_hash: sample.anchor.hash, state_root: sample.anchor.state_root,
    block_timestamp_unix: sample.anchor.timestamp, sources: sample.observations,
    cache: 'none', served_at: sample.served_at,
    verification_scope: 'Exact hash-pinned account state agrees at two named public RPC services. No consensus or account trie proof claimed. Backend operator independence is not attested.',
    license: 'LicenseRef-zkasuran-SAND-1.0; upstream data rights in DATA-SOURCES.md',
    summary: `${query.address} at ${query.chain} block ${query.block}: ${balance.scored} Gwei; account nonce ${sample.nonce}.`,
  };
  // Emit the exact decimal as a JSON NUMBER. Never round it through JS Number.
  return JSON.stringify(result).replace(`"balance_gwei":${JSON.stringify(balance.scored)}`, `"balance_gwei":${balance.scored}`);
}
let inflight = 0; let windowStart = 0; let requestCount = 0;
export async function handle(request, fetcher = fetch) {
  try {
    if (request.method !== 'GET') throw new Fault('method_not_allowed', 405);
    const query = parseQuery(request.url);
    const now = Date.now();
    if (now - windowStart >= 60000) { windowStart = now; requestCount = 0; }
    if (inflight >= LIMITS.maxInflight || requestCount >= LIMITS.maxRequestsPerMinutePerIsolate) throw new Fault('request_limit', 429);
    requestCount++; inflight++;
    try { return new Response(serialize(await observe(query, fetcher)), { headers: HEADERS }); }
    finally { inflight--; }
  } catch (error) {
    const fault = error instanceof Fault ? error : new Fault('request_failed');
    return new Response(JSON.stringify({ error: fault.code, code: fault.code }), { status: fault.status, headers: HEADERS });
  }
}
// Cloudflare passes env as argument two. It must not become the injected fetcher.
export default { fetch(request) { return handle(request); } };
