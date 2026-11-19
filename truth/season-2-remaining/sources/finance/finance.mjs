// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Own read-only calculations. No issuer API content or upstream software is embedded.
export const LIMITS = Object.freeze({ queryBytes: 2048, responseBytes: 1048576,
  timeoutMs: 12000, totalMs: 45000, attemptMs: 4000, maxAttempts: 3, maxSubrequests: 48,
  retryBaseMs: 200, minAttemptMs: 100, jsonDepth: 64, jsonNodes: 50000,
  oracleAge: 172800, finalizedAge: 7200 });
export const RPC = 'https://ethereum-public.nodies.app';
export const CONTRACTS = Object.freeze({
  reserve: '0xa81FE04086865e63E12dD3776978E49DEEa2ea4e',
  pool: '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
  usdc: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
  comet: '0xc3d688B66703497DAA19211EEdff47f25384cdc3',
});
const YEAR = 31536000n;
const PRECISION = 10n ** 36n;
const PREFIX = '/truth/v2/';

export class Refusal extends Error {
  constructor(code, status = 422, details = {}) { super(code); this.code = code; this.status = status; this.details = details; }
}

function boundedString(value) {
  if (typeof value !== 'string' || value.length > 256 || /[\u0000-\u001f]/.test(value)) throw new Refusal('invalid_input');
  return value.trim();
}

export function abiWords(value, min = 1, max = min) {
  if (typeof value !== 'string' || value.length > 2 + max * 64 || !/^0x(?:[0-9a-fA-F]{64})+$/.test(value)) throw new Refusal('invalid_abi', 503);
  const words = value.slice(2).match(/.{64}/g).map(x => BigInt('0x' + x));
  if (words.length < min || words.length > max) throw new Refusal('invalid_abi', 503);
  return words;
}

export function fixed6(value) {
  if (value < 0n) return '-' + fixed6(-value);
  return `${value / 1000000n}.${String(value % 1000000n).padStart(6, '0')}`;
}

export function apyBps(rate, denominator) {
  if (typeof rate !== 'bigint' || rate < 0n || denominator <= 0n || rate * 10000000n > denominator) throw new Refusal('rate_out_of_range', 503);
  let base = PRECISION + rate * PRECISION / denominator;
  let power = YEAR;
  let result = PRECISION;
  while (power > 0n) {
    if (power & 1n) result = result * base / PRECISION;
    power >>= 1n;
    if (power) base = base * base / PRECISION;
  }
  // Six decimal basis points, rounded to nearest with half upward.
  const answer = ((result - PRECISION) * 10000000000n + PRECISION / 2n) / PRECISION;
  if (answer > 100000000000n) throw new Refusal('yield_out_of_range', 503);
  return fixed6(answer);
}

export function reserveFromRound(words, decimals, timestamp) {
  if (words.length !== 5 || decimals !== 8n || words[0] === 0n || words[1] === 0n || words[1] >= 2n**255n ||
      words[3] === 0n || words[2] > words[3] || words[4] < words[0] || words[3] > timestamp ||
      timestamp - words[3] > BigInt(LIMITS.oracleAge)) throw new Refusal('invalid_or_stale_attestation', 503);
  const micro = (words[1] + 50n) / 100n;
  if (micro < 1n || micro > 1000000000000000000n) throw new Refusal('reserve_out_of_range', 503);
  return micro.toString();
}

export function paymentContract(bin) {
  if (!/^[1-9]\d{5,7}$/.test(bin) || Number(bin) < 100000 || Number(bin) > 99999999) throw new Refusal('bin_outside_current_range');
  // BIN prefixes do not include the account number or its check digit. No verdict can establish a full PAN checksum.
  throw new Refusal('owner_definition_required');
}

export function gridQuote(rows, node, start) {
  boundedString(node); boundedString(start);
  if (!Array.isArray(rows) || rows.length > 100000) throw new Refusal('archive_out_of_range', 503);
  const epoch = Date.parse(start);
  if (!Number.isFinite(epoch) || !/T\d\d:\d\d:\d\d(?:Z|[+-]\d\d:\d\d)$/.test(start) || epoch % 3600000) throw new Refusal('invalid_dam_hour');
  const calendar = /^(\d{4})-(\d{2})-(\d{2})T/.exec(start);
  if (!calendar || new Date(Date.UTC(Number(calendar[1]), Number(calendar[2]) - 1, Number(calendar[3]))).toISOString().slice(0, 10) !== start.slice(0, 10)) throw new Refusal('invalid_dam_date');
  const matches = rows.filter(x => x.node_or_zone === node && Date.parse(x.interval_start) === epoch && x.market === 'DAM' && x.currency === 'USD');
  if (matches.length !== 1) throw new Refusal(matches.length ? 'duplicate_dam_quote' : 'dam_quote_unavailable', 503);
  const row = matches[0];
  if (Date.parse(row.interval_end) - epoch !== 3600000 || typeof row.lmp_usd_per_mwh !== 'string' ||
      !/^-?\d{1,5}(?:\.\d{1,6})?$/.test(row.lmp_usd_per_mwh) || !Number.isFinite(Number(row.lmp_usd_per_mwh)) ||
      Number(row.lmp_usd_per_mwh) < -2000 || Number(row.lmp_usd_per_mwh) > 20000) throw new Refusal('invalid_dam_quote', 503);
  return { lmp_usd_per_mwh: row.lmp_usd_per_mwh, node_or_zone: node, interval_start: new Date(epoch).toISOString().replace('.000Z', 'Z'),
    market: 'DAM', currency: 'USD', source: row.source, source_precision_decimals: row.source_precision_decimals,
    archive_rounding_not_original_six_decimal_precision: row.archive_rounding_not_original_six_decimal_precision };
}

const own = (x, key) => Object.prototype.hasOwnProperty.call(x, key);
const object = x => x !== null && typeof x === 'object' && !Array.isArray(x);
const quantity = x => typeof x === 'string' && /^0x(?:0|[1-9a-f][0-9a-f]*)$/.test(x);
const hash = x => typeof x === 'string' && /^0x[0-9a-f]{64}$/i.test(x);
const METHODS = new Set(['eth_chainId', 'eth_getBlockByNumber', 'eth_call']);
const TRANSIENT_HTTP = new Set([500, 502, 503, 504, 520, 521, 522, 523, 524]);

// JSON.parse alone would silently keep the last duplicate id or round a fractional id to 1.
export function strictJson(text) {
  let i = 0, nodes = 0;
  const tokens = new Map();
  const bad = () => { throw new Refusal('rpc_error', 503, { classification: 'invalid_json' }); };
  const ws = () => { while (/^[\x20\x09\x0a\x0d]$/.test(text[i] || '')) i++; };
  function string() {
    const start = i++;
    while (i < text.length) {
      const c = text[i++];
      if (c === '\\') { i++; continue; }
      if (c === '"') { try { return JSON.parse(text.slice(start, i)); } catch { bad(); } }
    }
    bad();
  }
  function value(depth, path) {
    ws();
    if (++nodes > LIMITS.jsonNodes || depth > LIMITS.jsonDepth) bad();
    const start = i;
    if (text[i] === '{' || text[i] === '[') {
      const isObject = text[i++] === '{', end = isObject ? '}' : ']';
      const keys = new Set();
      ws();
      if (text[i] === end) { i++; return; }
      for (;;) {
        let key = '';
        if (isObject) {
          ws(); if (text[i] !== '"') bad(); key = string();
          if (keys.has(key)) throw new Refusal('rpc_error', 503, { classification: 'duplicate_json_key' });
          keys.add(key); ws(); if (text[i++] !== ':') bad();
        }
        value(depth + 1, isObject ? path + '/' + key : path + '/[]');
        ws(); if (text[i] === end) { i++; break; }
        if (text[i++] !== ',') bad();
      }
    } else if (text[i] === '"') string();
    else {
      while (i < text.length && !/[\x20\x09\x0a\x0d,\]}]/.test(text[i])) i++;
      if (i === start) bad();
      if (path === '/id' || path === '/error/code') tokens.set(path, text.slice(start, i));
    }
  }
  value(0, ''); ws(); if (i !== text.length) bad();
  let parsed;
  try { parsed = JSON.parse(text); } catch { bad(); }
  return { parsed, tokens };
}

function envelope(parsed, tokens) {
  const reject = classification => { throw new Refusal('rpc_error', 503, { classification }); };
  if (!object(parsed) || parsed.jsonrpc !== '2.0' || Object.keys(parsed).some(k => !['jsonrpc', 'id', 'result', 'error'].includes(k))) reject('invalid_rpc_envelope');
  if (parsed.id !== 1 || tokens.get('/id') !== '1') reject('invalid_rpc_id');
  if (own(parsed, 'result') === own(parsed, 'error')) reject('ambiguous_rpc_envelope');
  if (own(parsed, 'error')) {
    const error = parsed.error, token = tokens.get('/error/code');
    if (!object(error) || typeof error.message !== 'string' || !Number.isSafeInteger(error.code) ||
        Math.abs(error.code) > 2147483647 || !/^-?(?:0|[1-9][0-9]*)$/.test(token || '') ||
        Object.keys(error).some(k => !['code', 'message', 'data'].includes(k))) reject('invalid_rpc_error');
    const classification = ({ '-32001': 'rpc_resource_not_found', '-32000': 'rpc_invalid_input_or_state',
      '-32602': 'rpc_invalid_params', '-32601': 'rpc_method_not_found', '-32603': 'rpc_internal_error',
      '-32600': 'rpc_invalid_request', '-32700': 'rpc_parse_error' })[error.code] || 'rpc_error_unknown';
    throw new Refusal('rpc_error', 503, { classification, rpc_code: error.code });
  }
  if (parsed.result === null) reject('rpc_result_missing');
  return parsed.result;
}

export function retryAfter(value, now) {
  if (value === null) return { state: 'missing', ms: null };
  if (/^[0-9]{1,10}$/.test(value)) return { state: 'delay', ms: Number(value) * 1000 };
  if (/^(Mon|Tue|Wed|Thu|Fri|Sat|Sun), [0-9]{2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT$/.test(value)) {
    const stamp = Date.parse(value);
    if (Number.isFinite(stamp) && new Date(stamp).toUTCString() === value) return { state: 'date', ms: Math.max(0, stamp - now) };
  }
  return { state: 'invalid', ms: null };
}

function headerClasses(headers, now) {
  const type = (headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
  return { content_type: type === 'application/json' || /^application\/[a-z0-9.+-]+\+json$/.test(type) ? 'json' :
      type === 'text/html' ? 'html' : type === 'text/plain' ? 'text' : type ? 'other' : 'missing',
    server_class: (headers.get('server') || '').toLowerCase() === 'cloudflare' ? 'cloudflare' : headers.has('server') ? 'other' : 'missing',
    waf_challenge: headers.get('cf-mitigated') === 'challenge', auth_challenge: headers.has('www-authenticate'),
    retry_after: retryAfter(headers.get('retry-after'), now) };
}

function transport(error) {
  if (error instanceof Refusal) return error;
  const code = error?.cause?.code || error?.code;
  const classification = ({ ECONNRESET: 'connection_reset', ECONNREFUSED: 'connection_refused', EPIPE: 'connection_lost',
    EAI_AGAIN: 'dns_temporary', ETIMEDOUT: 'transport_timeout', UND_ERR_CONNECT_TIMEOUT: 'transport_timeout',
    UND_ERR_SOCKET: 'connection_lost', UND_ERR_RES_CONTENT_LENGTH_MISMATCH: 'transport_truncated' })[code];
  if (classification) return new Refusal('upstream_unavailable', 503, { classification, transient: true });
  // This exact local workerd exception is not a remote RPC message.
  if (error?.message === 'Network connection lost.' || error?.message === 'Network connection lost')
    return new Refusal('upstream_unavailable', 503, { classification: 'connection_lost', transient: true });
  return new Refusal('upstream_unavailable', 503, { classification: 'transport_unknown' });
}

// Options only lower limits or inject clocks/sleep for tests. The HTTP API never exposes them.
export function rpcReadClient(fetcher, options = {}) {
  const lower = (value, maximum) => Number.isFinite(value) && value > 0 ? Math.min(value, maximum) : maximum;
  const now = options.now || Date.now;
  const started = now(), deadline = started + lower(options.totalMs, LIMITS.totalMs);
  const readMs = lower(options.readMs, LIMITS.timeoutMs), attemptMs = lower(options.attemptMs, LIMITS.attemptMs);
  const scope = new AbortController(), trace = [];
  let calls = 0, failure = null, closed = false;
  const fail = (code, classification) => new Refusal(code, 503, { classification });
  const totalTimer = setTimeout(() => scope.abort(fail('deadline_exceeded', 'request_deadline')), deadline - started);
  async function bounded(operation, ms, controller, expired, readSignal) {
    if (scope.signal.aborted) throw scope.signal.reason;
    if (readSignal?.aborted) throw readSignal.reason;
    let timer, abort;
    const timeout = new Promise((_, reject) => { timer = setTimeout(() => { controller.abort(expired); reject(expired); }, ms); });
    const stopped = new Promise((_, reject) => {
      abort = () => { const reason = scope.signal.aborted ? scope.signal.reason : readSignal.reason; controller.abort(reason); reject(reason); };
      scope.signal.addEventListener('abort', abort, { once: true });
      readSignal?.addEventListener('abort', abort, { once: true });
    });
    try { return await Promise.race([Promise.resolve().then(operation), timeout, stopped]); }
    finally { clearTimeout(timer); scope.signal.removeEventListener('abort', abort); readSignal?.removeEventListener('abort', abort); }
  }
  async function read(method, params, validate = x => x) {
    if (!METHODS.has(method)) throw fail('rpc_error', 'rpc_method_not_allowed');
    const readDeadline = Math.min(deadline, now() + readMs);
    const body = JSON.stringify({ jsonrpc: '2.0', id: 1, method, params });
    const readScope = new AbortController();
    const readTimer = setTimeout(() => readScope.abort(fail('deadline_exceeded', 'read_deadline')), Math.max(0, readDeadline - now()));
    try {
    for (let attempt = 1; attempt <= LIMITS.maxAttempts; attempt++) {
      if (closed || scope.signal.aborted) throw scope.signal.reason || fail('deadline_exceeded', 'request_closed');
      if (readScope.signal.aborted) throw readScope.signal.reason;
      const remaining = readDeadline - now();
      if (remaining <= 0) throw fail('deadline_exceeded', now() >= deadline ? 'request_deadline' : 'read_deadline');
      if (calls >= LIMITS.maxSubrequests) throw fail('subrequest_budget_exceeded', 'subrequest_budget');
      calls++;
      const row = { method, attempt, classification: 'in_flight', http_status: null };
      trace.push(row);
      const controller = new AbortController();
      const expiry = new Refusal('upstream_unavailable', 503, { classification: 'transport_timeout', transient: true });
      try {
        const answer = await bounded(async () => {
          let response, reader;
          try {
            response = await fetcher(RPC, { method: 'POST', headers: { 'content-type': 'application/json' },
              body, redirect: 'manual', signal: controller.signal });
            row.http_status = response.status;
            const headers = headerClasses(response.headers, now());
            Object.assign(row, headers);
            if (response.status >= 300 && response.status < 400) throw fail('upstream_redirect_refused', 'http_redirect');
            if (headers.waf_challenge || headers.auth_challenge) throw fail('upstream_status_' + response.status, 'http_access_challenge');
            const success = response.status === 200;
            // Opaque gateway bodies are never RPC values. JSON RPC errors, even on 5xx/429, never retry.
            let decoded;
            if (success || headers.content_type === 'json') {
              if (headers.content_type !== 'json') throw fail('upstream_not_json', 'http_not_json');
              if (!response.body) throw fail('rpc_error', 'empty_body');
              reader = response.body.getReader();
              const chunks = []; let bytes = 0;
              for (;;) {
                const part = await reader.read();
                if (part.done) break;
                bytes += part.value.byteLength;
                if (bytes > LIMITS.responseBytes) throw fail('upstream_too_large', 'response_limit');
                chunks.push(part.value);
              }
              const data = new Uint8Array(bytes); let offset = 0;
              for (const chunk of chunks) { data.set(chunk, offset); offset += chunk.byteLength; }
              let text;
              try { text = new TextDecoder('utf-8', { fatal: true }).decode(data); } catch { throw fail('rpc_error', 'invalid_utf8'); }
              decoded = strictJson(text);
              if (success || (object(decoded.parsed) && (own(decoded.parsed, 'jsonrpc') || own(decoded.parsed, 'id') || own(decoded.parsed, 'result') || own(decoded.parsed, 'error')))) {
                const value = envelope(decoded.parsed, decoded.tokens);
                if (success) {
                  try { return validate(value); }
                  catch (error) { throw error instanceof Refusal ? error : fail('rpc_error', 'invalid_result_schema'); }
                }
                throw fail('upstream_status_' + response.status, 'rpc_http_status_conflict');
              }
            }
            if (!success) throw new Refusal('upstream_status_' + response.status, 503,
              { classification: response.status === 429 ? 'http_rate_limited' : 'http_error',
                transient: response.status === 429 || TRANSIENT_HTTP.has(response.status) });
            throw fail('rpc_error', 'invalid_rpc_envelope');
          } finally {
            if (reader) void reader.cancel().catch(() => {});
            else if (response?.body) void response.body.cancel().catch(() => {});
          }
        }, Math.min(attemptMs, remaining), controller, expiry, readScope.signal);
        if (scope.signal.aborted || readScope.signal.aborted || now() >= readDeadline) throw fail('deadline_exceeded', now() >= deadline ? 'request_deadline' : 'read_deadline');
        row.classification = 'success';
        return answer;
      } catch (caught) {
        controller.abort();
        const error = transport(caught);
        Object.assign(row, error.details); delete row.transient;
        const advice = row.retry_after || { state: 'missing', ms: null };
        let retry = error.details.transient === true && attempt < LIMITS.maxAttempts && !scope.signal.aborted && !readScope.signal.aborted && !closed;
        if (row.http_status === 429 && !['delay', 'date'].includes(advice.state)) { retry = false; row.retry_policy = 'retry_after_required'; }
        if (advice.state === 'invalid') { retry = false; row.retry_policy = 'invalid_retry_after'; }
        const delay = Math.max(LIMITS.retryBaseMs * attempt, advice.ms || 0);
        if (retry && (calls >= LIMITS.maxSubrequests || now() + delay + LIMITS.minAttemptMs >= readDeadline)) {
          retry = false; row.retry_policy = 'budget_insufficient';
        }
        row.retried = retry;
        if (!retry) { failure = { ...row }; throw error; }
        row.retry_delay_ms = delay;
        const sleepController = new AbortController();
        try { await bounded(() => options.sleep ? options.sleep(delay) : new Promise(resolve => {
          const timer = setTimeout(resolve, delay);
          sleepController.signal.addEventListener('abort', () => { clearTimeout(timer); }, { once: true });
        }), Math.max(1, readDeadline - now()), sleepController, fail('deadline_exceeded', 'retry_deadline'), readScope.signal); }
        catch (stopped) { const reason = transport(stopped); failure = { ...row, ...reason.details }; throw reason; }
      }
    }
    throw fail('rpc_error', 'attempt_budget');
    } finally { clearTimeout(readTimer); }
  }
  return { read,
    summary: () => ({ subrequests: calls, retries: trace.filter(r => r.attempt > 1).length,
      elapsed_ms: Math.max(0, Math.round(now() - started)), failure,
      trace: trace.map(r => ({ ...r })) }),
    close: () => { closed = true; clearTimeout(totalTimer); if (!scope.signal.aborted) scope.abort(fail('deadline_exceeded', 'request_closed')); } };
}

function blockHeader(value, requested) {
  if (!object(value) || !hash(value.hash) || !quantity(value.number) || !quantity(value.timestamp) ||
      value.number.length > 18 || value.timestamp.length > 18) throw new Refusal('invalid_block_response', 503, { classification: 'invalid_block_header' });
  if (quantity(requested) && value.number !== requested) throw new Refusal('invalid_block_response', 503, { classification: 'wrong_block_number' });
  return { ...value, hash: value.hash.toLowerCase() };
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: {
    'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store',
    'x-content-type-options': 'nosniff', 'content-security-policy': "default-src 'none'; frame-ancestors 'none'",
    'x-frame-options': 'DENY', 'referrer-policy': 'no-referrer' } });
}

export async function handleFinance(request, fetcher = fetch, archive = []) {
  const url = new URL(request.url);
  if (![PREFIX + 'finance-reserve', PREFIX + 'finance-yield', PREFIX + 'finance-payment', PREFIX + 'finance-dam'].includes(url.pathname)) return null;
  let client;
  try {
    if (request.method !== 'GET') throw new Refusal('method_not_allowed', 405);
    if (new TextEncoder().encode(url.search).length > LIMITS.queryBytes) throw new Refusal('query_too_large', 413);
    const allowed = url.pathname.endsWith('reserve') ? ['asset', 'block'] : url.pathname.endsWith('yield') ? ['project', 'symbol', 'chain', 'block'] :
      url.pathname.endsWith('payment') ? ['bin'] : ['node_or_zone', 'interval_start'];
    for (const key of url.searchParams.keys()) if (!allowed.includes(key) || url.searchParams.getAll(key).length !== 1) throw new Refusal('unexpected_or_duplicate_parameter');
    const value = name => boundedString(url.searchParams.get(name));
    if (url.pathname.endsWith('payment')) paymentContract(value('bin'));
    if (url.pathname.endsWith('dam')) return json(gridQuote(archive, value('node_or_zone'), value('interval_start')));
    const isReserve = url.pathname.endsWith('reserve');
    let project;
    if (isReserve) {
      const asset = value('asset').toUpperCase();
      if (asset !== 'WBTC' && asset !== '0X2260FAC5E5542A773AA44FBCFEDF7C193BC2C599') throw new Refusal('unsupported_asset');
    } else {
      project = value('project').toLowerCase();
      if (!['aave-v3', 'compound-v3'].includes(project) || value('symbol').toUpperCase() !== 'USDC' || value('chain').toLowerCase() !== 'ethereum') throw new Refusal('unsupported_market');
    }
    const supplied = url.searchParams.get('block');
    if (supplied !== null && (!/^[1-9]\d{0,8}$/.test(supplied) || Number(supplied) > 100000000)) throw new Refusal('invalid_block');
    const tag = supplied === null ? 'finalized' : '0x' + BigInt(supplied).toString(16);
    client = rpcReadClient(fetcher);
    await client.read('eth_chainId', [], value => {
      if (value !== '0x1') throw new Refusal('wrong_chain', 503, { classification: 'wrong_chain' });
      return value;
    });
    const head = await client.read('eth_getBlockByNumber', [tag, false], value => blockHeader(value, tag));
    const timestamp = BigInt(head.timestamp);
    const now = BigInt(Math.floor(Date.now() / 1000));
    if (timestamp > now || (supplied === null && now - timestamp > BigInt(LIMITS.finalizedAge))) throw new Refusal('stale_block', 503);
    const point = Object.freeze({ blockHash: head.hash, requireCanonical: true });
    // EIP-1898 pins every state read to this identity, never a changing number or latest.
    const call = (to, data, words = 1) => client.read('eth_call', [{ to, data }, point], value => abiWords(value, words));
    let answer;
    if (isReserve) {
      const [words, decimals] = await Promise.all([call(CONTRACTS.reserve, '0xfeaf968c', 5), call(CONTRACTS.reserve, '0x313ce567')]);
      answer = { attested_reserve: reserveFromRound(words, decimals[0], timestamp), asset: 'WBTC',
        block: BigInt(head.number).toString(), source: 'Ethereum mainnet WBTC PoR oracle ' + CONTRACTS.reserve,
        round: words[0].toString(), attestation_updated_at: words[3].toString(),
        scope: 'Oracle-attested BTC reserves. No liability completeness, bank attestation or independent custodian audit is claimed.' };
    } else if (project === 'aave-v3') {
      const reserve = await call(CONTRACTS.pool, '0x35ea6a75' + CONTRACTS.usdc.slice(2).toLowerCase().padStart(64, '0'), 15);
      answer = { supply_apy: apyBps(reserve[2], 10n**27n * YEAR), annual_rate_ray: reserve[2].toString() };
    } else {
      const base = (await call(CONTRACTS.comet, '0xc55dae63'))[0];
      if (base !== BigInt(CONTRACTS.usdc)) throw new Refusal('wrong_base_asset', 503);
      const utilization = (await call(CONTRACTS.comet, '0x7eb71131'))[0];
      if (utilization > 10n**18n) throw new Refusal('invalid_utilization', 503);
      const rate = (await call(CONTRACTS.comet, '0xd955759d' + utilization.toString(16).padStart(64, '0')))[0];
      answer = { supply_apy: apyBps(rate, 10n**18n), per_second_rate_wad: rate.toString() };
    }
    if (!isReserve) Object.assign(answer, { project, symbol: 'USDC', chain: 'Ethereum', unit: 'basis points',
      convention: 'Constant current base supply rate compounded each second for 365 days. Incentive rewards excluded. Not a realised-yield forecast.' });
    const verifyHead = await client.read('eth_getBlockByNumber', [head.number, false], value => blockHeader(value, head.number));
    if (verifyHead.hash !== head.hash) throw new Refusal('block_changed', 503);
    return json({ ...answer, chain_id: 1, block_number: BigInt(head.number).toString(), block_hash: head.hash,
      block_timestamp: timestamp.toString(), provenance: 'Own read-only public-chain observations and calculations via Nodies public Ethereum RPC',
      observed_at: new Date().toISOString(), state_query: point, rpc_diagnostics: client.summary() });
  } catch (error) {
    return json({ error: error instanceof Refusal ? error.code : 'upstream_unavailable', provenance: 'No reference value is asserted on failure.',
      ...(client ? { rpc_diagnostics: { ...client.summary(),
        validation: error instanceof Refusal ? error.details.classification || error.code : 'unexpected_failure',
        validation_method: client.summary().trace.at(-1)?.method || 'none' } } : {}) },
      error instanceof Refusal ? error.status : 503);
  } finally { client?.close(); }
}
