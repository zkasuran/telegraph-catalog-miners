// Derived from the preserved deployed telegraph-s2-hashprice worker.
// Version 1.3.1: combine independent canonical/coinbase reads with bounded fallback, retaining txid and Merkle checks.
import { coinbaseReward, serializedCoinbase, verifyCoinbaseProof } from "./coinbase.mjs";
const INTENT = "MINING_HASHPRICE_VERIFY_GENERAL";
const VERSION = "1.3.1";
const DIFF_ONE = 65535n << 208n;
const SCALE = 1000000n;
const TTL_MS = 10000;
const HISTORICAL_TTL_MS = 300000;
const MAX_BYTES = 262144;
const memo = new Map();
const pending = new Map();
const recent = [];
let lastUpstreamAt = -Infinity;
let upstreamQueue = Promise.resolve();

function json(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET, OPTIONS",
      "cache-control": "no-store", // Height-keyed internal caching retains audit timestamps.
      "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
      "x-content-type-options": "nosniff",
      "referrer-policy": "no-referrer"
    }
  });
}

function fail(code, summary, height = null) {
  return {
    intent: INTENT, status: "unavailable", error: { code, message: summary },
    hashprice_sats_per_ph_day: null, block_height: height, hashrate_eh_s: null,
    hashprice_usd_per_ph_day: null, summary, confidence: 0, source: null, as_of: null
  };
}

function requireValue(condition, reason = "unverified_block") {
  if (!condition) throw new Error(reason);
}

function compactTarget(bits) {
  requireValue(Number.isInteger(bits) && bits >= 0 && bits <= 4294967295);
  const exponent = bits >>> 24;
  const mantissa = bits & 8388607;
  requireValue(!(bits & 8388608) && mantissa > 0 && exponent <= 32);
  const target = exponent <= 3 ? BigInt(mantissa) >> BigInt(8 * (3 - exponent)) : BigInt(mantissa) << BigInt(8 * (exponent - 3));
  requireValue(target > 0n && target <= DIFF_ONE);
  return target;
}

function sixDecimals(numerator, denominator) {
  return Number((numerator * SCALE + denominator / 2n) / denominator) / 1e6;
}

async function verifiedAnswer(block, height, source) {
  requireValue(block && block.height === height && /^[a-f0-9]{64}$/.test(block.id), "invalid_block_identity");
  requireValue((block.stale === 0 || block.stale === false), "stale_or_missing_block_status");
  const headerHex = block.extras?.header;
  requireValue(typeof headerHex === "string" && /^[a-f0-9]{160}$/.test(headerHex), "missing_block_header");
  const header = Uint8Array.from(headerHex.match(/../g), x => parseInt(x, 16));
  const view = new DataView(header.buffer);
  requireValue(view.getUint32(72, true) === block.bits);
  requireValue(view.getUint32(68, true) === block.timestamp);
  requireValue(view.getUint32(76, true) === block.nonce);
  requireValue(view.getUint32(0, true) === block.version >>> 0);
  const reverseHex = bytes => Array.from(bytes).reverse().map(x => x.toString(16).padStart(2, "0")).join("");
  requireValue(reverseHex(header.slice(36, 68)) === block.merkle_root);
  if (height > 0) requireValue(reverseHex(header.slice(4, 36)) === block.previousblockhash);
  const first = await crypto.subtle.digest("SHA-256", header);
  const hash = reverseHex(new Uint8Array(await crypto.subtle.digest("SHA-256", first)));
  requireValue(hash === block.id, "header_hash_mismatch");
  const target = compactTarget(block.bits);
  requireValue(BigInt("0x" + hash) <= target);
  const difficulty = Number(DIFF_ONE) / Number(target);
  requireValue(typeof block.difficulty === "number" && Number.isFinite(block.difficulty) && Math.abs(block.difficulty / difficulty - 1) < 5e-13, "difficulty_mismatch");
  const reward = block.extras.reward;
  const fees = block.extras.totalFees;
  const halvings = Math.floor(height / 210000);
  const subsidy = Number(halvings >= 64 ? 0n : 5000000000n >> BigInt(halvings));
  requireValue(Number.isSafeInteger(reward) && reward > 0 && reward <= 2100000000000000);
  requireValue(Number.isSafeInteger(fees) && fees >= 0 && reward <= subsidy + fees, "reward_exceeds_maximum");
  requireValue(Number.isInteger(block.timestamp) && block.timestamp > 0 && block.timestamp <= Date.now() / 1000 + 7200);
  const hashprice = sixDecimals(BigInt(reward) * 86400n * 1000000000000000n * target, DIFF_ONE * 4294967296n);
  const hashrate = sixDecimals(DIFF_ONE * 4294967296n, target * 600n * 1000000000000000000n);
  requireValue(hashprice >= 1 && hashprice <= 1e9 && hashrate <= 1e7);
  return {
    intent: INTENT, status: "ok", hashprice_sats_per_ph_day: hashprice, block_height: height,
    hashrate_eh_s: hashrate, hashprice_usd_per_ph_day: null,
    summary: `Hashprice at Bitcoin block ${height}: ${hashprice.toFixed(6)} sats per PH/s per day (${hashprice.toFixed(2)} rounded to two decimals), using that block's actual coinbase output sum and difficulty. Scheduled subsidy is a maximum; unclaimed reward is not counted as revenue.`,
    confidence: 0.95, source, as_of: new Date(block.timestamp * 1000).toISOString(),
    fetched_at: new Date().toISOString(),
    attribution: "Bitcoin block facts from mempool.space; canonical block identity, canonical serialized coinbase transaction ID, output sum and Merkle inclusion independently checked with Blockstream Esplora.",
    block_hash: block.id, block_time: new Date(block.timestamp * 1000).toISOString(),
    difficulty, difficulty_bits: block.bits, target_hex: target.toString(16).padStart(64, "0"),
    block_reward_sats: reward, subsidy_sats: subsidy, fees_sats: fees,
    maximum_reward_sats: subsidy + fees, unclaimed_reward_sats: subsidy + fees - reward,
    reward_accounting: "subsidy_sats is the scheduled maximum; fees_sats is the publisher-reported transaction fee total; block_reward_sats is the proof-verified coinbase output sum. Any unclaimed amount is excluded from revenue.",
    merkle_root: block.merkle_root,
    formula: "reward_sats * 86400 * 10^15 / (difficulty * 2^32)",
    hashrate_formula: "difficulty * 2^32 / 600 / 10^18",
    methodology: "Single-block gross expected revenue using conventional 2^32 hashes per difficulty unit; not a daily average or a pool payout.",
    usd_unavailable_reason: "No historical BTC/USD rate is fetched; USD hashprice is unavailable.",
    verification: {
      block_header_sha256d: true, proof_of_work_target: true, difficulty_from_bits: true,
      reward_subsidy_plus_fees: reward === subsidy + fees, reward_not_above_subsidy_plus_fees: true, independent_live_reward_check: false
    }
  };
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function reserveRead(deadline) {
  // Serialize request starts across heights; no bursts when clients overlap.
  const slot = upstreamQueue.then(async () => {
    await sleep(Math.max(0, 500 - (Date.now() - lastUpstreamAt)));
    requireValue(Date.now() < deadline, "upstream_timeout");
    lastUpstreamAt = Date.now();
  });
  upstreamQueue = slot.catch(() => {});
  await slot;
}

async function readSource(url, type, deadline) {
  for (let attempt = 0; attempt < 2; attempt++) {
    if (attempt) await sleep(1000);
    await reserveRead(deadline);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), Math.min(4500, Math.max(1, deadline - Date.now())));
    let reader;
    try {
      const response = await fetch(url, {
        signal: controller.signal, redirect: "manual",
        headers: { accept: type === "json" ? "application/json" : "text/plain", "User-Agent": "zkasuran-telegraph-s2-hashprice/1.2 (sequential public Bitcoin block verification)" }
      });
      if (!response.ok) {
        if (attempt === 0 && [429, 500, 502, 503, 504].includes(response.status)) {
          await response.body?.cancel();
          continue;
        }
        throw new Error(`upstream_http_${response.status}`);
      }
      if (Number(response.headers.get("content-length") || 0) > MAX_BYTES) throw new Error("upstream_too_large");
      if (!response.body) throw new Error("upstream_empty");
      reader = response.body.getReader();
      const chunks = [];
      let size = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > MAX_BYTES) throw new Error("upstream_too_large");
        chunks.push(value);
      }
      const bytes = new Uint8Array(size);
      let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
      const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
      return type === "json" ? JSON.parse(text) : text.trim();
    } catch (error) {
      const timedOut = controller.signal.aborted || error.name === "AbortError";
      // Retry transient transport failures once, only inside the original total budget.
      if (attempt === 0 && deadline - Date.now() > 1500 && (timedOut || error instanceof TypeError)) continue;
      if (timedOut) throw new Error("upstream_timeout");
      throw error;
    } finally {
      controller.abort();
      if (reader) await reader.cancel().catch(() => {});
      clearTimeout(timeout);
    }
  }
}

async function originalCrossCheck(value, deadline) {
  const base = "https://blockstream.info/api";
  const identityUrl = `${base}/block-height/${value.block_height}`;
  const identity = await readSource(identityUrl, "text", deadline);
  requireValue(identity === value.block_hash, "independent_block_mismatch");
  const coinbaseIdUrl = `${base}/block/${identity}/txid/0`;
  const txid = await readSource(coinbaseIdUrl, "text", deadline);
  requireValue(/^[a-f0-9]{64}$/.test(txid), "invalid_coinbase_identity");
  const coinbaseUrl = `${base}/tx/${txid}/hex`;
  const reward = await coinbaseReward(await readSource(coinbaseUrl, "text", deadline), txid);
  requireValue(reward === value.block_reward_sats, "independent_reward_mismatch");
  const proofUrl = `${base}/tx/${txid}/merkle-proof`;
  await verifyCoinbaseProof(await readSource(proofUrl, "json", deadline), txid, value.block_height, value.merkle_root);
  value.coinbase_txid = txid;
  value.verification.independent_live_reward_check = true;
  value.verification.independent_canonical_height_check = true;
  value.verification.coinbase_serialized_txid_checked = true;
  value.verification.coinbase_merkle_proof_checked = true;
  value.verification.checked_at = new Date().toISOString();
  value.independent_sources = [identityUrl, coinbaseIdUrl, coinbaseUrl, proofUrl];
  return value;
}

async function crossCheck(value, deadline) {
  const base = "https://blockstream.info/api";
  const pageUrl = `${base}/block/${value.block_hash}/txs/0`;
  let page;
  try {
    // Do not let this optional fast path consume the original verifier's budget.
    page = await readSource(pageUrl, "json", Math.min(deadline, Date.now() + 3500));
  } catch (error) {
    // Large first pages and transient source errors use the original bounded path.
    // Invalid facts never fall back to a less strict verifier.
    if (/^(upstream_http_\d{3}|upstream_timeout|upstream_too_large|upstream_empty)$/.test(error.message) || error instanceof TypeError) {
      requireValue(deadline - Date.now() > 2500, "upstream_timeout");
      return originalCrossCheck(value, deadline);
    }
    throw error;
  }
  requireValue(Array.isArray(page) && page.length > 0 && page.length <= 25, "invalid_coinbase_page");
  const tx = page[0];
  requireValue(tx?.status?.confirmed === true && tx.status.block_height === value.block_height &&
               tx.status.block_hash === value.block_hash && tx.status.block_time * 1000 === Date.parse(value.block_time),
               "independent_block_mismatch");
  const raw = serializedCoinbase(tx);
  const reward = await coinbaseReward(raw, tx.txid);
  requireValue(reward === value.block_reward_sats, "independent_reward_mismatch");
  const proofUrl = `${base}/tx/${tx.txid}/merkle-proof`;
  await verifyCoinbaseProof(await readSource(proofUrl, "json", deadline), tx.txid, value.block_height, value.merkle_root);
  value.coinbase_txid = tx.txid;
  value.verification.independent_live_reward_check = true;
  value.verification.independent_canonical_height_check = true;
  value.verification.coinbase_serialized_txid_checked = true;
  value.verification.coinbase_merkle_proof_checked = true;
  value.verification.coinbase_serialization_source = "Esplora first-transaction JSON, canonically encoded without witness; txid is SHA256d of these bytes.";
  value.verification.checked_at = new Date().toISOString();
  value.independent_sources = [pageUrl, proofUrl];
  return value;
}

function cacheHit(entry) {
  return { ...entry.value, cache: {
    hit: true, age_seconds: Math.max(0, (Date.now() - entry.at) / 1000), ttl_seconds: entry.ttl / 1000,
    policy: "Bounded reuse of verified block facts; source verification timestamp is retained."
  } };
}

async function loadAnswer(height, cacheKey, edge, ctx) {
  const source = `https://mempool.space/api/v1/blocks/${height}`;
  const deadline = Date.now() + 12000;
  try {
    const blocks = await readSource(source, "json", deadline);
    requireValue(Array.isArray(blocks) && blocks.length > 0 && blocks.length <= 20, "invalid_block_list");
    const exact = blocks.filter(b => b && b.height === height);
    requireValue(exact.length === 1, "requested_height_not_unique");
    const value = await crossCheck(await verifiedAnswer(exact[0], height, source), deadline);
    // Old blocks get five minutes, recent blocks ten seconds. Neither is permanent.
    const historical = Date.now() - exact[0].timestamp * 1000 > 7 * 86400000;
    const ttl = historical ? HISTORICAL_TTL_MS : TTL_MS;
    const entry = { at: Date.now(), ttl, value };
    memo.set(height, entry);
    while (memo.size > 128) memo.delete(memo.keys().next().value);
    if (edge) {
      const response = new Response(JSON.stringify(entry), { headers: {
        "content-type": "application/json", "cache-control": `public, max-age=${ttl / 1000}`
      } });
      const write = edge.put(cacheKey, response).catch(() => {});
      if (ctx?.waitUntil) ctx.waitUntil(write); else await write;
    }
    return { ...value, cache: { hit: false, age_seconds: 0, ttl_seconds: ttl / 1000 } };
  } catch (error) {
    const code = /^(upstream_http_\d{3}|upstream_timeout|upstream_too_large|upstream_empty)$/.test(error.message) ? error.message : "unverified_block";
    return {
      ...fail(code, `Hashprice for Bitcoin block ${height} is unavailable because the public sources could not be verified; no value is estimated.`, height),
      source, fetched_at: new Date().toISOString(),
      diagnostic: /^[a-z_]{1,64}$/.test(error.message) ? error.message : "upstream_parse_or_network_error"
    };
  }
}

async function answer(request, ctx) {
  if (request.url.length > 4096) return fail("invalid_params", "Request URL exceeds 4096 characters.");
  const url = new URL(request.url);
  if (request.method !== "GET") return fail("method_not_allowed", "Use GET with the block_height query parameter.");
  if ([...url.searchParams].length > 16) return fail("invalid_params", "Too many query parameters.");
  const heights = url.searchParams.getAll("block_height");
  if (heights.length !== 1 || !/^(0|[1-9][0-9]{0,8})$/.test(heights[0]) || Number(heights[0]) > 1e8) {
    return fail("invalid_params", "Supply one Bitcoin block_height integer from 0 to 100000000; no block is inferred.");
  }
  const height = Number(heights[0]);
  const cached = memo.get(height);
  const age = cached ? Date.now() - cached.at : -1;
  if (cached && age >= 0 && age < cached.ttl) return cacheHit(cached);
  memo.delete(height);
  if (pending.has(height)) return pending.get(height);
  const cacheKey = new Request(`${url.origin}/__verified-cache/${VERSION}/${height}`);
  const edge = globalThis.caches?.default;
  const operation = (async () => {
    if (edge) {
      try {
        const response = await edge.match(cacheKey);
        if (response) {
          const entry = await response.json();
          const age = Date.now() - entry.at;
          if (entry.value?.status === "ok" && entry.value.block_height === height && entry.value.verification?.independent_live_reward_check === true &&
              entry.value.verification.coinbase_serialized_txid_checked === true && entry.value.verification.coinbase_merkle_proof_checked === true &&
              [TTL_MS, HISTORICAL_TTL_MS].includes(entry.ttl) && age >= 0 && age < entry.ttl) {
            memo.set(height, entry);
            while (memo.size > 128) memo.delete(memo.keys().next().value);
            return cacheHit(entry);
          }
        }
      } catch { /* Cache failure never prevents fresh source verification. */ }
    }
    return loadAnswer(height, cacheKey, edge, ctx);
  })();
  pending.set(height, operation);
  try { return await operation; } finally { pending.delete(height); }
}

export default {
  async fetch(request, env, ctx) {
    const start = Date.now();
    const pathname = new URL(request.url).pathname;
    if (pathname === "/health") return json({ status: "ready", intent: INTENT, version: VERSION, checks_upstream: false });
    if (pathname === "/__last") return json({ requests: recent.slice() });
    const literal = /^\/hashprice\/(?:\{block_height\}|%7Bblock_height%7D)$/i.test(pathname);
    if (pathname !== "/hashprice" && !literal) return json({ error: { code: "not_found", message: "Use /hashprice?block_height=840000." } }, 404);
    if (request.method === "OPTIONS") return json({ status: "ready" });
    const value = literal ? fail("unfilled_template", "Use /hashprice?block_height=840000; an unfilled path does not identify a Bitcoin block.") : await answer(request, ctx);
    recent.push({ block_height: value.block_height, status: value.status, code: value.error?.code || null, latency_ms: Date.now() - start, at: new Date().toISOString() });
    if (recent.length > 16) recent.shift();
    return json(value);
  }
};
