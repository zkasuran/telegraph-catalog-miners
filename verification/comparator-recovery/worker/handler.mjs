// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
export const MAX_BODY = 1048576;
const I64_MIN = -(1n << 63n);
const I64_MAX = (1n << 63n) - 1n;

export function parseExact(text) {
  // Preserve integer lexemes before JSON.parse can round int64 values.
  const encoded = text.replace(/"(?:\\.|[^"\\])*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/g, token => {
    if (token.startsWith('"') || /[.eE]/.test(token)) return token;
    return JSON.stringify(token);
  });
  return JSON.parse(encoded);
}

function int64(v) {
  if (typeof v !== "string" || !/^-?(0|[1-9]\d*)$/.test(v)) throw new Error("integer required");
  const n = BigInt(v);
  if (n < I64_MIN || n > I64_MAX) throw new Error("int64 out of range");
  return n;
}

function score(instance, q) {
  if (!q || typeof q !== "object" || !Array.isArray(q.spec) || q.spec.length !== 2) throw new Error("invalid comparator question");
  const truth = int64(q.ground_truth_value), miner = int64(q.miner_value);
  const code = int64(q.spec[0]), tol = int64(q.spec[1]);
  if (truth === 0n) throw new Error("zero truth is not supported by the protocol");
  const e = instance.exports, ptr = e.imw_alloc(16);
  if (!ptr) throw new Error("allocation failed");
  try {
    const view = new DataView(e.memory.buffer);
    view.setBigInt64(ptr, code, true);
    view.setBigInt64(ptr + 8, tol, true);
    const value = e.imw_score_deterministic(truth, miner, ptr);
    if (value < 0n || value > 10000n) return {error: "reference rule refuses overflow input"};
    return {score: Number(value)};
  } finally {
    e.imw_free(ptr, 16);
  }
}

async function readBody(req) {
  if (Number(req.headers.get("content-length") || 0) > MAX_BODY) throw new Error("body too large");
  const reader = req.body?.getReader();
  if (!reader) throw new Error("empty body");
  const chunks = []; let size = 0;
  let timer;
  const deadline = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("body timeout")), 5000); });
  try {
    while (true) {
      const {done, value} = await Promise.race([reader.read(), deadline]);
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY) throw new Error("body too large");
      chunks.push(value);
    }
    const data = new Uint8Array(size); let at = 0;
    for (const chunk of chunks) { data.set(chunk, at); at += chunk.byteLength; }
    return new TextDecoder("utf-8", {fatal: true}).decode(data);
  } finally { clearTimeout(timer); await reader.cancel().catch(() => {}); }
}

const HEADERS = {"content-type": "application/json", "cache-control": "no-store", "x-content-type-options": "nosniff",
  "content-security-policy": "default-src 'none'; frame-ancestors 'none'", "referrer-policy": "no-referrer"};
function reply(body, status = 200) { return new Response(JSON.stringify(body), {status, headers: HEADERS}); }

export function createHandler(instances, binaries, manifest) {
  return {
    async fetch(req) {
      const path = new URL(req.url).pathname;
      if (req.method === "GET" && path === "/health") return reply({version: manifest.version, families: Object.keys(instances)});
      const m = /^\/(runnable|binary)\/([A-Z_]+)(?:\/([a-f0-9]{64})\.wasm)?$/.exec(path);
      if (!m || !Object.hasOwn(instances, m[2])) return reply({error: "not found"}, 404);
      const family = m[2];
      if (m[1] === "binary") {
        if (req.method !== "GET") return reply({error: "GET required"}, 405);
        if (m[3] !== manifest.families[family].keccak.slice(2)) return reply({error: "hash not found"}, 404);
        return new Response(binaries[family], {headers: {...HEADERS, "content-type": "application/wasm", "cache-control": "public, max-age=31536000, immutable"}});
      }
      if (m[3]) return reply({error: "not found"}, 404);
      if (req.method !== "POST") return reply({error: "POST required"}, 405);
      if (!(req.headers.get("content-type") || "").startsWith("application/json")) return reply({error: "JSON required"}, 415);
      try {
        const body = parseExact(await readBody(req));
        if (!body || !Array.isArray(body.batch) || body.batch.length > 100 || body.batch.length < 1) throw new Error("batch must hold 1 to 100 questions");
        const results = body.batch.map(q => score(instances[family], q));
        return reply({results});
      } catch (err) {
        return reply({error: err.message}, err.message === "body too large" ? 413 : err.message === "body timeout" ? 408 : 400);
      }
    }
  };
}
