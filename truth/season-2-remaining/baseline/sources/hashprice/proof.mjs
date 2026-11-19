// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Bitcoin wire rules: [CORE-TX], [CORE-MERKLE], [CORE-POW] in bibliography.json.
export const LIMITS = Object.freeze({ hexBytes: 100000, outputs: 1000, hashes: 64, txCount: 30000 });
export function requireValue(value, reason = 'invalid_proof') {
  if (!value) throw new Error(reason);
}
export function unhex(text, limit = LIMITS.hexBytes) {
  requireValue(typeof text === 'string' && text.length > 0 && text.length % 2 === 0
    && text.length <= limit * 2 && /^[0-9a-f]+$/.test(text), 'invalid_hex');
  return Uint8Array.from(text.match(/../g), x => parseInt(x, 16));
}
export const hex = bytes => Array.from(bytes, x => x.toString(16).padStart(2, '0')).join('');
export const displayHash = bytes => hex(bytes.slice().reverse());
export function concat(...parts) {
  const out = new Uint8Array(parts.reduce((n, part) => n + part.length, 0));
  let index = 0;
  for (const part of parts) { out.set(part, index); index += part.length; }
  return out;
}
export async function sha256d(bytes) {
  return new Uint8Array(await crypto.subtle.digest('SHA-256', await crypto.subtle.digest('SHA-256', bytes)));
}
export class Reader {
  constructor(bytes) { this.bytes = bytes; this.index = 0; }
  take(count) {
    requireValue(Number.isSafeInteger(count) && count >= 0 && count <= this.bytes.length - this.index, 'truncated_bytes');
    const result = this.bytes.slice(this.index, this.index + count);
    this.index += count;
    return result;
  }
  uint(count) {
    let value = 0n;
    const bytes = this.take(count);
    for (let i = count - 1; i >= 0; i--) value = (value << 8n) | BigInt(bytes[i]);
    return value;
  }
  compact(max) {
    const tag = Number(this.uint(1));
    const value = tag < 253 ? BigInt(tag) : this.uint(tag === 253 ? 2 : tag === 254 ? 4 : 8);
    requireValue(value >= (tag === 253 ? 253n : tag === 254 ? 65536n : tag === 255 ? 4294967296n : 0n), 'noncanonical_compactsize');
    requireValue(value <= BigInt(max), 'count_limit');
    return Number(value);
  }
  end() { requireValue(this.index === this.bytes.length, 'trailing_bytes'); }
}

export async function decodeCoinbase(text, height) {
  const bytes = unhex(text);
  const r = new Reader(bytes);
  const version = r.take(4);
  const witness = bytes[4] === 0;
  if (witness) requireValue(hex(r.take(2)) === '0001', 'unsupported_witness_flags');
  const start = r.index;
  requireValue(r.compact(1) === 1, 'not_coinbase');
  requireValue(hex(r.take(32)) === '00'.repeat(32) && hex(r.take(4)) === 'ffffffff', 'not_coinbase');
  const script = r.take(r.compact(100));
  requireValue(script.length >= 2, 'coinbase_script_size');
  // BIP34 became mandatory on mainnet at height 227931.
  if (height >= 227931) {
    const count = script[0];
    requireValue(count >= 1 && count <= 5 && count + 1 <= script.length, 'coinbase_height_encoding');
    const n = script.slice(1, count + 1);
    requireValue(!(n[n.length - 1] & 128), 'negative_coinbase_height');
    requireValue(n[n.length - 1] !== 0 || (n.length > 1 && (n[n.length - 2] & 128)), 'nonminimal_coinbase_height');
    let recorded = 0n;
    for (let i = n.length - 1; i >= 0; i--) recorded = (recorded << 8n) | BigInt(n[i]);
    requireValue(recorded === BigInt(height), 'coinbase_height_mismatch');
  }
  r.take(4);
  const count = r.compact(LIMITS.outputs);
  requireValue(count > 0, 'no_outputs');
  let reward = 0n;
  for (let i = 0; i < count; i++) {
    const value = r.uint(8);
    requireValue(value <= 2100000000000000n, 'money_range');
    reward += value;
    requireValue(reward <= 2100000000000000n, 'money_range');
    r.take(r.compact(10000));
  }
  const end = r.index;
  if (witness) {
    const items = r.compact(100);
    requireValue(items > 0, 'superfluous_witness');
    for (let i = 0; i < items; i++) r.take(r.compact(10000));
  }
  const locktime = r.take(4);
  r.end();
  return { txid: displayHash(await sha256d(concat(version, bytes.slice(start, end), locktime))), reward: Number(reward), outputs: count };
}

export async function verifyInclusion(text, header, txid, expectedCount) {
  const r = new Reader(unhex(text, 4096));
  requireValue(hex(r.take(80)) === header, 'proof_header_mismatch');
  const count = Number(r.uint(4));
  requireValue(count > 0 && count <= LIMITS.txCount && count === expectedCount, 'proof_count_mismatch');
  const size = r.compact(LIMITS.hashes);
  requireValue(size > 0 && size <= count, 'proof_hash_count');
  const hashes = Array.from({ length: size }, () => r.take(32));
  const flags = r.take(r.compact(32));
  r.end();
  requireValue(flags.length > 0, 'proof_flags');
  let bitsUsed = 0, hashUsed = 0;
  const matches = [];
  const width = level => Math.ceil(count / 2 ** level);
  async function walk(level, pos) {
    requireValue(bitsUsed < flags.length * 8, 'proof_bit_overflow');
    const expanded = (flags[Math.floor(bitsUsed / 8)] >> (bitsUsed % 8)) & 1;
    bitsUsed++;
    if (level === 0 || !expanded) {
      requireValue(hashUsed < hashes.length, 'proof_hash_overflow');
      const value = hashes[hashUsed++];
      if (level === 0 && expanded) matches.push({ pos, txid: displayHash(value) });
      return value;
    }
    const left = await walk(level - 1, pos * 2);
    let right = left;
    if (pos * 2 + 1 < width(level - 1)) {
      right = await walk(level - 1, pos * 2 + 1);
      requireValue(hex(left) !== hex(right), 'merkle_mutation');
    }
    return sha256d(concat(left, right));
  }
  let level = 0;
  while (width(level) > 1) level++;
  const root = await walk(level, 0);
  requireValue(hashUsed === hashes.length && Math.ceil(bitsUsed / 8) === flags.length, 'unused_proof_data');
  requireValue(matches.length === 1 && matches[0].pos === 0 && matches[0].txid === txid, 'coinbase_not_proven');
  requireValue(hex(root) === header.slice(72, 136), 'merkle_root_mismatch');
  return true;
}
