// Adapted from this team's generation-2 historical.mjs verifier.
// Bitcoin consensus transaction serialization; no rival code or answer data.
const requireValue = (ok, reason) => { if (!ok) throw new Error(reason); };
const hex = bytes => Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
const unhex = text => Uint8Array.from(text.match(/../g), b => parseInt(b, 16));
const reverseHex = bytes => hex(Uint8Array.from(bytes).reverse());
const sha256d = async bytes => new Uint8Array(await crypto.subtle.digest("SHA-256", await crypto.subtle.digest("SHA-256", bytes)));

export async function coinbaseReward(raw, txid) {
  requireValue(typeof raw === "string" && raw.length <= 262144 && /^(?:[a-f0-9]{2})+$/.test(raw), "invalid_coinbase_hex");
  const bytes = unhex(raw), view = new DataView(bytes.buffer);
  let at = 0;
  const take = n => {
    requireValue(Number.isSafeInteger(n) && n >= 0 && at + n <= bytes.length, "truncated_coinbase");
    const part = bytes.slice(at, at + n); at += n; return part;
  };
  const variable = () => {
    const first = take(1)[0];
    if (first < 253) return first;
    const pos = at;
    take(first === 253 ? 2 : first === 254 ? 4 : 8);
    const value = first === 253 ? view.getUint16(pos, true) : first === 254 ? view.getUint32(pos, true) : Number(view.getBigUint64(pos, true));
    requireValue(Number.isSafeInteger(value) && value <= 262144, "invalid_coinbase_length");
    requireValue(value >= (first === 253 ? 253 : first === 254 ? 65536 : 4294967296), "noncanonical_coinbase_length");
    return value;
  };
  const version = take(4);
  const witness = bytes[at] === 0;
  if (witness) requireValue(take(2)[1] === 1, "invalid_witness_flag");
  const inputStart = at;
  requireValue(variable() === 1, "invalid_coinbase_input_count");
  requireValue(hex(take(32)) === "0".repeat(64) && hex(take(4)) === "ffffffff", "not_coinbase");
  const scriptLength = variable();
  requireValue(scriptLength >= 2 && scriptLength <= 100, "invalid_coinbase_script");
  take(scriptLength); take(4);
  const outputs = variable();
  requireValue(outputs > 0 && outputs <= 10000, "invalid_coinbase_outputs");
  let reward = 0n;
  for (let i = 0; i < outputs; i++) {
    const pos = at;
    take(8);
    reward += view.getBigUint64(pos, true);
    requireValue(reward <= 2100000000000000n, "invalid_coinbase_value");
    take(variable());
  }
  const outputEnd = at;
  if (witness) {
    const count = variable();
    for (let i = 0; i < count; i++) take(variable());
  }
  const locktime = take(4);
  requireValue(at === bytes.length, "coinbase_trailing_bytes");
  const serialized = new Uint8Array(4 + outputEnd - inputStart + 4);
  serialized.set(version); serialized.set(bytes.slice(inputStart, outputEnd), 4); serialized.set(locktime, serialized.length - 4);
  requireValue(reverseHex(await sha256d(serialized)) === txid, "coinbase_txid_mismatch");
  return Number(reward);
}

export async function verifyCoinbaseProof(proof, txid, height, merkleRoot) {
  requireValue(/^[a-f0-9]{64}$/.test(txid) && /^[a-f0-9]{64}$/.test(merkleRoot), "invalid_coinbase_identity");
  requireValue(proof?.block_height === height && proof.pos === 0 && Array.isArray(proof.merkle) && proof.merkle.length <= 32, "invalid_coinbase_proof");
  let root = unhex(txid).reverse();
  for (const sibling of proof.merkle) {
    requireValue(/^[a-f0-9]{64}$/.test(sibling), "invalid_coinbase_proof");
    const joined = new Uint8Array(64);
    joined.set(root); joined.set(unhex(sibling).reverse(), 32);
    root = await sha256d(joined);
  }
  requireValue(reverseHex(root) === merkleRoot, "coinbase_merkle_mismatch");
}

// Canonical non-witness serialization from independently published Esplora JSON.
// Witness is excluded by the definition of txid, as in coinbaseReward above.
export function serializedCoinbase(tx) {
  requireValue(tx && /^[a-f0-9]{64}$/.test(tx.txid), "invalid_coinbase_identity");
  requireValue(Number.isInteger(tx.version) && tx.version >= -2147483648 && tx.version <= 4294967295, "invalid_coinbase_version");
  requireValue(Number.isInteger(tx.locktime) && tx.locktime >= 0 && tx.locktime <= 4294967295, "invalid_coinbase_locktime");
  requireValue(Array.isArray(tx.vin) && tx.vin.length === 1 && tx.vin[0].is_coinbase === true, "not_coinbase");
  requireValue(Array.isArray(tx.vout) && tx.vout.length > 0 && tx.vout.length <= 10000, "invalid_coinbase_outputs");
  const vin = tx.vin[0];
  requireValue(vin.txid === "0".repeat(64) && vin.vout === 4294967295, "not_coinbase");
  requireValue(Number.isInteger(vin.sequence) && vin.sequence >= 0 && vin.sequence <= 4294967295, "invalid_coinbase_sequence");
  const number = (n, size) => {
    const b = new Uint8Array(size);
    let v = BigInt.asUintN(size * 8, BigInt(n));
    for (let i = 0; i < size; i++) { b[i] = Number(v & 255n); v >>= 8n; }
    return b;
  };
  const variable = n => n < 253 ? number(n, 1) : n <= 65535 ? new Uint8Array([253, ...number(n, 2)]) : new Uint8Array([254, ...number(n, 4)]);
  const script = text => {
    requireValue(typeof text === "string" && text.length <= 262144 && /^(?:[a-f0-9]{2})*$/.test(text), "invalid_coinbase_script");
    const bytes = text.length ? unhex(text) : new Uint8Array();
    return [variable(bytes.length), bytes];
  };
  const chunks = [number(tx.version, 4), number(1, 1), new Uint8Array(32), number(vin.vout, 4), ...script(vin.scriptsig), number(vin.sequence, 4), variable(tx.vout.length)];
  for (const out of tx.vout) {
    requireValue(Number.isSafeInteger(out.value) && out.value >= 0 && out.value <= 2100000000000000, "invalid_coinbase_value");
    chunks.push(number(out.value, 8), ...script(out.scriptpubkey));
  }
  chunks.push(number(tx.locktime, 4));
  const size = chunks.reduce((sum, b) => sum + b.length, 0);
  requireValue(size <= 131072, "invalid_coinbase_length");
  const bytes = new Uint8Array(size);
  let at = 0;
  for (const b of chunks) { bytes.set(b, at); at += b.length; }
  return hex(bytes);
}
