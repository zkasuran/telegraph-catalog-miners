// Bitcoin block-anchored hashprice from a keyless public Bitcoin explorer.
/**
 * Licence: source-available, no derivatives. Copyright (c) 2026 zkasuran.
 * SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
 *
 * Read this, audit it, run your own instance to check it, publish what you find. Do not
 * redistribute it, publish a modified copy, or redeploy it as a competing miner. Calling
 * the live endpoint is not restricted by the licence at all.
 *
 * Full terms: LICENSE. Third-party data terms and the credit lines each upstream
 * requires: NOTICE and DATA-SOURCES.md. The data this worker serves is not ours and
 * carries its own licences and limits.
 */

const INTENT = 'MINING_HASHPRICE_VERIFY_GENERAL';
const DIFF_ONE = 65535n << 208n;
const SCALE = 1000000n;
function requireValue(condition, reason = 'unverified_block') {
  if (!condition) throw new Error(reason);
}

function compactTarget(bits) {
  requireValue(Number.isInteger(bits) && bits >= 0 && bits <= 0xffffffff);
  const exponent = bits >>> 24;
  const mantissa = bits & 0x7fffff;
  requireValue(!(bits & 0x800000) && mantissa > 0 && exponent <= 32);
  const target = exponent <= 3
    ? BigInt(mantissa) >> BigInt(8 * (3 - exponent))
    : BigInt(mantissa) << BigInt(8 * (exponent - 3));
  requireValue(target > 0n && target <= DIFF_ONE);
  return target;
}

function sixDecimals(numerator, denominator) {
  return Number((numerator * SCALE + denominator / 2n) / denominator) / 1000000;
}

async function verifiedAnswer(block, height, source) {
  requireValue(block && block.height === height && /^[a-f0-9]{64}$/.test(block.id), 'invalid_block_identity');
  requireValue(block.stale === 0, 'stale_or_missing_block_status');
  const headerHex = block.extras?.header;
  requireValue(typeof headerHex === 'string' && /^[a-f0-9]{160}$/.test(headerHex), 'missing_block_header');
  const header = Uint8Array.from(headerHex.match(/../g), x => parseInt(x, 16));
  const view = new DataView(header.buffer);
  requireValue(view.getUint32(72, true) === block.bits);
  requireValue(view.getUint32(68, true) === block.timestamp);
  requireValue(view.getUint32(76, true) === block.nonce);
  requireValue(view.getUint32(0, true) === (block.version >>> 0));
  const reverseHex = bytes => Array.from(bytes).reverse().map(x => x.toString(16).padStart(2, '0')).join('');
  requireValue(reverseHex(header.slice(36, 68)) === block.merkle_root);
  if (height > 0) requireValue(reverseHex(header.slice(4, 36)) === block.previousblockhash);
  const first = await crypto.subtle.digest('SHA-256', header);
  const hash = reverseHex(new Uint8Array(await crypto.subtle.digest('SHA-256', first)));
  requireValue(hash === block.id, 'header_hash_mismatch');
  const target = compactTarget(block.bits);
  requireValue(BigInt('0x' + hash) <= target);
  const difficulty = Number(DIFF_ONE) / Number(target);
  requireValue(typeof block.difficulty === 'number' && Number.isFinite(block.difficulty)
    && Math.abs(block.difficulty / difficulty - 1) < 5e-13, 'difficulty_mismatch');
  const reward = block.extras.reward;
  const fees = block.extras.totalFees;
  const halvings = Math.floor(height / 210000);
  const subsidy = Number(halvings >= 64 ? 0n : 5000000000n >> BigInt(halvings));
  requireValue(Number.isSafeInteger(reward) && reward > 0 && reward <= 2100000000000000);
  requireValue(Number.isSafeInteger(fees) && fees >= 0 && reward === subsidy + fees, 'reward_mismatch');
  requireValue(Number.isInteger(block.timestamp) && block.timestamp > 0
    && block.timestamp <= Date.now() / 1000 + 7200);
  // Exact rational arithmetic rounds once, at the intent's six-decimal boundary.
  const hashprice = sixDecimals(BigInt(reward) * 86400n * 1000000000000000n * target,
    DIFF_ONE * 4294967296n);
  const hashrate = sixDecimals(DIFF_ONE * 4294967296n, target * 600n * 1000000000000000000n);
  requireValue(hashprice >= 1 && hashprice <= 1000000000 && hashrate <= 10000000);
  return {
    intent: INTENT, status: 'ok',
    hashprice_sats_per_ph_day: hashprice, block_height: height,
    hashrate_eh_s: hashrate, hashprice_usd_per_ph_day: null,
    summary: `Hashprice at Bitcoin block ${height}: ${hashprice.toFixed(6)} sats per PH/s per day (${hashprice.toFixed(2)} rounded to two decimals), using that block's subsidy plus fees and difficulty.`,
    confidence: 0.95,
    source, as_of: new Date(block.timestamp * 1000).toISOString(),
    fetched_at: new Date().toISOString(),
    attribution: 'Bitcoin block facts from mempool.space, independently reproducible from Bitcoin mainnet.',
    block_hash: block.id, block_time: new Date(block.timestamp * 1000).toISOString(),
    difficulty, difficulty_bits: block.bits, target_hex: target.toString(16).padStart(64, '0'),
    block_reward_sats: reward, subsidy_sats: subsidy, fees_sats: fees,
    formula: 'reward_sats * 86400 * 10^15 / (difficulty * 2^32)',
    hashrate_formula: 'difficulty * 2^32 / 600 / 10^18',
    methodology: 'Single-block gross expected revenue using conventional 2^32 hashes per difficulty unit; not a daily average or a pool payout.',
    usd_unavailable_reason: 'No historical BTC/USD rate is fetched; USD hashprice is unavailable.',
    verification: {
      block_header_sha256d: true, proof_of_work_target: true,
      difficulty_from_bits: true, reward_subsidy_plus_fees: true,
      independent_live_reward_check: false,
    },
  };
}


export { compactTarget, sixDecimals, verifiedAnswer };
