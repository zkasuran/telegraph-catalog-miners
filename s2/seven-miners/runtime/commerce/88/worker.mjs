// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import { bounded, guarded, params, requireValue, text } from '../common.mjs';

export const STOCK_QUERY = `SELECT merchant, sku, location, on_hand, reserved, revision, updated_at,
  strftime('%Y-%m-%dT%H:%M:%fZ', 'now') AS observed_at
  FROM stock WHERE merchant = ? AND sku = ? AND location = ? LIMIT 1`;
export const OBSERVATION_FRESH_MS = 300000;
// A strict calendar check rejects Date.parse's normalization of impossible dates.
export function strictTime(value) {
  if (typeof value !== 'string') return NaN;
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(Z|[+-]\d{2}:\d{2})$/.exec(value);
  if (!m) return NaN;
  const [year, month, day, hour, minute, second] = m.slice(1, 7).map(Number);
  const days = new Date(Date.UTC(year, month, 0)).getUTCDate();
  if (year < 1000 || month < 1 || month > 12 || day < 1 || day > days || hour > 23 || minute > 59 || second > 59) return NaN;
  if (m[8] !== 'Z' && (Number(m[8].slice(1, 3)) > 23 || Number(m[8].slice(4)) > 59)) return NaN;
  return Date.parse(value);
}
export function normalizeControlledStock(row, merchant, sku, location, now) {
  requireValue(row && row.merchant === merchant && row.sku === sku && row.location === location, 'sku_not_found', 404);
  requireValue(Number.isSafeInteger(row.on_hand) && Number.isSafeInteger(row.reserved) && row.reserved >= 0 && row.on_hand >= row.reserved && row.on_hand <= 1000000, 'invalid_inventory', 502);
  requireValue(Number.isSafeInteger(row.revision) && row.revision >= 0, 'invalid_revision', 502);
  const changed = strictTime(row.updated_at), observed = strictTime(row.observed_at);
  requireValue(Number.isSafeInteger(now) && now >= 0 && Number.isFinite(changed) && changed >= 0 && changed <= now, 'invalid_or_future_mutation', 503);
  requireValue(Number.isFinite(observed) && observed >= changed && observed <= now && now - observed <= OBSERVATION_FRESH_MS, 'stale_or_future_primary_observation', 503);
  return { merchant, sku, location, in_stock: row.on_hand > row.reserved, quantity: row.on_hand - row.reserved,
    on_hand: row.on_hand, reserved: row.reserved, revision: row.revision,
    observed_at: row.observed_at, last_changed_at: row.updated_at, as_of: row.observed_at,
    authority: 'current primary D1 controlled digital allocation state',
    provenance: 'observed controlled digital allocation store; not external retailer inventory',
    external_customer: false, physical_inventory: false };
}
export async function handle(request, env, now) {
  return guarded(async () => {
    const p = params(request, ['merchant', 'sku', 'location']);
    const merchant = text(p.get('merchant'), 'merchant'), sku = text(p.get('sku'), 'sku'), location = text(p.get('location'), 'location');
    requireValue(merchant === 'zkasuran-controlled-digital-store' && location === 'local-digital-allocation', 'unsupported_merchant_or_location');
    requireValue(env.COMMERCE_DB?.withSession, 'primary_commerce_database_not_bound', 503);
    const primary = env.COMMERCE_DB.withSession('first-primary');
    const row = await bounded(primary.prepare(STOCK_QUERY).bind(merchant, sku, location).first());
    // Time is checked after the primary read. The SQL clock is never a mutation.
    return normalizeControlledStock(row, merchant, sku, location, now ?? Date.now());
  });
}
