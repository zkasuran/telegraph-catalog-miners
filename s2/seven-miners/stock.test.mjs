// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Independent generated states only. No submitted evaluation answers are used.
import assert from 'node:assert/strict';
import test from 'node:test';
import { handle, normalizeControlledStock, strictTime } from './runtime/commerce/88/worker.mjs';

const NOW = Date.parse('2026-11-20T12:00:00.000Z');
const merchant = 'zkasuran-controlled-digital-store';
const location = 'local-digital-allocation';
const sku = 'independent-runtime-stock-probe';
function row(patch = {}) {
  return { merchant, location, sku, on_hand: 37, reserved: 12, revision: 91,
    updated_at: '2026-11-19T09:30:00.000Z', observed_at: '2026-11-20T11:59:59.000Z', ...patch };
}
function normalize(patch) { return normalizeControlledStock(row(patch), merchant, sku, location, NOW); }

test('available stock is current on-hand minus reservations', () => {
  const result = normalize();
  assert.equal(result.quantity, 25);
  assert.equal(result.in_stock, true);
  assert.equal(result.revision, 91);
  assert.notEqual(result.observed_at, result.last_changed_at);
  assert.equal(result.physical_inventory, false);
  assert.equal(result.external_customer, false);
});
test('reservation and empty inventory boundaries', () => {
  assert.equal(normalize({reserved:37}).in_stock, false);
  assert.equal(normalize({reserved:0, on_hand:0}).quantity, 0);
});
test('unknown identity is refused rather than fabricated as out of stock', () => {
  for (const patch of [{sku:'other'}, {merchant:'other'}, {location:'other'}])
    assert.throws(() => normalize(patch), /sku_not_found/);
  assert.throws(() => normalizeControlledStock(null, merchant, sku, location, NOW), /sku_not_found/);
});
test('impossible quantities and revisions are refused', () => {
  for (const patch of [{reserved:-1}, {reserved:38}, {on_hand:1.1}, {on_hand:1000001},
    {reserved:'12'}, {revision:-1}, {revision:NaN}, {revision:1.1}])
    assert.throws(() => normalize(patch));
});
test('old, future and impossible-calendar observations cannot become fresh stock', () => {
  for (const observed_at of ['2026-11-20T11:54:59.999Z', '2026-11-20T12:00:00.001Z',
    '2026-11-19T09:29:59.000Z', '2026-02-30T12:00:00.000Z', 'not a date'])
    assert.throws(() => normalize({observed_at}));
  assert.throws(() => normalize({updated_at:'2026-11-20T12:00:00.001Z'}));
  assert.ok(Number.isNaN(strictTime('2026-02-29T00:00:00Z')));
});
test('generated independent inventory states conserve allocations', () => {
  for (let on_hand = 0; on_hand <= 100; on_hand++) {
    for (const reserved of new Set([0, Math.floor(on_hand / 2), on_hand])) {
      const result = normalize({on_hand, reserved});
      assert.equal(result.quantity + result.reserved, result.on_hand);
      assert.equal(result.in_stock, result.quantity > 0);
    }
  }
});
test('HTTP primary path binds all three identifiers and uses primary session', async () => {
  const seen = [];
  const env = {COMMERCE_DB:{withSession(mode) {
    seen.push(mode);
    return {prepare(sql) { seen.push(sql); return {bind(...args) {
      seen.push(args); return {async first() { return row(); }};
    }}; }};
  }}};
  const response = await handle(new Request(`https://stock.test/truth/v2/sku?merchant=${merchant}&sku=${sku}&location=${location}`), env, NOW);
  assert.equal(response.status, 200);
  assert.deepEqual(seen[2], [merchant, sku, location]);
  assert.equal(seen[0], 'first-primary');
  assert.equal((await response.json()).quantity, 25);
  assert.equal(response.headers.get('cache-control'), 'no-store');
});
test('HTTP validation rejects unsupported merchants, duplicate keys and non-GET', async () => {
  const query = `merchant=${merchant}&sku=${sku}&location=${location}`;
  for (const request of [
    new Request(`https://stock.test/truth/v2/sku?${query}&sku=duplicate`),
    new Request(`https://stock.test/truth/v2/sku?${query}&extra=1`),
    new Request('https://stock.test/truth/v2/sku', {method:'POST'}),
    new Request('https://stock.test/truth/v2/sku?merchant=unrelated&sku=a&location=x'),
  ]) assert.ok((await handle(request, {}, NOW)).status >= 400);
});
test('missing primary binding is unavailable rather than invented stock', async () => {
  const response = await handle(new Request(`https://stock.test/truth/v2/sku?merchant=${merchant}&sku=${sku}&location=${location}`), {}, NOW);
  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), {error:'primary_commerce_database_not_bound'});
});
