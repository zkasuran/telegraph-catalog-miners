// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Test-only replay clock. Original captured data is never redated. Live probes do not preload this.
import { readFileSync } from 'node:fs';

const names = ['valid-name', 'inactive', 'ambiguous', 'lapsed-active'];
const observations = names.map(name => {
  const doc = JSON.parse(readFileSync(new URL(`./baseline/sources/corporate/evidence/public/${name}.json`, import.meta.url)));
  const value = Date.parse(doc.fetched_at);
  if (!Number.isFinite(value)) throw new Error('Fixture observation time is missing');
  return value;
});
const ActualDate = Date;
const anchor = Math.max(...observations);
const offset = ActualDate.now() - anchor;
class FixtureDate extends ActualDate {
  constructor(...args) { super(...(args.length ? args : [ActualDate.now() - offset])); }
  static now() { return ActualDate.now() - offset; }
}
globalThis.Date = FixtureDate;
