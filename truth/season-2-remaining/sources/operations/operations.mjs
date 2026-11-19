// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Data: ethPandaOps Xatu and MET Norway, CC BY 4.0. Fintraffic, CC BY 4.0.
export const LIMITS = Object.freeze({requestBytes: 4096, responseBytes: 1000000,
  timeoutMs: 12000, records: 500, text: 256, validatorLagSeconds: 3600,
  metarAgeSeconds: 5400, aisAgeSeconds: 900, forecastAgeSeconds: 21600});
export class Refusal extends Error {}
const must = (value, message) => { if (!value) throw new Refusal(message); };
const numeric = (n, min, max) => typeof n === 'number' && Number.isFinite(n) && n >= min && n <= max;
export function time(value) {
  must(typeof value === 'string' && /(?:Z|[+-]\d\d:\d\d)$/.test(value), 'explicit timestamp offset required');
  const seconds = Date.parse(value) / 1000;
  must(Number.isFinite(seconds), 'invalid timestamp');
  return seconds;
}
export function inputText(value) {
  must(typeof value === 'string' && new TextEncoder().encode(value).length <= LIMITS.text
    && value.trim().length && !/[\u0000-\u001f\u007f]/.test(value), 'invalid bounded text');
  return value.normalize('NFC').trim().replace(/\s+/g, ' ');
}
export function validatorRecord(payload, identity, requested, nowSeconds, genesis) {
  must(Number.isInteger(requested) && requested >= 0 && requested <= 1000000000, 'invalid validator index');
  const rows = payload?.fct_validator_balance;
  const identities = identity?.dim_validator;
  must(Array.isArray(rows) && rows.length === 1 && Array.isArray(identities) && identities.length === 1, 'missing or ambiguous validator');
  const row = rows[0], id = identities[0];
  must(row.validator_index === requested && id.validator_index === requested, 'validator identity mismatch');
  must(typeof id.pubkey === 'string' && /^0x[0-9a-f]{96}$/i.test(id.pubkey), 'invalid validator pubkey');
  must(Number.isInteger(row.epoch) && row.epoch >= 0 && row.epoch_start_date_time === genesis + row.epoch * 384, 'epoch semantics mismatch');
  const age = nowSeconds - row.epoch_start_date_time;
  must(age >= 0 && age <= LIMITS.validatorLagSeconds, 'stale or future validator epoch');
  must(Number.isSafeInteger(row.effective_balance) && numeric(row.effective_balance, 0, 2100000000000), 'invalid effective balance');
  return {effective_balance: row.effective_balance, validator_index: requested, pubkey: id.pubkey,
    epoch: row.epoch, epoch_start_time: new Date(row.epoch_start_date_time * 1000).toISOString(),
    age_seconds: age, network: 'Ethereum mainnet', source_kind: 'delayed finalized consensus data',
    attribution: 'ethPandaOps Xatu, CC BY 4.0', source: 'https://lab.ethpandaops.io/api/v1/mainnet/fct_validator_balance'};
}
export function checkBeacon(record, response, expectedPubkey) {
  must(response?.execution_optimistic === false && response?.finalized === true, 'unfinalized or optimistic beacon state');
  const rows = response.data;
  must(Array.isArray(rows) && rows.length <= 64, 'invalid beacon rows');
  const matches = rows.filter(row => row.index === String(record.validator_index));
  must(matches.length === 1, 'missing or ambiguous beacon identity');
  const row = matches[0];
  must(row.validator?.pubkey === expectedPubkey && expectedPubkey === record.pubkey, 'beacon pubkey mismatch');
  must(/^\d+$/.test(row.validator.effective_balance) && Number(row.validator.effective_balance) === record.effective_balance, 'beacon balance disagreement');
  return true;
}
const CITIES = Object.freeze({London: 'EGLC', Karachi: 'OPKC', Tokyo: 'RJTT'});
export function stationFor(location) {
  const key = inputText(location);
  const found = Object.keys(CITIES).find(city => city.toLowerCase() === key.toLowerCase());
  if (found) return {city: found, station: CITIES[found]};
  if (Object.values(CITIES).includes(key.toUpperCase())) return {city: key.toUpperCase(), station: key.toUpperCase()};
  throw new Refusal('unsupported location, no guessed station');
}
export function metarObservation(xml, location, nowSeconds) {
  const {city, station} = stationFor(location);
  must(typeof xml === 'string' && new TextEncoder().encode(xml).length <= LIMITS.responseBytes
    && !/<!DOCTYPE|<!ENTITY/i.test(xml) && /<metno:aviationProducts[\s>]/.test(xml)
    && /<\/metno:aviationProducts>\s*$/.test(xml), 'invalid bounded METAR XML');
  const reports = [...xml.matchAll(/<metno:meteorologicalAerodromeReport>([\s\S]*?)<\/metno:meteorologicalAerodromeReport>/g)];
  must(reports.length > 0 && reports.length <= LIMITS.records, 'missing or excessive METAR reports');
  const eligible = [];
  for (const [, report] of reports) {
    const reportedStation = report.match(/<metno:icaoAirportIdentifier>([^<]+)<\/metno:icaoAirportIdentifier>/)?.[1];
    must(reportedStation === station, 'METAR station mismatch');
    const stamp = report.match(/<gml:timePosition>([^<]+)<\/gml:timePosition>/)?.[1];
    // MET's documented XML times are UTC, unlike arbitrary user timestamps.
    must(typeof stamp === 'string' && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:Z)?$/.test(stamp), 'bad MET UTC time');
    const observed = time(stamp.endsWith('Z') ? stamp : stamp + 'Z');
    const text = report.match(/<metno:metarText>([\s\S]*?)<\/metno:metarText>/)?.[1]?.trim();
    must(typeof text === 'string' && text.length <= 2048 && !/[<&]/.test(text), 'invalid raw METAR');
    const reportTime = text.match(/^(?:METAR |SPECI )?(?:COR )?([A-Z]{4}) (\d{2})(\d{2})(\d{2})Z\b/);
    must(reportTime && reportTime[1] === station, 'raw METAR identity missing');
    const dt = new Date(observed * 1000);
    must(dt.getUTCDate() === Number(reportTime[2]) && dt.getUTCHours() === Number(reportTime[3])
      && dt.getUTCMinutes() === Number(reportTime[4]), 'METAR raw/XML time mismatch');
    if (nowSeconds - observed < 0 || nowSeconds - observed > LIMITS.metarAgeSeconds) continue;
    const temperatures = [...text.matchAll(/(?:^|\s)(M?\d{2})\/(M?\d{2}|\/\/)(?=\s|=|$)/g)];
    if (temperatures.length !== 1) continue;
    const code = temperatures[0][1];
    const celsius = Number(code.replace('M', '-'));
    must(numeric(celsius, -90, 60), 'out-of-range METAR temperature');
    eligible.push({observed, text, celsius});
  }
  must(eligible.length, 'no fresh observed station temperature');
  eligible.sort((a, b) => b.observed - a.observed);
  const latest = eligible[0];
  const sameTime = eligible.filter(r => r.observed === latest.observed);
  must(sameTime.every(r => r.celsius === latest.celsius), 'conflicting latest station reports');
  return {temperature_c: latest.celsius, temp_c: latest.celsius, location: city,
    station, observed_at: new Date(latest.observed * 1000).toISOString(),
    age_seconds: nowSeconds - latest.observed, observation_type: 'station observation',
    spatial_scope: 'Named city airport station, not an interpolated city-centre reading',
    raw_metar: latest.text, attribution: 'MET Norway / Avinor, CC BY 4.0 and NLOD 2.0',
    source: `https://api.met.no/weatherapi/tafmetar/1.0/metar.xml?icao=${station}`};
}
export function forecast(payload, nowSeconds) {
  must(payload?.properties?.meta?.units?.air_temperature === 'celsius', 'forecast units mismatch');
  const meta = payload.properties.meta;
  must(nowSeconds - time(meta.updated_at) >= 0 && nowSeconds - time(meta.updated_at) <= LIMITS.forecastAgeSeconds, 'stale forecast issue');
  const hour = Math.floor(nowSeconds / 3600) * 3600;
  const rows = payload.properties.timeseries;
  must(Array.isArray(rows) && rows.length <= LIMITS.records, 'invalid forecast series');
  const matches = rows.filter(r => time(r.time) === hour);
  must(matches.length === 1, 'exact forecast hour missing or ambiguous');
  const value = matches[0].data?.instant?.details?.air_temperature;
  must(numeric(value, -90, 60), 'invalid forecast temperature');
  return {temperature_c: value, temp_c: value, observation_type: 'forecast estimate',
    valid_time: matches[0].time, issued_at: meta.updated_at, attribution: 'MET Norway, CC BY 4.0 and NLOD 2.0'};
}
export function distanceKm(origin, destination) {
  must(Array.isArray(origin) && Array.isArray(destination) && origin.length === 2 && destination.length === 2
    && numeric(origin[0], -180, 180) && numeric(origin[1], -90, 90)
    && numeric(destination[0], -180, 180) && numeric(destination[1], -90, 90), 'bad coordinates');
  const rad = x => x * Math.PI / 180;
  const a = Math.sin(rad(destination[1] - origin[1]) / 2) ** 2
    + Math.cos(rad(origin[1])) * Math.cos(rad(destination[1])) * Math.sin(rad(destination[0] - origin[0]) / 2) ** 2;
  return 6371.0088 * 2 * Math.atan2(Math.sqrt(Math.min(1, a)), Math.sqrt(Math.max(0, 1 - a)));
}
export const RISK_POLICY = Object.freeze({id: 'constant-speed-arrival-screen-v1',
  probability: null, assumption: 'Continue at reported speed along a great-circle lower-bound route',
  target: 'The agent-announced port ETA, not a contractual delivery deadline',
  cuts: {low: 0.5, moderate: 1, high: 1.5}, maximumHours: 48});
export function assessShipment(task, nowSeconds) {
  must(task && task.arrival_with_cargo === true && task.actual_arrival === null, 'not a cargo lane in transit');
  must(Number.isInteger(task.mmsi) && task.mmsi >= 100000000 && task.mmsi <= 999999999
    && Number.isInteger(task.port_call_id), 'missing real vessel/call identity');
  const eta = time(task.eta), observation = time(task.position_time), issued = time(task.eta_issue_time);
  const canonicalPort = value => value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toUpperCase().replace(/[^A-Z0-9]/g, '');
  const destinationTokens = inputText(task.ais_destination).split(/[<>\-/]+/).map(canonicalPort);
  const aliases = inputText(task.port_name).split(/[()]+/).filter(Boolean).map(canonicalPort);
  must(destinationTokens.some(token => token === task.port || aliases.includes(token)), 'AIS destination does not identify this port');
  must(nowSeconds - time(task.metadata_time) >= 0 && nowSeconds - time(task.metadata_time) <= 86400, 'stale destination metadata');
  must(issued <= nowSeconds && observation <= nowSeconds && nowSeconds - observation <= LIMITS.aisAgeSeconds, 'future or stale input');
  const remaining = (eta - nowSeconds) / 3600;
  must(remaining > 0 && remaining <= RISK_POLICY.maximumHours, 'not a prospective 48-hour arrival');
  must(numeric(task.sog_knots, 2, 50) && task.nav_status === 0, 'not observed underway at supported speed');
  const km = distanceKm(task.position, task.destination);
  must(km >= 10 && km <= 2000, 'unsupported distance');
  const rad = x => x * Math.PI / 180;
  const [a, b] = task.position.map(rad), [c, d] = task.destination.map(rad);
  const bearing = (Math.atan2(Math.sin(c - a) * Math.cos(d), Math.cos(b) * Math.sin(d)
    - Math.sin(b) * Math.cos(d) * Math.cos(c - a)) * 180 / Math.PI + 360) % 360;
  must(numeric(task.cog_degrees, 0, 359.999), 'missing vessel course');
  const divergence = Math.abs(((task.cog_degrees - bearing + 540) % 360) - 180);
  must(divergence <= 75, 'observed course not toward target port');
  const ratio = km / 1.852 / remaining / task.sog_knots;
  const risk = ratio <= 0.5 ? 'low' : ratio <= 1 ? 'moderate' : ratio <= 1.5 ? 'high' : 'severe';
  return {risk_level: risk, factors: [`Required/reported speed ratio ${ratio.toFixed(4)}`,
    `Great-circle lower-bound distance ${km.toFixed(3)} km`, 'No customs, queue or weather forecast in this screen'],
    summary: `Own uncalibrated forward-looking arrival screen: ${risk}. Target is the announced port ETA, not a cargo delivery promise.`,
    assessment_time: new Date(nowSeconds * 1000).toISOString(), eta: task.eta,
    mmsi: task.mmsi, port_call_id: task.port_call_id, policy: RISK_POLICY.id,
    attribution: 'Fintraffic / digitraffic.fi, CC BY 4.0', provenance_class: 'unverified',
    source_kind: 'model', physical_outcome_verified: false};
}
export function serviceEstimate(record, carrier, route) {
  must(record && inputText(carrier) === record.carrier && inputText(route) === record.route, 'unsupported exact service scope');
  must(Number.isInteger(record.transit_days_max) && numeric(record.transit_days_max, 0, 365)
    && record.is_estimate === true && record.unit === 'business days', 'invalid licensed estimate');
  return {...record, summary: 'Published standard service estimate. No booked order, lane quote or actual delivery is asserted.'};
}
export async function boundedGet(url, fetcher = fetch) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), LIMITS.timeoutMs);
  try {
    const response = await fetcher(url, {redirect: 'manual', signal: controller.signal,
      headers: {'User-Agent': 'Telegraph-operations/1.0 (https://github.com/zkasuran)', 'Accept': 'application/json,text/xml,text/plain'}});
    must(response.ok && response.body, `upstream status ${response.status}`);
    const reader = response.body.getReader(); let bytes = 0; const chunks = [];
    try {
      while (true) {
        const {done, value} = await reader.read(); if (done) break;
        bytes += value.length; must(bytes <= LIMITS.responseBytes, 'upstream response too large'); chunks.push(value);
      }
    } finally { await reader.cancel(); }
    const result = new Uint8Array(bytes); let offset = 0;
    for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
    return new TextDecoder('utf-8', {fatal: true}).decode(result);
  } finally { clearTimeout(timer); }
}
export async function handle(request, fetcher = fetch, nowSeconds = Date.now() / 1000, catalog = {}) {
  try {
    must(request.method === 'GET', 'only GET is supported');
    const url = new URL(request.url);
    must(new TextEncoder().encode(url.search).length <= LIMITS.requestBytes, 'request too large');
    const params = Object.fromEntries(url.searchParams);
    must(Object.keys(params).length === url.searchParams.size, 'duplicate parameters');
    let value;
    if (url.pathname.endsWith('/validator')) {
      must(Object.keys(params).length === 1 && /^\d{1,10}$/.test(params.validator_index), 'only validator_index is accepted');
      const index = Number(params.validator_index);
      must(index <= 1000000000, 'validator index range');
      const base = 'https://lab.ethpandaops.io/api/v1/mainnet/';
      const payload = JSON.parse(await boundedGet(base + `fct_validator_balance?validator_index_eq=${index}&order_by=epoch%20desc&page_size=1`, fetcher));
      const identity = JSON.parse(await boundedGet(base + `dim_validator?validator_index_eq=${index}&page_size=1`, fetcher));
      value = validatorRecord(payload, identity, index, nowSeconds, 1606824023);
    } else if (url.pathname.endsWith('/weather')) {
      must(Object.keys(params).length === 1, 'only location is accepted');
      const selected = stationFor(params.location);
      const xml = await boundedGet(`https://api.met.no/weatherapi/tafmetar/1.0/metar.xml?icao=${selected.station}`, fetcher);
      value = metarObservation(xml, params.location, nowSeconds);
    } else if (url.pathname.endsWith('/delivery')) {
      must(Object.keys(params).length === 2 && 'carrier' in params && 'route' in params, 'carrier and route required');
      const record = catalog.delivery?.find(r => r.carrier === inputText(params.carrier) && r.route === inputText(params.route));
      value = serviceEstimate(record, params.carrier, params.route);
    } else if (url.pathname.endsWith('/shipment')) {
      must(Object.keys(params).length === 1 && 'shipment' in params, 'only shipment is accepted');
      const record = catalog.shipment?.find(r => r.request === inputText(params.shipment));
      must(record, 'no exact dated model assessment for this shipment');
      value = {...record.result, snapshot_only: true};
    } else { throw new Refusal('unknown operation path'); }
    return Response.json(value, {headers: {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}});
  } catch (error) {
    return Response.json({error: error instanceof Refusal ? error.message : 'upstream unavailable or invalid', code: 'UNAVAILABLE'},
      {status: 422, headers: {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}});
  }
}
export default {fetch(request) { return handle(request); }};
