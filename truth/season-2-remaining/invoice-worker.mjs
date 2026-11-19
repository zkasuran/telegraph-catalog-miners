// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Supplied-record candidate only. No private records, network requests or deployment.
const MAX_BODY = 65536;
const MAX_ITEMS = 32;
const LIMIT = 1000000000000n * 1000000n;
const SCALE = 1000000n;

export function strictJSON(raw) {
  if (typeof raw !== 'string' || raw.length > MAX_BODY) fail('JSON size limit');
  let index = 0;
  function whitespace() { while (/[\x20\t\r\n]/.test(raw[index] || '\0')) index++; }
  function string() {
    if (raw[index] !== '"') fail('string required');
    const start = index++;
    while (index < raw.length) {
      const char = raw[index++];
      if (char === '\\') { index++; continue; }
      if (char === '"') {
        const result = JSON.parse(raw.slice(start, index));
        if (result.length > 16000 || /[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(result)) fail('string limit');
        return result;
      }
    }
    fail('unterminated string');
  }
  function value(depth) {
    if (depth > 16) fail('JSON depth limit');
    whitespace();
    if (raw[index] === '"') return string();
    if (raw[index] === '{') {
      const result = Object.create(null); const seen = new Set(); index++; whitespace();
      if (raw[index] === '}') { index++; return result; }
      while (true) {
        whitespace(); const key = string(); whitespace();
        if (seen.has(key) || seen.size >= 64) fail('duplicate key or object limit');
        seen.add(key);
        if (raw[index++] !== ':') fail('colon required');
        result[key] = value(depth + 1); whitespace();
        const char = raw[index++];
        if (char === '}') return result;
        if (char !== ',') fail('comma required');
      }
    }
    if (raw[index] === '[') {
      const result = []; index++; whitespace();
      if (raw[index] === ']') { index++; return result; }
      while (true) {
        if (result.length >= 200) fail('array limit');
        result.push(value(depth + 1)); whitespace();
        const char = raw[index++];
        if (char === ']') return result;
        if (char !== ',') fail('comma required');
      }
    }
    const match = /^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/.exec(raw.slice(index));
    if (!match) fail('invalid JSON value');
    index += match[0].length;
    const result = JSON.parse(match[0]);
    if (typeof result === 'number' && !Number.isFinite(result)) fail('nonfinite number');
    return result;
  }
  const result = value(0); whitespace();
  if (index !== raw.length) fail('trailing JSON');
  return result;
}

function fail(message) { throw new Error(message); }
function keys(value, required) {
  if (!value || Array.isArray(value) || typeof value !== 'object') fail('object required');
  if (Object.keys(value).sort().join('|') !== [...required].sort().join('|')) fail('unexpected or missing fields');
}
function ident(value) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value)) fail('invalid identifier');
  return value;
}
function money(value) {
  if (typeof value !== 'string' || !/^-?\d{1,13}(\.\d{1,6})?$/.test(value)) fail('decimal string required');
  const negative = value.startsWith('-');
  const [whole, part = ''] = value.replace(/^-/, '').split('.');
  const result = BigInt(whole) * SCALE + BigInt(part.padEnd(6, '0'));
  if (result > LIMIT) fail('amount limit');
  return negative ? -result : result;
}
function moment(value) {
  if (typeof value !== 'string' || value.length > 40 || !/T.*(?:Z|[+-]\d{2}:\d{2})$/.test(value)) fail('timezone required');
  const stamp = Date.parse(value);
  if (!Number.isFinite(stamp)) fail('invalid timestamp');
  return stamp;
}
function inventory(rows) {
  if (!Array.isArray(rows) || rows.length < 1 || rows.length > MAX_ITEMS) fail('line count limit');
  const result = new Map();
  for (const row of rows) {
    keys(row, ['sku', 'quantity', 'unit_price', 'tax']);
    const sku = ident(row.sku);
    if (result.has(sku)) fail('duplicate SKU');
    const numbers = [row.quantity, row.unit_price, row.tax].map(money);
    if (numbers.some(number => number < 0n)) fail('negative line component');
    result.set(sku, numbers);
  }
  return result;
}
function equals(left, right) {
  return left.size === right.size && [...left].every(([sku, numbers]) =>
    right.has(sku) && numbers.every((number, index) => right.get(sku)[index] === number));
}
function amount(value) {
  const negative = value < 0n;
  const positive = negative ? -value : value;
  return `${negative ? '-' : ''}${positive / SCALE}.${String(positive % SCALE).padStart(6, '0')}`;
}
export function compare(invoice, ledger) {
  keys(invoice, ['id', 'po_id', 'currency', 'issued_at', 'lines', 'total']);
  keys(ledger, ['id', 'as_of', 'complete', 'order', 'receipt', 'entries']);
  ident(invoice.id); ident(invoice.po_id); ident(ledger.id);
  if (typeof invoice.currency !== 'string' || !/^[A-Z]{3}$/.test(invoice.currency)) fail('currency code required');
  const cutoff = moment(ledger.as_of);
  if (moment(invoice.issued_at) > cutoff) fail('invoice issued after ledger snapshot');
  if (ledger.complete !== true) fail('incomplete ledger');
  const billed = inventory(invoice.lines);
  const total = money(invoice.total);
  const problems = [];
  let calculated = 0n;
  for (const [quantity, price, tax] of billed.values()) {
    const product = quantity * price;
    if (product % SCALE) fail('sub-six-decimal line total');
    calculated += product / SCALE + tax;
  }
  if (calculated > LIMIT) fail('total limit');
  if (calculated !== total) problems.push('invoice line sum differs from declared total');
  keys(ledger.order, ['id', 'currency', 'created_at', 'lines']);
  keys(ledger.receipt, ['id', 'po_id', 'received_at', 'lines', 'evidence_sha256']);
  if (moment(ledger.order.created_at) > cutoff || moment(ledger.receipt.received_at) > cutoff) fail('future order or receipt');
  if (ledger.order.id !== invoice.po_id || ledger.receipt.po_id !== invoice.po_id) problems.push('purchase order reference differs');
  if (ledger.order.currency !== invoice.currency) problems.push('order currency differs');
  if (!equals(billed, inventory(ledger.order.lines))) problems.push('invoice lines differ from purchase order');
  if (!equals(billed, inventory(ledger.receipt.lines))) problems.push('invoice lines differ from receipt');
  if (!/^[a-f0-9]{64}$/.test(ledger.receipt.evidence_sha256)) fail('receipt evidence hash required');
  if (!Array.isArray(ledger.entries) || ledger.entries.length > MAX_ITEMS) fail('ledger entry count limit');
  let posted = 0n;
  const seen = new Set();
  for (const entry of ledger.entries) {
    keys(entry, ['id', 'invoice_id', 'currency', 'amount', 'posted_at']);
    ident(entry.id); ident(entry.invoice_id);
    if (seen.has(entry.id)) fail('duplicate ledger entry');
    seen.add(entry.id);
    if (moment(entry.posted_at) > cutoff) fail('future posting');
    const quantity = money(entry.amount);
    if (entry.invoice_id !== invoice.id) continue;
    if (entry.currency !== invoice.currency) { problems.push('ledger currency differs'); continue; }
    posted += quantity;
  }
  const discrepancy = total - posted;
  if (discrepancy > LIMIT || discrepancy < -LIMIT) fail('discrepancy limit');
  if (discrepancy) problems.push('invoice total differs from same-currency ledger net');
  return {reconciled: problems.length === 0, discrepancy: amount(discrepancy),
    explanation: problems.length ? problems.join('; ') : 'Order lines, receipt lines and ledger net match.'};
}

function response(data, status = 200) {
  return Response.json(data, {status, headers: {'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': "default-src 'none'; frame-ancestors 'none'"}});
}
export async function handle(request) {
  if (new URL(request.url).pathname !== '/remaining-risk/invoice') return response({error: 'unknown route'}, 404);
  if (request.method !== 'POST') return response({error: 'POST supplied records required'}, 405);
  if (!(request.headers.get('content-type') || '').startsWith('application/json')) return response({error: 'JSON required'}, 415);
  const reader = request.body?.getReader();
  if (!reader) return response({error: 'body required'}, 400);
  const chunks = []; let size = 0;
  try {
    const deadline = Date.now() + 3000;
    while (true) {
      let timer;
      const part = await Promise.race([reader.read(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('body timeout')), Math.max(1, deadline - Date.now()));
      })]).finally(() => clearTimeout(timer));
      if (part.done) break;
      size += part.value.length;
      if (size > MAX_BODY) { await reader.cancel(); return response({error: 'body size limit'}, 413); }
      chunks.push(part.value);
    }
    const body = new Uint8Array(size); let offset = 0;
    for (const chunk of chunks) { body.set(chunk, offset); offset += chunk.length; }
    const supplied = strictJSON(new TextDecoder('utf-8', {fatal: true}).decode(body));
    keys(supplied, ['invoice', 'ledger']);
    const invoice = typeof supplied.invoice === 'string' ? strictJSON(supplied.invoice) : supplied.invoice;
    const ledger = typeof supplied.ledger === 'string' ? strictJSON(supplied.ledger) : supplied.ledger;
    return response(compare(invoice, ledger));
  } catch {
    await reader.cancel().catch(() => {});
    return response({error: 'invalid, incomplete or unsupported supplied records'}, 422);
  } finally { reader.releaseLock(); }
}
export default {fetch: handle};
