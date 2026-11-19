// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
// Pure rollout planning only. No credentials, file writes or network calls.
import {createHash} from 'node:crypto';

export const PREFIX = '/_zkcmp_s2_20261118/';
const sha = value => createHash('sha256').update(value).digest('hex');
function fail(message) { throw new Error(message); }
function canonical(value) {
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  if (value && typeof value === 'object') {
    return '{' + Object.keys(value).sort().map(k => JSON.stringify(k) + ':' + canonical(value[k])).join(',') + '}';
  }
  if (value === undefined || typeof value === 'bigint') fail('non-JSON snapshot');
  return JSON.stringify(value);
}
const clone = value => JSON.parse(canonical(value));
const equal = (a, b) => canonical(a) === canonical(b);
export const fingerprint = value => sha(canonical(value));

function moduleBytes(module) {
  if (!/^[A-Za-z0-9_.-]+$/.test(module.name)) fail('unsafe module name');
  if (!['esm', 'wasm', 'text', 'data', 'commonjs'].includes(module.type)) fail('unknown module type');
  if (typeof module.content_base64 !== 'string' || module.content_base64.length > 90000000) fail('invalid module bytes');
  const bytes = Buffer.from(module.content_base64, 'base64');
  if (bytes.toString('base64') !== module.content_base64 || sha(bytes) !== module.sha256) fail('module bytes/hash mismatch');
  return bytes;
}

export function validateSnapshot(snapshot) {
  if (!snapshot || snapshot.schema !== 1 || snapshot.complete !== true || snapshot.worker !== 'telegraph-fin') fail('complete telegraph-fin snapshot required');
  for (const key of ['deployment', 'modules', 'settings', 'bindings', 'routes', 'domains', 'entry_contract', 'old_binary_routes']) {
    if (!Object.hasOwn(snapshot, key)) fail('snapshot field missing: ' + key);
  }
  if (!snapshot.deployment.id || !Array.isArray(snapshot.deployment.versions) || snapshot.deployment.versions.length !== 1 ||
      snapshot.deployment.versions[0].percentage !== 100 || !snapshot.deployment.versions[0].id) fail('single full active version required; reconcile split first');
  if (!Array.isArray(snapshot.modules) || snapshot.modules.length < 1 || snapshot.modules.length > 1000) fail('invalid module inventory');
  const names = new Set();
  for (const module of snapshot.modules) {
    moduleBytes(module);
    if (names.has(module.name)) fail('duplicate module');
    names.add(module.name);
  }
  const main = snapshot.modules.find(m => m.name === snapshot.main_module);
  if (!main || main.type !== 'esm') fail('ES module entry required');
  const contract = snapshot.entry_contract;
  if (!equal(contract.named_exports, []) || !equal(contract.default_handlers, ['fetch']) ||
      (contract.namespace_unclaimed !== PREFIX && contract.namespace_owned !== PREFIX)) fail('entry export or namespace compatibility unverified');
  if (!snapshot.settings.compatibility_date || !Array.isArray(snapshot.settings.compatibility_flags) || !snapshot.settings.usage_model ||
      !Array.isArray(snapshot.bindings) || !Array.isArray(snapshot.routes) || typeof snapshot.domains.workers_dev !== 'boolean') fail('settings/bindings/routes/domains incomplete');
  if (!Array.isArray(snapshot.old_binary_routes) || snapshot.old_binary_routes.length < 6) fail('old binary route inventory incomplete');
  const urls = new Set();
  for (const route of snapshot.old_binary_routes) {
    if (urls.has(route.url) || !/^https:\/\//.test(route.url) || !/^[a-f0-9]{64}$/.test(route.sha256) || route.content_type !== 'application/wasm' || !route.cache_control.includes('immutable')) fail('invalid old binary route proof');
    urls.add(route.url);
    const module = snapshot.modules.find(m => m.name === route.module_name);
    if (!module || module.type !== 'wasm' || module.sha256 !== route.sha256) fail('old binary not bound to retained module');
  }
  if (Buffer.byteLength(canonical(snapshot)) > 100000000) fail('snapshot size ceiling');
  return fingerprint(snapshot);
}

function moduleRecord(name, type, bytes) {
  return {name, type, content_base64: Buffer.from(bytes).toString('base64'), sha256: sha(bytes)};
}

export function assertPreserved(snapshot, target, retainedRoutes = []) {
  for (const key of ['worker', 'settings', 'bindings', 'routes', 'domains', 'old_binary_routes']) {
    if (!equal(snapshot[key], target[key])) fail('preservation failed: ' + key);
  }
  const names = new Set();
  for (const module of target.modules) {
    moduleBytes(module);
    if (names.has(module.name)) fail('duplicate target module');
    names.add(module.name);
  }
  for (const module of snapshot.modules) {
    const retained = target.modules.find(m => m.name === module.name);
    if (!retained || !equal(retained, module)) fail('old module changed or missing: ' + module.name);
  }
  for (const route of retainedRoutes) {
    const module = target.modules.find(m => m.name === route.module_name);
    if (!module || module.sha256 !== route.sha256 || !target.supported_urls?.includes(route.binary_url) ||
        !target.supported_urls.includes(route.runnable_url)) fail('registered URL lost');
  }
  return true;
}

export function assertNoDrift(plan, current) {
  validateSnapshot(current);
  if (fingerprint(current) !== plan.expected_snapshot_sha256) fail('deployment drift; abort and re-stage');
  return true;
}

export function buildPlan(snapshot, release, files) {
  const expected = validateSnapshot(snapshot);
  if (snapshot.entry_contract.namespace_unclaimed !== PREFIX) fail('namespace already claimed');
  if (release.prefix !== PREFIX || release.publication_state !== 'prospective' || Object.keys(release.families).length !== 6) fail('wrong release');
  const target = clone(snapshot);
  delete target.deployment;
  target.supported_urls = [];
  const records = [], imports = [], binaries = [];
  for (const [family, row] of Object.entries(release.families)) {
    if (!/^[A-Z_]+$/.test(family) || !/^0x[a-f0-9]{64}$/.test(row.keccak)) fail('invalid release family/hash');
    if (row.binary_url !== release.origin + PREFIX + 'binary/' + family + '/' + row.keccak.slice(2) + '.wasm' ||
        row.runnable_url !== release.origin + PREFIX + 'runnable/' + family) fail('release URL mismatch');
    const bytes = files[row.file];
    if (!bytes || sha(bytes) !== row.sha256 || bytes.length !== row.bytes) fail('release bytes changed');
    const name = '_zkcmp_s2_20261118_' + family + '_' + row.sha256 + '.wasm';
    records.push(moduleRecord(name, 'wasm', bytes));
    imports.push(`import wasm_${family} from './${name}';`);
    binaries.push(`${JSON.stringify(family)}: Uint8Array.from(atob(${JSON.stringify(Buffer.from(bytes).toString('base64'))}), c => c.charCodeAt(0))`);
    target.supported_urls.push(row.binary_url, row.runnable_url);
  }
  const handler = files['worker/handler.mjs'];
  if (!handler || sha(handler) !== release.handler_sha256) fail('handler drift');
  const handlerName = '_zkcmp_s2_20261118_handler_' + sha(handler) + '.mjs';
  records.push(moduleRecord(handlerName, 'esm', handler));
  const runtime = files['worker/runtime.mjs'];
  if (!runtime) fail('runtime missing');
  const runtimeText = Buffer.from(runtime).toString().replace("'./handler.mjs'", JSON.stringify('./' + handlerName));
  const runtimeName = '_zkcmp_s2_20261118_runtime_' + sha(runtimeText) + '.mjs';
  records.push(moduleRecord(runtimeName, 'esm', runtimeText));
  const wrapper = ['// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0', '// Copyright (c) 2026 zkasuran',
    `import prior from './${snapshot.main_module}';`, `import {createRecovery, wrapPrior} from './${runtimeName}';`, ...imports,
    `const manifest = ${canonical(release)};`,
    `const modules = {${Object.keys(release.families).map(f => `${JSON.stringify(f)}: wasm_${f}`).join(',')}};`,
    `const binaries = {${binaries.join(',')}};`,
    'const recovery = createRecovery(modules, binaries, manifest);',
    'export default wrapPrior(prior, recovery, manifest.prefix);', ''].join('\n');
  const entry = '_zkcmp_s2_20261118_entry_' + sha(wrapper) + '.mjs';
  records.push(moduleRecord(entry, 'esm', wrapper));
  const repair = wrapper.replace('export default wrapPrior(prior, recovery, manifest.prefix);',
    'export default { fetch(request, env, ctx) { if (new URL(request.url).pathname.startsWith(manifest.prefix)) return recovery.fetch(request); return prior.fetch(request, env, ctx); } };');
  const repairEntry = '_zkcmp_s2_20261118_repair_' + sha(repair) + '.mjs';
  records.push(moduleRecord(repairEntry, 'esm', repair));
  const oldNames = new Set(snapshot.modules.map(m => m.name));
  for (const record of records) if (oldNames.has(record.name)) fail('additive module collision');
  target.modules.push(...records);
  target.main_module = entry;
  target.entry_contract = {named_exports: [], default_handlers: ['fetch'], namespace_owned: PREFIX};
  if (target.modules.reduce((n, m) => n + moduleBytes(m).length, 0) > 64 * 1024 * 1024) fail('aggregate Worker size ceiling');
  assertPreserved(snapshot, target);
  const retained = Object.values(release.families).map(row => ({...row,
    module_name: records.find(m => m.type === 'wasm' && m.sha256 === row.sha256).name}));
  assertPreserved(snapshot, target, retained);
  return {schema: 1, mode: 'additive_plan_only', expected_snapshot_sha256: expected,
    source_deployment: clone(snapshot.deployment), target, target_sha256: fingerprint(target), retained,
    promotion_percentage_required_before_registration: 100,
    rollback: {mode: 'compatibility_repair', target: {...clone(target), main_module: repairEntry},
      note: 'Old paths delegate to the untouched prior entry. New immutable binary and callable URLs stay available.'}};
}

export function planRollback(plan, current, registered, expectedCurrentFingerprint) {
  validateSnapshot(current);
  if (fingerprint(current) !== expectedCurrentFingerprint) fail('rollback deployment drift');
  if (!equal(current.modules, plan.target.modules) || current.main_module !== plan.target.main_module) fail('rollback drift');
  for (const key of ['settings', 'bindings', 'routes', 'domains', 'old_binary_routes']) {
    if (!equal(current[key], plan.target[key])) fail('rollback resource drift');
  }
  for (const route of registered) {
    if (!plan.retained.some(r => equal(r, route))) fail('unknown registered route; retain before rollback');
  }
  const target = clone(plan.rollback.target);
  assertPreserved(current, target, registered);
  return {mode: 'compatibility_repair_only', expected_snapshot_sha256: fingerprint(current), target,
    percentage: 100, retained_registered_urls: registered.flatMap(r => [r.binary_url, r.runnable_url])};
}
