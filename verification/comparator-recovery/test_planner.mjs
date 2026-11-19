// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {buildPlan, assertNoDrift, assertPreserved, planRollback, fingerprint, validateSnapshot, PREFIX} from './planner.mjs';
import {wrapPrior, createRecovery} from './worker/runtime.mjs';

const root = new URL('./', import.meta.url);
const release = JSON.parse(readFileSync(new URL('manifest.json', root)));
const files = {}, modules = {}, binaries = {};
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const module = (name, type, bytes) => ({name, type, content_base64: bytes.toString('base64'), sha256: hash(bytes)});
for (const [family, row] of Object.entries(release.families)) {
  const bytes = readFileSync(new URL(row.file, root));
  files[row.file] = bytes;
  binaries[family] = bytes;
  modules[family] = await WebAssembly.compile(bytes);
}
for (const path of ['worker/runtime.mjs', 'worker/handler.mjs']) files[path] = readFileSync(new URL(path, root));
const snapshot = {schema: 1, complete: true, worker: 'telegraph-fin',
  deployment: {id: 'fixture-deployment', versions: [{id: 'fixture-version', percentage: 100}]},
  main_module: 'index.mjs', modules: [module('index.mjs', 'esm', Buffer.from('export default {fetch(r){return new Response(r.url)}};'))],
  settings: {compatibility_date: '2026-11-17', compatibility_flags: ['nodejs_compat'], usage_model: 'standard', observability: {enabled: true}},
  bindings: [{name: 'KV', type: 'kv_namespace', namespace_id: 'fixture-resource'}, {name: 'TOKEN', type: 'secret_text'}],
  routes: [{pattern: 'fixture.invalid/*', zone_id: 'fixture-zone'}], domains: {workers_dev: true, previews_enabled: false},
  entry_contract: {named_exports: [], default_handlers: ['fetch'], namespace_unclaimed: PREFIX}, old_binary_routes: []};
for (const [family, bytes] of Object.entries(binaries)) {
  const name = 'old_' + family + '.wasm';
  snapshot.modules.push(module(name, 'wasm', bytes));
  snapshot.old_binary_routes.push({url: 'https://fixture.invalid/old/' + family + '.wasm', module_name: name,
    sha256: hash(bytes), content_type: 'application/wasm', cache_control: 'public, max-age=31536000, immutable'});
}
const before = structuredClone(snapshot);
const plan = buildPlan(snapshot, release, files);
assert.deepEqual(snapshot, before);
assertNoDrift(plan, snapshot);
assertPreserved(snapshot, plan.target, plan.retained);
assert.equal(plan.promotion_percentage_required_before_registration, 100);
let controls = 0;
for (const mutate of [s => s.deployment.id += 'changed', s => s.deployment.versions[0].id += 'changed',
  s => s.bindings[0].namespace_id += 'changed', s => s.settings.observability.enabled = false,
  s => s.routes.push({pattern: '*'}), s => s.domains.workers_dev = false,
  s => s.modules[0].content_base64 = Buffer.from('different').toString('base64')]) {
  const changed = structuredClone(snapshot); mutate(changed);
  assert.throws(() => assertNoDrift(plan, changed)); controls++;
}
for (const mutate of [s => s.complete = false, s => s.modules.push(s.modules[0]),
  s => s.entry_contract.named_exports = ['DurableObject'], s => s.entry_contract.default_handlers.push('scheduled'),
  s => s.deployment.versions[0].percentage = 50, s => s.old_binary_routes.pop(),
  s => s.old_binary_routes[0].sha256 = 'a'.repeat(64)]) {
  const changed = structuredClone(snapshot); mutate(changed);
  assert.throws(() => validateSnapshot(changed)); controls++;
}
const collision = structuredClone(snapshot); collision.modules.push(plan.target.modules.at(-1));
assert.throws(() => buildPlan(collision, release, files)); controls++;
const tampered = {...files, [Object.values(release.families)[0].file]: Buffer.from('wrong')};
assert.throws(() => buildPlan(snapshot, release, tampered)); controls++;
const current = {...structuredClone(plan.target), deployment: {id: 'promoted-fixture', versions: [{id: 'new-fixture', percentage: 100}]}};
const repair = planRollback(plan, current, [plan.retained[2]], fingerprint(current));
assert.notEqual(repair.target.main_module, current.main_module);
assert.equal(repair.retained_registered_urls.length, 2);
assertPreserved(current, repair.target, plan.retained);
const missing = structuredClone(repair.target); missing.modules = missing.modules.filter(m => m.name !== plan.retained[2].module_name);
assert.throws(() => assertPreserved(current, missing, plan.retained)); controls++;
const drift = structuredClone(current); drift.deployment.id = 'other-writer';
assert.throws(() => planRollback(plan, drift, plan.retained, fingerprint(current))); controls++;
assert.throws(() => planRollback(plan, current, [{...plan.retained[0], runnable_url: 'https://unknown.invalid'}], fingerprint(current))); controls++;

const recovery = createRecovery(modules, binaries, release);
const env = {marker: 1}, ctx = {marker: 2};
let delegated = 0;
const prior = {fetch(request, actualEnv, actualCtx) { assert.equal(actualEnv, env); assert.equal(actualCtx, ctx); delegated++; return request; }};
const wrapper = wrapPrior(prior, recovery, PREFIX);
function evaluateEntry(name) {
  const source = Buffer.from(plan.target.modules.find(m => m.name === name).content_base64, 'base64').toString();
  const body = source.replace(/^import .+;\n/gm, '').replace('export default ', 'return ');
  const familyNames = Object.keys(modules);
  return new Function('prior', 'createRecovery', 'wrapPrior', ...familyNames.map(f => 'wasm_' + f), body)
    (prior, createRecovery, wrapPrior, ...familyNames.map(f => modules[f]));
}
const generated = evaluateEntry(plan.target.main_module);
const generatedRepair = evaluateEntry(plan.rollback.target.main_module);
for (const entry of [generated, generatedRepair]) {
  const request = new Request('https://fixture.invalid/_zkcmp_20261117/health');
  assert.equal(entry.fetch(request, env, ctx), request);
  const binary = await entry.fetch(new Request(release.families.FX_NOW.binary_url), env, ctx);
  assert.equal(hash(Buffer.from(await binary.arrayBuffer())), release.families.FX_NOW.sha256);
  const callable = await entry.fetch(new Request(release.families.FX_NOW.runnable_url, {method: 'POST',
    headers: {'content-type': 'application/json'},
    body: '{"batch":[{"ground_truth_value":9007199254740993,"miner_value":9007199254740992,"spec":[0,0]}]}'}), env, ctx);
  assert.equal((await callable.json()).results[0].score, 0);
}
for (const path of ['/', '/health', '/_zkcmp_20261117/health', '/_zkcmp_20261117/binary/FX_NOW/old.wasm',
  '/_zkcmp_s2_20261118', '/_zkcmp_s2_20261118_extra/', '/miners/weather']) {
  const request = new Request('https://fixture.invalid' + path);
  assert.equal(wrapper.fetch(request, env, ctx), request);
}
assert.equal(delegated, 9);
assert.deepEqual(Object.keys(recovery.initialized), []);
const health = await wrapper.fetch(new Request(release.origin + PREFIX + 'health'), env, ctx);
assert.equal((await health.json()).version, release.version);
assert.deepEqual(Object.keys(recovery.initialized), []);
for (const row of Object.values(release.families)) {
  const r = await wrapper.fetch(new Request(row.binary_url), env, ctx);
  assert.equal(r.status, 200); assert.equal(r.headers.get('content-type'), 'application/wasm');
  assert.match(r.headers.get('cache-control'), /immutable/);
  assert.equal(hash(Buffer.from(await r.arrayBuffer())), row.sha256);
}
assert.deepEqual(Object.keys(recovery.initialized), []);
const fx = release.families.FX_NOW;
const r = await wrapper.fetch(new Request(fx.runnable_url, {method: 'POST', headers: {'content-type': 'application/json'},
  body: '{"batch":[{"ground_truth_value":10000,"miner_value":10099,"spec":[2,50]}]}'}), env, ctx);
assert.equal((await r.json()).results[0].score, 100);
assert.deepEqual(Object.keys(recovery.initialized), ['FX_NOW']);
console.log(JSON.stringify({status: 'PASS', negative_controls: controls, old_path_identity_delegations: delegated,
  exact_binary_responses: 6, lazy_family_initialization: true, rollback_keeps_registered_urls: true,
  generated_entry_and_repair_executed_in_memory: true,
  full_snapshot_fixture_only: true, authenticated_inventory_tested: false}));
