// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {performance} from 'node:perf_hooks';
import {createRecovery} from './worker/runtime.mjs';
import {MAX_BODY} from './worker/handler.mjs';

const root = new URL('./', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('manifest.json', root)));
const modules = {}, binaries = {};
for (const [family, row] of Object.entries(manifest.families)) {
  binaries[family] = readFileSync(new URL(row.file, root));
  modules[family] = await WebAssembly.compile(binaries[family]);
  assert.equal(WebAssembly.Module.imports(modules[family]).length, 0);
  assert.deepEqual(WebAssembly.Module.exports(modules[family]).map(e => e.name).sort(),
    ['memory', 'imw_alloc', 'imw_free', 'imw_score_deterministic'].sort());
}
const recovery = createRecovery(modules, binaries, manifest);
const url = manifest.families.FX_NOW.runnable_url;
const request = (body, headers = {}, options = {}) => new Request(url, {method: 'POST',
  headers: {'content-type': 'application/json', ...headers}, body, ...options});
let refused = 0;
for (const body of ['{', '{"batch":null}', '{"batch":[null]}', '{"batch":[1]}', '{"batch":[{}]}',
  '{"batch":[{"ground_truth_value":"no","miner_value":1,"spec":[0,0]}]}',
  '{"batch":[{"ground_truth_value":1,"miner_value":[],"spec":[0,0]}]}']) {
  const response = await recovery.fetch(request(body));
  assert.equal(response.status, 400); refused++;
}
assert.equal((await recovery.fetch(request('{}', {'content-type': 'text/plain'}))).status, 415); refused++;
assert.equal((await recovery.fetch(request('{}', {'content-length': String(MAX_BODY + 1)}))).status, 413); refused++;
const large = new ReadableStream({start(controller) { controller.enqueue(new Uint8Array(MAX_BODY)); controller.enqueue(new Uint8Array(1)); controller.close(); }});
assert.equal((await recovery.fetch(request(large, {}, {duplex: 'half'}))).status, 413); refused++;
const invalidUtf8 = new ReadableStream({start(controller) { controller.enqueue(new Uint8Array([0xff])); controller.close(); }});
assert.equal((await recovery.fetch(request(invalidUtf8, {}, {duplex: 'half'}))).status, 400); refused++;
const slow = new ReadableStream({pull() { return new Promise(() => {}); }});
const start = performance.now();
assert.equal((await recovery.fetch(request(slow, {}, {duplex: 'half'}))).status, 408); refused++;
const timeoutMs = performance.now() - start;
assert.ok(timeoutMs >= 4900 && timeoutMs < 6500);
const cpuMeasurements = {}, memory = {};
for (const family of Object.keys(modules)) {
  const body = '{"batch":[' + Array.from({length: 100}, (_, i) =>
    `{"ground_truth_value":9007199254740993,"miner_value":${i % 2 ? '9007199254740992' : '9007199254740993'},"spec":[0,0]}`).join(',') + ']}';
  const start = performance.now();
  const response = await recovery.fetch(new Request(manifest.families[family].runnable_url, {method: 'POST', headers: {'content-type': 'application/json'}, body}));
  const results = (await response.json()).results;
  assert.equal(response.status, 200); assert.equal(results.length, 100);
  for (let i = 0; i < 100; i++) assert.equal(results[i].score, i % 2 ? 0 : 10000);
  cpuMeasurements[family] = Number((performance.now() - start).toFixed(3));
  const instance = recovery.initialized[family];
  const before = instance.exports.memory.buffer.byteLength;
  assert.equal(instance.exports.imw_alloc(4097), 0);
  assert.equal(instance.exports.memory.buffer.byteLength, before);
  assert.ok(before <= 2 * 1024 * 1024);
  memory[family] = before;
}
assert.equal((await recovery.fetch(new Request(manifest.origin + manifest.prefix + 'binary/FX_NOW/' + '0'.repeat(64) + '.wasm'))).status, 404);
console.log(JSON.stringify({status: 'PASS', invalid_body_and_resource_refusals: refused,
  body_limit_bytes: MAX_BODY, max_batch: 100, body_deadline_ms: 5000, observed_timeout_ms: Number(timeoutMs.toFixed(3)),
  wasm_memory_bytes_per_family: memory, local_node_100_question_elapsed_ms: cpuMeasurements,
  cloudflare_cpu_startup_memory_headroom: 'unverified', no_network: true}));
