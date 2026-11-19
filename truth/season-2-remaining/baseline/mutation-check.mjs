// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Only scratch copies of the wrapper are mutated. Imported modules stay intact.
import { readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
const source = readFileSync(new URL('./dispatcher.mjs', import.meta.url), 'utf8');
const changes = [
  ['wrong-balance-path', "forwardPath(request, '/metric')", "forwardPath(request, '/balance')"],
  ['caller-host-overwritten', 'url.pathname = pathname;', "url.pathname = pathname; url.hostname = 'unexpected.example';"],
  ['original-env-lost', 'handlers.original.fetch(request, env, ctx)', 'handlers.original.fetch(request, {}, ctx)'],
  ['original-request-cloned', 'handlers.original.fetch(request, env, ctx)', 'handlers.original.fetch(new Request(request), env, ctx)'],
];
const findings = [];
for (const [name, old, replacement] of [['control', null, null], ...changes]) {
  if (old) assert.equal(source.split(old).length - 1, 1);
  const variant = (old ? source.replace(old, replacement) : source).replaceAll("'./sources/", "'../sources/").replace("'./workers-fetch.mjs'", "'../workers-fetch.mjs'");
  const file = new URL('./.test-results/' + name + '.mjs', import.meta.url);
  writeFileSync(file, variant);
  const probe = `
    import assert from 'node:assert/strict';
    import { createDispatcher } from ${JSON.stringify(file.href)};
    const env = { marker: 'env' }, ctx = { marker: 'ctx' };
    let observed;
    const spy = { fetch(request, e, c) { assert.equal(e, env); assert.equal(c, ctx); observed = request; return new Response('ok'); } };
    const d = createDispatcher({ original: spy, company: spy, balance: spy, hashprice: spy });
    const legacy = new Request('https://caller.example/health');
    await d.fetch(legacy, env, ctx); assert.equal(observed, legacy);
    const request = new Request('https://caller.example/truth/v1/balance?x=a%2Fb', { method: 'POST', body: 'exact\\n' });
    await d.fetch(request, env, ctx);
    assert.equal(observed.url, 'https://caller.example/metric?x=a%2Fb');
    assert.equal(observed.method, 'POST'); assert.equal(await observed.text(), 'exact\\n');
  `;
  const result = spawnSync(process.execPath, ['--input-type=module', '-e', probe], { encoding: 'utf8' });
  assert.equal(result.status === 0, name === 'control', name + ' mutation verdict');
  findings.push({ mutation: name, returncode: result.status, caught: name !== 'control' });
}
writeFileSync(new URL('./.test-results/mutations.json', import.meta.url), JSON.stringify(findings, null, 2) + '\n');
console.log('MUTATION GATE PASS: control passes, 4 wrapper regressions caught');
