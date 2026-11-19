// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import path from "node:path";
import {createHash} from "node:crypto";
import {createHandler, parseExact} from "./worker/handler.mjs";

const root = path.dirname(fileURLToPath(import.meta.url));
const release = JSON.parse(readFileSync(path.join(root,"manifest.json")));
const families = ["CRYPTO_PRICE", "CURRENCY_EXCHANGE", "FX_NOW", "SSL_VERIFICATION", "STOCK_PRICE", "WEATHER_CHECK"];
const vectors = [
  [0,0,9007199254740993n,9007199254740993n,10000],
  [0,0,9007199254740993n,9007199254740992n,0],
  [2,50,10000n,10099n,100],
  [2,50,10099n,10000n,200],
  [1,20,-100n,-80n,5000],
  [1,20,-100n,-61n,250],
  [1,20,-100n,-60n,0],
  [0,0,-9223372036854775808n,-9223372036854775808n,10000],
  [1,1,-9223372036854775808n,-9223372036854775807n,5000],
  [2,50,-9223372036854775808n,-9223372036854775807n,0],
  [2,50,9223372036854775807n,-9223372036854775808n,10000],
  [1,1,9223372036854775807n,-1n,"error"],
  [1,9223372036854775807n,1n,1n,1],
  [4294967298,50,10000n,10099n,100],
  [2,0,100000n,100001n,10000]
];
const wire = v => `{"ground_truth_value":${v[2]},"miner_value":${v[3]},"spec":[${v[0]},${v[1]}]}`;
const payload = vs => `{"batch":[${vs.map(wire).join(",")}]}`;
const request = (f, body) => new Request(`https://unused.invalid/runnable/${f}`, {method:"POST",headers:{"content-type":"application/json"},body});
const instances = {}, binaries = {}, manifest = {version:"private-math-verification",families:{}};
for (const f of families) {
  binaries[f] = readFileSync(path.join(root,release.families[f].file));
  instances[f] = (await WebAssembly.instantiate(binaries[f])).instance;
  manifest.families[f] = {keccak:"0x" + "a".repeat(64)};
}
const handler = createHandler(instances,binaries,manifest);
let cases=0, refusals=0;
for (const f of families) {
  for (const vs of [vectors, Array.from({length:100},(_,i)=>vectors[i%vectors.length])]) {
    const r = await handler.fetch(request(f,payload(vs)));
    assert.equal(r.status,200);
    const b=await r.json();assert.equal(b.results.length,vs.length);
    for (let i=0;i<vs.length;i++) {
      cases++;
      if (vs[i][4]==="error") {assert.ok(b.results[i].error);refusals++;}
      else {assert.equal(b.results[i].score,vs[i][4]);assert.equal(b.results[i].error,undefined);}
    }
  }
  console.log(f, "115 ordered handler cases passed; exact SHA256",createHash("sha256").update(binaries[f]).digest("hex"));
}
const badBodies = [
  '{"batch":[{"ground_truth_value":0,"miner_value":0,"spec":[3,0]}]}',
  '{"batch":[{"ground_truth_value":9223372036854775808,"miner_value":1,"spec":[0,0]}]}',
  '{"batch":[{"ground_truth_value":-9223372036854775809,"miner_value":1,"spec":[0,0]}]}',
  '{"batch":[{"ground_truth_value":1.0,"miner_value":1,"spec":[0,0]}]}',
  '{"batch":[{"ground_truth_value":1e0,"miner_value":1,"spec":[0,0]}]}',
  '{"batch":[{"ground_truth_value":true,"miner_value":1,"spec":[0,0]}]}',
  '{"batch":[{"ground_truth_value":1,"miner_value":1,"spec":[0]}]}',
  '{"batch":[]}', payload(Array.from({length:101},()=>vectors[0])),
  '{"batch":[{"ground_truth_value":01,"miner_value":1,"spec":[0,0]}]}'
];
for (const body of badBodies) {const r=await handler.fetch(request("FX_NOW",body));assert.equal(r.status,400);assert.ok((await r.json()).error);}
const precision = '{"batch":[{"ground_truth_value":9007199254740993,"miner_value":9007199254740992,"spec":[0,0]}]}';
assert.equal(parseExact(precision).batch[0].ground_truth_value,"9007199254740993");
const rounded=JSON.parse(precision);
assert.equal(rounded.batch[0].ground_truth_value,rounded.batch[0].miner_value);
const mutated = (await handler.fetch(request("FX_NOW",JSON.stringify(rounded))));
assert.equal((await mutated.json()).results[0].score,10000);
const correct = await handler.fetch(request("FX_NOW",precision));
assert.equal((await correct.json()).results[0].score,0);
console.log("precision failure control: plain JSON.parse changes correct score 0 to 10000");
console.log(JSON.stringify({status:"PASS",cases,ordered_batches:12,per_item_overflow_refusals:refusals,invalid_body_refusals:badBodies.length,precision_mutation_caught:true,no_network:true}));
