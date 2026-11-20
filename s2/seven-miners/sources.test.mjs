// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// All source records here are independent artificial fixtures, not live observations.
import assert from 'node:assert/strict';
import test from 'node:test';
import {encoder,hex} from './network.mjs';
import {handlePackage,handlePort,handleX,normalizePackage,normalizePort,normalizeX,verifyPortEnvelope} from './sources.mjs';
const NOW=Date.parse('2026-11-20T12:00:00Z');
const port={version:1,port:'FIHEL',wait_definition:'mean_completed_anchorage_wait_hours',wait_unit:'hours',wait_hours:3.5,vessels_waiting:7,congestion_level:'moderate',as_of:'2026-11-20T11:59:30Z'};
const SECRET='artificial-publisher-fixture-key-2026';
async function envelope(value) {
  const raw=encoder.encode(JSON.stringify(value));
  const key=await crypto.subtle.importKey('raw',encoder.encode(SECRET),{name:'HMAC',hash:'SHA-256'},false,['sign']);
  return {payload:btoa(String.fromCharCode(...raw)),signature:hex(await crypto.subtle.sign('HMAC',key,raw))};
}
test('X discovery returns IDs and links without forbidden text redistribution',()=>{
  const result=normalizeX({data:[{id:'112233445566',text:'must not be redistributed',author_id:'998877'}],meta:{result_count:1}});
  assert.deepEqual(result.post_ids,['112233445566']);
  assert.ok(!JSON.stringify(result).includes('must not be redistributed'));
  assert.equal(result.urls[0],'https://x.com/i/status/112233445566');
});
test('X errors, count mismatches and duplicate IDs fail closed',()=>{
  for(const value of [{meta:{result_count:1}},{data:[{id:'a'}],meta:{result_count:1}},
    {data:[{id:'1'},{id:'1'}],meta:{result_count:2}},{errors:[{}],meta:{result_count:0}}]) assert.throws(()=>normalizeX(value));
});
test('X, carrier and port unconfigured paths return real unavailability without calls',async()=>{
  const cases=[[handleX,'/x-search?query=shipping'],[handlePackage,'/package?tracking_number=ARTIFICIAL123'],[handlePort,'/port?port=FIHEL']];
  for(const [handler,path] of cases) {
    let calls=0;
    const result=await handler(new Request('https://miner.test'+path),{},async()=>{calls++;});
    assert.equal(result.status,503);assert.equal(calls,0);
  }
});
test('tracking response identity and privacy are bound to the requested shipment',()=>{
  const value={shipments:[{id:'ARTIFICIAL123',service:'express',status:{statusCode:'delivered',timestamp:'2026-11-19T12:00:00',description:'private name'},destination:{address:{streetAddress:'private street'}}}]};
  const result=normalizePackage(value,'ARTIFICIAL123');
  assert.equal(result.status_code,'delivered');
  assert.ok(!JSON.stringify(result).includes('private'));
  assert.throws(()=>normalizePackage(value,'OTHER123'));
  assert.throws(()=>normalizePackage({...value,shipments:[...value.shipments,...value.shipments]},'ARTIFICIAL123'));
});
test('carrier needs both source consent and a current per-shipment grant',async()=>{
  const env={DHL_API_KEY:'artificial-key',DHL_PUBLIC_STATUS_CONSENT_CONFIRMED:true,MINER_DB:{withSession(){return {prepare(){return {bind(){return {async first(){return null;}};}};}};}}};
  let calls=0;
  const response=await handlePackage(new Request('https://miner.test/package?tracking_number=ARTIFICIAL123'),env,async()=>{calls++;});
  assert.equal(response.status,403);assert.equal(calls,0);
});
test('authenticated port observations retain the actual hours and source meaning',async()=>{
  const result=await verifyPortEnvelope(await envelope(port),SECRET,'FIHEL',NOW);
  assert.equal(result.wait_hours,3.5);assert.equal(result.vessels_waiting,7);
  assert.match(result.payload_sha256,/^[0-9a-f]{64}$/);
});
test('wrong port, ETA proxies, stale times and impossible dates are rejected',()=>{
  for(const change of [{port:'NLRTM'},{wait_definition:'eta_minus_ata'},{wait_unit:'minutes'},
    {as_of:'2026-11-20T11:50:00Z'},{as_of:'2026-11-20T12:00:01Z'},{as_of:'2026-02-30T12:00:00Z'},
    {wait_hours:-1},{wait_hours:NaN},{vessels_waiting:1.1}]) assert.throws(()=>normalizePort({...port,...change},'FIHEL',NOW));
});
test('tampered and wrong-key port signatures fail',async()=>{
  const signed=await envelope(port);
  await assert.rejects(()=>verifyPortEnvelope({...signed,signature:'0'.repeat(64)},SECRET,'FIHEL',NOW));
  await assert.rejects(()=>verifyPortEnvelope(signed,'wrong-artificial-publisher-key-2026','FIHEL',NOW));
  await assert.rejects(()=>verifyPortEnvelope({...signed,payload:btoa('not JSON')},SECRET,'FIHEL',NOW));
});
