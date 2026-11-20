// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {API,MODEL,handleMedia,mediaType,normalizeAssessment,videoDuration} from './media.mjs';
import {createRemainingIntegration} from './remaining-router.mjs';
const png=new Uint8Array(readFileSync(new URL('./fixtures/owned-rendered.png',import.meta.url)));
const mp4=new Uint8Array(readFileSync(new URL('./fixtures/owned-rendered.mp4',import.meta.url)));
const source='https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/main/owned.png';
const env={MINIMAX_API_KEY:'artificial-media-fixture-key',MINER_DB:{withSession() {return {prepare() {return {bind() {return {async first() {return {calls:1};}};}};}};}}};
const assessment={visual_form:'rendered_graphic',verdict:'Synthetic',confidence:0.8,evidence:['Two flat colour fields and a sharp geometric edge.'],summary:'The input appears to be a rendered graphic.'};
const response=value=>Response.json({model:MODEL,base_resp:{status_code:0},choices:[{finish_reason:'stop',message:{content:JSON.stringify(value)}}]});
test('owned PNG and MP4 have supported metadata',()=>{
  assert.equal(mediaType(png).mime,'image/png');
  assert.equal(mediaType(mp4,true).mime,'video/mp4');
  assert.equal(videoDuration(mp4),2);
});
test('model assessments do not become forensic certainty',()=>{
  assert.equal(normalizeAssessment({...assessment,visual_form:'photographic_or_uncertain',verdict:'Authentic'}).verdict,'Likely authentic');
  assert.equal(normalizeAssessment({...assessment,visual_form:'photographic_or_uncertain',verdict:'Manipulated'}).verdict,'Likely manipulated');
  assert.match(normalizeAssessment(assessment).confidence_basis,/Uncalibrated/);
});
test('rendered graphics cannot inherit an authentic-original verdict',()=>{
  assert.equal(normalizeAssessment({...assessment,verdict:'Authentic'}).verdict,'Synthetic');
  assert.equal(normalizeAssessment({...assessment,verdict:'Likely authentic'}).verdict,'Synthetic');
});
test('invalid labels, confidence and evidence are refused',()=>{
  for(const change of [{verdict:'Guaranteed real'},{confidence:NaN},{confidence:1.1},{confidence:'0.8'},{evidence:[]},{summary:''}])
    assert.throws(()=>normalizeAssessment({...assessment,...change}));
});
test('private and unsupported media URLs fail before fetch',async()=>{
  for(const url of ['https://127.0.0.1/a.png','https://other-domain.org/a.png','https://raw.githubusercontent.com/a.png?key=private','http://raw.githubusercontent.com/a.png']) {
    let calls=0;
    const result=await handleMedia(new Request('https://miner.test/miners/v3/deepfake?media_url='+encodeURIComponent(url)),env,false,async()=>{calls++;});
    assert.ok(result.status>=400);assert.equal(calls,0);
  }
});
test('actual image handler sends exact source bytes rather than an unfetched URL',async()=>{
  const seen=[];
  const fetcher=async(url,options)=>{
    seen.push(url);
    assert.equal(options.redirect,'manual');
    if(url===API) {
      const payload=JSON.parse(options.body);
      assert.equal(payload.model,MODEL);
      assert.ok(payload.messages[0].content[1].image_url.url.startsWith('data:image/png;base64,'));
      assert.ok(!options.body.includes(source));
      return response(assessment);
    }
    return new Response(png);
  };
  const result=await handleMedia(new Request('https://miner.test/miners/v3/deepfake?media_url='+encodeURIComponent(source)),env,false,fetcher);
  assert.equal(result.status,200);
  const body=await result.json();
  assert.equal(body.source_bytes,png.length);assert.match(body.source_sha256,/^[0-9a-f]{64}$/);
  assert.equal(body.verdict,'Synthetic');assert.equal(seen.length,2);
});
test('actual video handler uses the video modality with bounded frame sampling',async()=>{
  const fetcher=async(url,options)=>{
    if(url===API) {
      const part=JSON.parse(options.body).messages[0].content[1];
      assert.equal(part.type,'video_url');assert.equal(part.video_url.fps,1);
      return response(assessment);
    }
    return new Response(mp4);
  };
  const result=await handleMedia(new Request('https://miner.test/miners/v3/video?video_url='+encodeURIComponent(source)),env,true,fetcher);
  assert.equal(result.status,200);assert.equal((await result.json()).video_seconds,2);
});
test('source redirects, excessive bodies and wrong modality cannot reach model',async()=>{
  for(const upstream of [new Response('',{status:302,headers:{location:'https://private.internal/'}}),new Response(new Uint8Array(262145)),new Response(png)]) {
    let modelCalls=0;
    const result=await handleMedia(new Request('https://miner.test/miners/v3/video?video_url='+encodeURIComponent(source)),env,true,async(url)=>{if(url===API)modelCalls++;return upstream;});
    assert.ok(result.status>=400);assert.equal(modelCalls,0);
  }
});
test('budget refusal and missing key do not invoke the provider',async()=>{
  const exhausted={...env,MINER_DB:{withSession(){return {prepare(){return {bind(){return {async first(){return null;}};}};}};}}};
  for(const bindings of [{},exhausted]) {
    let calls=0;
    const result=await handleMedia(new Request('https://miner.test/miners/v3/deepfake?media_url='+encodeURIComponent(source)),bindings,false,async()=>{calls++;});
    assert.ok(result.status>=400);assert.equal(calls,0);
  }
});
test('malformed media containers fail without unchecked offsets',()=>{
  for(let size=0;size<128;size++) {
    const bytes=new Uint8Array(size);bytes.fill(255);
    assert.throws(()=>mediaType(bytes,true));
  }
  assert.throws(()=>videoDuration(mp4.subarray(0,40)));
});
test('existing routes preserve original requests and bindings',async()=>{
  const original={fetch(request,bindings){assert.equal(bindings.marker,42);return new Response(new URL(request.url).pathname);}};
  const router=createRemainingIntegration(original);
  for(const path of ['/health','/media-forensic','/satellite','/unknown']) {
    const result=await router.fetch(new Request('https://miner.test'+path),{marker:42});
    assert.equal(await result.text(),path);
  }
});
