// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import assert from 'node:assert/strict';
import test from 'node:test';
import {API, MODEL, handleChat, messageInput, normalizeReply} from './chat.mjs';
import {createChatIntegration} from './chat-router.mjs';

const url=message => 'https://chat.test/miners/v3/chat?message='+encodeURIComponent(message);
const answer={reply:'Fourteen kilometres is fourteen thousand metres.',verdict:'Likely',brief:'Fourteen kilometres is fourteen thousand metres.'};
const env=(calls=1) => ({MINIMAX_API_KEY:'artificial-test-credential',CHAT_DB:{withSession(mode) {
  assert.equal(mode,'first-primary');
  return {prepare(sql) { assert.match(sql,/WHERE calls < 20 RETURNING calls/); return {async first() { return calls === null ? null : {calls}; }}; }};
}}});
const fetcher=(body) => async (endpoint,options) => {
  assert.equal(endpoint,API);
  assert.equal(options.method,'POST');
  assert.equal(options.redirect,'manual');
  assert.equal(JSON.parse(options.body).model,MODEL);
  assert.equal(JSON.parse(options.body).max_completion_tokens,512);
  return Response.json(body);
};
const upstream=value => ({model:MODEL,base_resp:{status_code:0},choices:[{finish_reason:'stop',message:{content:JSON.stringify(value)}}]});

test('non-prediction answers cannot inherit an unrelated Likely field', () => {
  assert.equal(normalizeReply(answer).verdict,'Uncertain');
  assert.equal(normalizeReply({...answer,reply:'Likely: rainfall will continue.'}).verdict,'Likely');
  assert.equal(normalizeReply({...answer,reply:'Unlikely: rainfall will continue.'}).verdict,'Uncertain');
});
test('UTF-8 contract boundary and malformed parameters', () => {
  assert.equal(messageInput(new Request(url('a'.repeat(256)))).length,256);
  assert.equal(messageInput(new Request(url(' x\n'))),'x');
  for (const message of ['a'.repeat(257),'🙂'.repeat(65),'']) {
    assert.throws(() => messageInput(new Request(url(message))));
  }
  assert.throws(() => messageInput(new Request(url('a')+'&message=b')));
  assert.throws(() => messageInput(new Request(url('a')+'&unknown=b')));
  assert.throws(() => messageInput(new Request(url('a\nb'))));
});
test('overlong, empty and invalid-enum replies fail closed', () => {
  for (const patch of [{brief:'x'.repeat(81)},{brief:'x\ny'},{reply:''},{verdict:'Authentic'},{brief:'🙂'.repeat(80)}])
    assert.throws(() => normalizeReply({...answer,...patch}));
});
test('actual handler consumes a reserved budget and produces canonical fields',async () => {
  const response=await handleChat(new Request(url('Convert kilometres to metres')),env(),fetcher(upstream(answer)));
  assert.equal(response.status,200);
  const body=await response.json();
  assert.equal(body.reply,answer.reply);
  assert.equal(body.verdict,'Uncertain');
  assert.equal(body.ai_generated,true);
  assert.equal(response.headers.get('cache-control'),'no-store');
});
test('daily cap and missing persistent binding stop before any provider call',async () => {
  for (const config of [env(null),env(21),{MINIMAX_API_KEY:'artificial-test-credential'}]) {
    let calls=0;
    const response=await handleChat(new Request(url('Hello')),config,async () => {calls++;});
    assert.ok(response.status >= 400);
    assert.equal(calls,0);
  }
});
test('HTTP-200 provider errors, wrong model and truncated output cannot pass',async () => {
  for (const body of [{...upstream(answer),base_resp:{status_code:1008}},
    {...upstream(answer),model:'wrong'},
    {...upstream(answer),choices:[{finish_reason:'length',message:{content:JSON.stringify(answer)}}]}]) {
    const response=await handleChat(new Request(url('Hello')),env(),fetcher(body));
    assert.equal(response.status,502);
  }
});
test('non-JSON and oversized upstream bodies are refused',async () => {
  for (const response of [new Response('<html>wrong</html>',{headers:{'content-type':'text/html'}}),
    new Response('x'.repeat(65537),{headers:{'content-type':'application/json'}})]) {
    const result=await handleChat(new Request(url('Hello')),env(),async () => response);
    assert.equal(result.status,502);
  }
});
test('redirect responses are refused without forwarding the provider credential',async () => {
  let calls=0;
  const result=await handleChat(new Request(url('Hello')),env(),async (_url,options) => {
    calls++;
    assert.equal(options.redirect,'manual');
    return new Response('',{status:302,headers:{location:'https://unrelated.test/key'}});
  });
  assert.equal(result.status,502);
  assert.equal(calls,1);
});
test('provider credential cannot be returned in an answer or error',async () => {
  const result=await handleChat(new Request(url('Hello')),env(),fetcher(upstream({...answer,reply:'artificial-test-credential'})));
  assert.equal(result.status,502);
  assert.ok(!(await result.text()).includes('artificial-test-credential'));
});
test('the router delegates existing paths with the exact original request and bindings',async () => {
  const original={fetch(request,bindings,context) {
    assert.equal(bindings.marker,'preserved');
    assert.equal(context.marker,'preserved');
    return new Response(new URL(request.url).pathname);
  }};
  const router=createChatIntegration(original);
  for (const path of ['/','/health','/grammar','/summary','/unknown']) {
    const response=await router.fetch(new Request('https://chat.test'+path),{marker:'preserved'},{marker:'preserved'});
    assert.equal(await response.text(),path);
  }
});
