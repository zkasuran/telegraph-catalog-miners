// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import assert from 'node:assert/strict';
import test from 'node:test';
import {candidates,confirmPost,postID} from './indexed-x.mjs';
const data={model:'MiniMax-M3.1-Flash',stop_reason:'pause_turn',base_resp:{status_code:0},content:[
  {type:'server_tool_use',name:'web_search',id:'owned-fixture-search'},
  {type:'web_search_tool_result',tool_use_id:'owned-fixture-search',content:[
    {type:'web_search_result',url:'https://x.com/Example/status/123456789',content:'source fixture'},
    {type:'web_search_result',url:'https://other.org/news',content:'Link https://twitter.com/Example/status/987654321'}]},
  {type:'text',text:'Invented https://x.com/Example/status/999999999'}]};
test('only retrieved source URLs are candidates, not model text',()=>{
  assert.deepEqual(candidates(data).map(item=>item.id),['123456789','987654321']);
  assert.equal(postID('http://x.com/Example/status/123'),null);
  assert.equal(postID('https://other.org/Example/status/123'),null);
});
test('completed search results remain usable at a partial server-search limit',()=>{
  assert.equal(candidates(data).length,2);
  assert.throws(()=>candidates({...data,content:[{type:'text',text:'No search was executed'}]}));
});
test('current publisher identity is verified without returning embedded text',async()=>{
  let calls=0;
  const result=await confirmPost({id:'123456789',url:'https://x.com/Example/status/123456789'},async()=>{
    calls++;return Response.json({url:'https://twitter.com/Example/status/123456789',html:'<a href="https://x.com/Example/status/123456789">owned fixture</a>'});
  },AbortSignal.timeout(1000));
  assert.equal(result,true);assert.equal(calls,1);
  assert.equal(await confirmPost({id:'123',url:'https://x.com/Example/status/123'},async()=>new Response('',{status:404}),AbortSignal.timeout(1000)),false);
});
