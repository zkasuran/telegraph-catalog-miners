// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
export class Refusal extends Error {
  constructor(code,status=422) { super(code); this.code=code; this.status=status; }
}
export function requireValue(ok,code,status) { if (!ok) throw new Refusal(code,status); }
export const encoder=new TextEncoder();
export function input(request,fields) {
  requireValue(request.method==='GET','method_not_allowed',405);
  const url=new URL(request.url);
  requireValue(url.href.length<=4096,'request_too_large',413);
  requireValue([...url.searchParams.keys()].every(key=>fields.includes(key)&&url.searchParams.getAll(key).length===1),'unknown_or_duplicate_parameter');
  return url.searchParams;
}
export function text(value,name,maxBytes=256) {
  requireValue(typeof value==='string','missing_'+name);
  value=value.normalize('NFC').trim();
  requireValue(value.length>0&&!/[\x00-\x1f\x7f]/u.test(value)&&encoder.encode(value).length<=maxBytes,'invalid_'+name);
  return value;
}
export function publicURL(value,hosts=null) {
  let url;
  try { url=new URL(value); } catch { throw new Refusal('invalid_source_url'); }
  requireValue(url.protocol==='https:'&&!url.username&&!url.password&&!url.port&&!url.hash,'public_https_source_required');
  requireValue(url.hostname.includes('.')&&!url.hostname.includes(':')&&!/^[\d.]+$/.test(url.hostname)&&!/(?:^|\.)(?:local|localhost|internal|test|invalid|example|onion)$/.test(url.hostname),'public_source_hostname_required');
  if (hosts) requireValue(hosts.includes(url.hostname),'unsupported_media_origin',403);
  return url;
}
export async function bounded(promise,ms=45000) {
  let timer;
  try { return await Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Refusal('source_timeout',504)),ms);})]); }
  finally { clearTimeout(timer); }
}
export async function bytes(response,maxBytes) {
  requireValue(response.status===200,'source_http_error',502);
  requireValue(Number(response.headers.get('content-length'))<=maxBytes,'source_too_large',413);
  const reader=response.body?.getReader();
  requireValue(reader,'source_empty_body',502);
  const chunks=[];
  let size=0;
  try {
    for (;;) {
      const {done,value}=await reader.read();
      if (done) break;
      size+=value.length;
      requireValue(size<=maxBytes,'source_too_large',413);
      chunks.push(value);
    }
  } finally { await reader.cancel().catch(()=>{}); reader.releaseLock(); }
  const result=new Uint8Array(size);
  let offset=0;
  for (const chunk of chunks) {result.set(chunk,offset);offset+=chunk.length;}
  return result;
}
export async function jsonResponse(response,maxBytes=65536) {
  requireValue(response.headers.get('content-type')?.includes('json'),'source_not_json',502);
  try { return JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(await bytes(response,maxBytes))); }
  catch(error) { if (error instanceof Refusal) throw error; throw new Refusal('source_invalid_json',502); }
}
export async function quota(env,name,max=20) {
  requireValue(env.MINER_DB?.withSession,'persistent_budget_not_bound',503);
  const sql=`INSERT INTO s2_miner_quota(feature,day,calls) VALUES (?,strftime('%Y-%m-%d','now'),1)
    ON CONFLICT(feature,day) DO UPDATE SET calls=calls+1 WHERE calls < ? RETURNING calls`;
  const row=await bounded(env.MINER_DB.withSession('first-primary').prepare(sql).bind(name,max).first(),6000);
  requireValue(row&&Number.isSafeInteger(row.calls)&&row.calls>=1&&row.calls<=max,'daily_source_budget',429);
}
export function reply(value,status=200) {
  return Response.json(value,{status,headers:{'cache-control':'no-store','x-content-type-options':'nosniff',
    'content-security-policy':"default-src 'none'; frame-ancestors 'none'",'referrer-policy':'no-referrer'}});
}
export async function guarded(task) {
  try {return reply(await task());}
  catch(error) {return reply({error:error instanceof Refusal?error.code:'source_unavailable'},error instanceof Refusal?error.status:503);}
}
export const hex=bytes=>[...new Uint8Array(bytes)].map(value=>value.toString(16).padStart(2,'0')).join('');
