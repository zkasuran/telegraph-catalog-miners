// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import {bounded,encoder,guarded,hex,input,jsonResponse,publicURL,quota,requireValue,text} from './network.mjs';
export const X_API='https://api.x.com/2/tweets/search/recent';
export const DHL_API='https://api-eu.dhl.com/track/shipments';
const enabled=value=>value===true||value==='true';
export function timestamp(value) {
  requireValue(typeof value==='string','invalid_source_time',502);
  const match=/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,6})?Z$/.exec(value);
  requireValue(match,'invalid_source_time',502);
  const [year,month,day,hour,minute,second]=match.slice(1).map(Number);
  requireValue(year>=1970&&month>=1&&month<=12&&day>=1&&day<=new Date(Date.UTC(year,month,0)).getUTCDate()&&hour<24&&minute<60&&second<60,'invalid_source_time',502);
  return Date.parse(value);
}
export function normalizeX(data) {
  requireValue(data&&(!data.errors||data.errors.length===0)&&data.meta&&Number.isSafeInteger(data.meta.result_count),'invalid_x_response',502);
  const rows=data.data||[];
  requireValue(Array.isArray(rows)&&rows.length<=10&&rows.length===data.meta.result_count,'invalid_x_result_count',502);
  const ids=rows.map(row=>row.id);
  requireValue(ids.every(id=>typeof id==='string'&&/^[0-9]{1,19}$/.test(id))&&new Set(ids).size===ids.length,'invalid_x_post_ids',502);
  return {post_ids:ids,urls:ids.map(id=>'https://x.com/i/status/'+id),
    answer:`Found ${ids.length} public post IDs in the last-seven-day search. Open source links for content.`,
    brief:ids.length?`Found ${ids.length} matching public X post IDs.`:'No matching public X post IDs were returned.',
    content_policy:'ID discovery only. Post text and user objects are not redistributed.',as_of:new Date().toISOString()};
}
export function handleX(request,env,fetcher=(...args)=>globalThis.fetch(...args)) {
  return guarded(async()=>{
    const p=input(request,['query']);
    const query=text(p.get('query'),'query');
    requireValue(typeof env.X_API_BEARER_TOKEN==='string'&&enabled(env.X_API_SPEND_AUTHORIZED),'x_api_account_and_spending_authorization_required',503);
    await quota(env,'x-search',10);
    const url=new URL(X_API);
    url.search=new URLSearchParams({query,max_results:'10','post.fields':'id',sort_order:'recency'});
    return bounded((async()=>{
      const response=await fetcher(url.href,{redirect:'manual',signal:AbortSignal.timeout(15000),headers:{Authorization:'Bearer '+env.X_API_BEARER_TOKEN}});
      if(response.status===429) requireValue(false,'x_api_rate_limit',429);
      const data=await jsonResponse(response);
      return normalizeX(data);
    })(),15000);
  });
}
export function normalizePackage(data,trackingNumber) {
  requireValue(data&&Array.isArray(data.shipments)&&data.shipments.length<=10,'invalid_tracking_response',502);
  const rows=data.shipments.filter(row=>row.id===trackingNumber);
  requireValue(rows.length===1,'tracking_identity_unverified',502);
  const row=rows[0],status=row.status;
  requireValue(status&&typeof status.statusCode==='string'&&/^[A-Za-z0-9_-]{1,40}$/.test(status.statusCode),'tracking_status_unavailable',502);
  requireValue(typeof row.service==='string'&&/^[A-Za-z0-9_-]{1,80}$/.test(row.service),'tracking_provider_unavailable',502);
  return {tracking_number:trackingNumber,provider:row.service,status_code:status.statusCode,
    reported_timestamp:typeof status.timestamp==='string'&&/^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{1,2}:\d{2})?)?$/.test(status.timestamp)&&status.timestamp.length<=64?status.timestamp:null,
    timestamp_basis:'Source string retained; a missing source timezone is not guessed.',
    attribution:'Delivered by Deutsche Post DHL Group',as_of:new Date().toISOString(),
    privacy_scope:'Approved public status only; no names, addresses, location or proof-of-delivery returned.'};
}
export function handlePackage(request,env,fetcher=(...args)=>globalThis.fetch(...args)) {
  return guarded(async()=>{
    const p=input(request,['tracking_number']);
    const trackingNumber=text(p.get('tracking_number'),'tracking_number',80);
    requireValue(/^[A-Za-z0-9-]{6,40}$/.test(trackingNumber),'invalid_tracking_number');
    requireValue(typeof env.DHL_API_KEY==='string'&&enabled(env.DHL_PUBLIC_STATUS_CONSENT_CONFIRMED),'dhl_account_and_data_permission_required',503);
    requireValue(env.MINER_DB?.withSession,'tracking_grant_store_not_bound',503);
    const digest=hex(await crypto.subtle.digest('SHA-256',encoder.encode(trackingNumber)));
    const grant=await bounded(env.MINER_DB.withSession('first-primary').prepare('SELECT public_status_consented,expires_at FROM s2_tracking_grants WHERE tracking_sha256=? LIMIT 1').bind(digest).first(),6000);
    requireValue(grant&&grant.public_status_consented===1&&timestamp(grant.expires_at)>Date.now(),'tracking_public_status_not_authorized',403);
    await quota(env,'package-status',20);
    const url=new URL(DHL_API);
    url.search=new URLSearchParams({trackingNumber,language:'en'});
    return bounded((async()=>{
      const response=await fetcher(url.href,{redirect:'manual',signal:AbortSignal.timeout(15000),headers:{'DHL-API-Key':env.DHL_API_KEY,Accept:'application/json'}});
      if(response.status===429) requireValue(false,'dhl_api_rate_limit',429);
      return normalizePackage(await jsonResponse(response),trackingNumber);
    })(),15000);
  });
}
export function normalizePort(value,port,now=Date.now()) {
  requireValue(value&&value.version===1&&value.wait_definition==='mean_completed_anchorage_wait_hours'&&value.wait_unit==='hours','port_wait_semantics_unverified',502);
  requireValue(typeof value.port==='string'&&value.port.normalize('NFC').trim().toLowerCase()===port.normalize('NFC').trim().toLowerCase(),'port_identity_mismatch',502);
  const at=timestamp(value.as_of);
  requireValue(at<=now&&now-at<=300000,'port_observation_stale_or_future',503);
  requireValue(typeof value.wait_hours==='number'&&Number.isFinite(value.wait_hours)&&value.wait_hours>=0&&value.wait_hours<=10000,'invalid_port_wait',502);
  requireValue(Number.isSafeInteger(value.vessels_waiting)&&value.vessels_waiting>=0&&value.vessels_waiting<=100000,'invalid_waiting_vessels',502);
  requireValue(['low','moderate','high','severe'].includes(value.congestion_level),'invalid_congestion_level',502);
  return {port:value.port,wait_hours:value.wait_hours,vessels_waiting:value.vessels_waiting,
    congestion_level:value.congestion_level,as_of:value.as_of,wait_definition:value.wait_definition,
    verification:'Authenticated licensed publisher observation; signature is not independent proof of the physical measurement.'};
}
export async function verifyPortEnvelope(envelope,secret,port,now) {
  requireValue(envelope&&typeof envelope.payload==='string'&&envelope.payload.length<=16384&&/^[A-Za-z0-9+/]+={0,2}$/.test(envelope.payload)&&typeof envelope.signature==='string'&&/^[0-9a-f]{64}$/.test(envelope.signature),'invalid_port_signature_envelope',502);
  let raw;
  try {raw=Uint8Array.from(atob(envelope.payload),char=>char.charCodeAt(0));} catch {requireValue(false,'invalid_port_payload',502);}
  const signature=Uint8Array.from(envelope.signature.match(/../g),pair=>parseInt(pair,16));
  const key=await crypto.subtle.importKey('raw',encoder.encode(secret),{name:'HMAC',hash:'SHA-256'},false,['verify']);
  requireValue(await crypto.subtle.verify('HMAC',key,signature,raw),'port_signature_invalid',502);
  let value;
  try {value=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(raw));} catch {requireValue(false,'invalid_port_payload',502);}
  return {...normalizePort(value,port,now),payload_sha256:hex(await crypto.subtle.digest('SHA-256',raw))};
}
export function handlePort(request,env,fetcher=(...args)=>globalThis.fetch(...args)) {
  return guarded(async()=>{
    const p=input(request,['port']),port=text(p.get('port'),'port');
    requireValue(typeof env.PORT_FEED_URL==='string'&&typeof env.PORT_FEED_HMAC_KEY==='string'&&env.PORT_FEED_HMAC_KEY.length>=32&&enabled(env.PORT_FEED_LICENSE_APPROVED),'licensed_signed_port_feed_required',503);
    const url=publicURL(env.PORT_FEED_URL);
    requireValue(!url.search,'port_source_query_not_supported',503);
    url.searchParams.set('port',port);
    await quota(env,'port-congestion',20);
    return bounded((async()=>{
      const response=await fetcher(url.href,{redirect:'manual',signal:AbortSignal.timeout(15000)});
      const result=await verifyPortEnvelope(await jsonResponse(response,32768),env.PORT_FEED_HMAC_KEY,port,Date.now());
      return {...result,source_url:url.href};
    })(),15000);
  });
}
