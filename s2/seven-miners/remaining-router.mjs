// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import {handleMedia} from './media.mjs';
import {handlePackage,handlePort,handleX} from './sources.mjs';
import {handleIndexedX} from './indexed-x.mjs';
export const PATHS=Object.freeze(['/miners/v3/deepfake','/miners/v3/video','/miners/v3/port','/miners/v3/package','/miners/v3/x-search']);
export function createRemainingIntegration(original,fetcher=(...args)=>globalThis.fetch(...args)) {
  return {fetch(request,env,context) {
    const path=new URL(request.url).pathname;
    if(path===PATHS[0]) return handleMedia(request,env,false,fetcher);
    if(path===PATHS[1]) return handleMedia(request,env,true,fetcher);
    if(path===PATHS[2]) return handlePort(request,env,fetcher);
    if(path===PATHS[3]) return handlePackage(request,env,fetcher);
    if(path===PATHS[4]) return env.X_API_BEARER_TOKEN&&env.X_API_SPEND_AUTHORIZED==='true'?handleX(request,env,fetcher):handleIndexedX(request,env,fetcher);
    return original.fetch(request,env,context);
  }};
}
