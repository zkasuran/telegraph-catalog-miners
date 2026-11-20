// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import {handleChat} from './chat.mjs';
export function createChatIntegration(original, fetcher=globalThis.fetch) {
  return {fetch(request,env,context) {
    if (new URL(request.url).pathname === '/miners/v3/chat') return handleChat(request,env,fetcher);
    return original.fetch(request,env,context);
  }};
}
