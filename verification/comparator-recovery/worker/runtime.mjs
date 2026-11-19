// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran
import {createHandler} from './handler.mjs';

export function createRecovery(modules, binaries, manifest) {
  const instances = {}, initialized = {};
  for (const family of Object.keys(modules)) {
    Object.defineProperty(instances, family, {
      enumerable: true,
      get() {
        return initialized[family] ??= new WebAssembly.Instance(modules[family]);
      }
    });
  }
  const handler = createHandler(instances, binaries, manifest);
  return {
    initialized,
    async fetch(request) {
      const url = new URL(request.url);
      if (!url.pathname.startsWith(manifest.prefix)) throw new Error('outside recovery namespace');
      url.pathname = '/' + url.pathname.slice(manifest.prefix.length);
      if (request.method === 'GET' && url.pathname === '/health') {
        return new Response(JSON.stringify({version: manifest.version, families: manifest.families}), {
          headers: {'content-type': 'application/json', 'cache-control': 'no-store',
                    'x-content-type-options': 'nosniff', 'content-security-policy': "default-src 'none'; frame-ancestors 'none'"}
        });
      }
      return handler.fetch(new Request(url, request));
    }
  };
}

export function wrapPrior(prior, recovery, prefix) {
  return {
    ...prior,
    fetch(request, env, ctx) {
      if (new URL(request.url).pathname.startsWith(prefix)) return recovery.fetch(request);
      return prior.fetch(request, env, ctx);
    }
  };
}
