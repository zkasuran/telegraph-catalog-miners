// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran. Additional evaluation permission in EVALUATION-PERMISSION.txt.
import original from './sources/original/worker.mjs';
import { handle as companyHandle } from './sources/corporate/worker.mjs';
import { handle as balanceHandle } from './sources/onchain/worker.mjs';
import hashprice from './sources/hashprice/entry.mjs';
import { workersFetch } from './workers-fetch.mjs';

const company = { fetch: request => companyHandle(request, workersFetch) };
const balance = { fetch: request => balanceHandle(request, workersFetch) };

export function forwardPath(request, pathname) {
  const url = new URL(request.url);
  url.pathname = pathname;
  // Request's own body stream and metadata are forwarded without parsing or re-encoding.
  return new Request(url.href, request);
}

export function createDispatcher(handlers = { original, company, balance, hashprice }) {
  return {
    fetch(request, env, ctx) {
      const path = new URL(request.url).pathname;
      if (path === '/truth/v1/company') {
        return handlers.company.fetch(forwardPath(request, '/company'), env, ctx);
      }
      if (path === '/truth/v1/balance') {
        return handlers.balance.fetch(forwardPath(request, '/metric'), env, ctx);
      }
      if (path === '/truth/v1/hashprice') {
        return handlers.hashprice.fetch(forwardPath(request, '/hashprice'), env, ctx);
      }
      // Same Request object, same env, same ctx and original receiver for all existing paths.
      return handlers.original.fetch(request, env, ctx);
    },
  };
}

export default createDispatcher();
