// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran. Existing service modules retain their original notices.
import baseline from './baseline/worker.mjs';
import { handleFinance } from './sources/finance/finance.mjs';
import { handle as handleOperations } from './sources/operations/operations.mjs';

export const NEW_PATHS = Object.freeze([
  '/truth/v2/finance-reserve',
  '/truth/v2/finance-yield',
  '/truth/v2/operations/validator',
  '/truth/v2/operations/weather',
]);

export function createIntegration(original = baseline, finance = handleFinance, operations = handleOperations) {
  return {
    fetch(request, env, context) {
      const path = new URL(request.url).pathname;
      if (path === NEW_PATHS[0] || path === NEW_PATHS[1]) return finance(request);
      if (path === NEW_PATHS[2] || path === NEW_PATHS[3]) return operations(request);
      return original.fetch(request, env, context);
    },
  };
}

export default createIntegration();
