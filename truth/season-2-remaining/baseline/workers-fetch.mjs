// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Copyright (c) 2026 zkasuran. Cloudflare Request [CF-REQUEST] in NOTICE.
// A scoped transport shim for two unchanged sources. Original Worker fetch is untouched.
export async function workersFetch(input, init) {
  if (init?.redirect !== 'error') return fetch(input, init);
  // workerd does not implement redirect:error. Never follow a redirect instead.
  const response = await fetch(input, { ...init, redirect: 'manual' });
  if ([301, 302, 303, 307, 308].includes(response.status)) {
    await response.body?.cancel();
    throw new TypeError('Redirect refused by truth-source transport');
  }
  return response;
}
