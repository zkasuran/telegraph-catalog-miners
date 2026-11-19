/*! SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
 * Copyright (c) 2026 zkasuran. Full terms in LICENSE.
 * Additional Telegraph evaluation permission: EVALUATION-PERMISSION.txt.
 * PublicNode upstream terms and attribution: NOTICE and DATA-SOURCES.md.
 */
// Cloudflare entry point exports only the handler. Pure helpers live in worker.mjs.
export { default } from './worker.mjs';
