// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
export default [{
  files: ['*.mjs'],
  languageOptions: { ecmaVersion: 2025, sourceType: 'module', globals: {
    Request: 'readonly', Response: 'readonly', URL: 'readonly', URLSearchParams: 'readonly',
    ReadableStream: 'readonly', AbortController: 'readonly', Uint8Array: 'readonly',
    Buffer: 'readonly', process: 'readonly', setTimeout: 'readonly', console: 'readonly', structuredClone: 'readonly', fetch: 'readonly',
  } },
  rules: { 'no-unused-vars': ['error', { argsIgnorePattern: '^_' }], 'no-undef': 'error',
    'no-constant-condition': ['error', { checkLoops: false }], 'constructor-super': 'error',
    'no-dupe-args': 'error', 'no-unreachable': 'error', 'valid-typeof': 'error' },
}];
