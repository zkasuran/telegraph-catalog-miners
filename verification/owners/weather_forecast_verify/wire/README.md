# Check the node client before qualifying a scorer

This source-only harness uses the pinned real Telegraph Go caller on constructed
HTTP replies. It separates malformed score bits, identity refusals and temporary
HTTP faults before a candidate reaches a neural runtime.

The controls cover exact finite FP32 bit patterns, each score-identity field,
input roots, nonfinite values, integer overflow, malformed result counts and
the distinction between HTTP 503 and HTTP 422. Eleven batch-hash cases include
lexical index ordering beyond ten items and the current 100-item ceiling.

## Run the source-only checks

The family release gate includes the constructed report/parser failure controls.
No Go compiler or model is needed for those Python controls.

To execute the real primary Go client, use Go 1.25.7 and a clean checkout of
telegraphprotocol/telegraph-wasm-baseline at revision
1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd:

```sh
python3 wirecheck.py --kit /path/to/telegraph-wasm-baseline --go go --out /tmp/weather-wire-proof
```

Use a new output directory. Existing logs and reports are preserved, not overwritten.
The command records source hashes, executable hashes, compiler identity and each
actual command result. It performs syntax/compilation, Go formatting and Go vet
then executes the real client with bounded local HTTP fixtures. No foundation,
package, tokenizer model or neural engine is loaded.

## What this does not prove

The replies and token matrices are constructed integration fixtures. They are
not human annotations, training material, protected evaluation rows or neural
score vectors. A passing source-wire.json is not registration-ready.json.
It cannot qualify model accuracy, grant observation rights or claim a canonical win.
The actual neural audit and live public serving checks remain separate gates.

The fixture uses a local test client, not the production SSRF transport. These
controls do not test production address restrictions or redirect handling.

## Terms

New harness additions: copyright 2026 zkasuran, LicenseRef-zkasuran-SAND-1.0,
under the family LICENSE. Assistance was used for implementation and tests.
No human review is claimed. The unchanged imported primary kit remains MIT,
copyright 2026 telegraphprotocol, under the family KIT-LICENSE. Go remains under
its BSD-style terms in GO-LICENSE. Those grants are not replaced by our terms.
