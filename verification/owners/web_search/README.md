# Score source-supported API version answers

This WEB_SEARCH_GENERAL pipeline ranks complete raw JSON answers against two compact documentation facts supplied in the query. It preserves the API-to-version relation rather than treating a URL or shared title as proof. The pinned ModernBERT LoRA reward model runs only on a bounded public CPU GitHub runner.

The corpus is unchanged from the source-only pilot: 256 training pairs on 63 Python pages, 64 validation pairs on 15 Python pages and 128 independent-audit pairs on 39 Node pages. Multiple pairs on one page are correlated. The full inventories cover 325 Python pages and 66 Node pages. Normalized source, API and answer overlap across splits is zero. Original source metadata is retained. The queries and wrong-version contrasts are constructed by AI-assisted code, not human-reviewed preferences.

## Verify the source

Run `bash verification/owners/web_search/verify.sh` from the standalone repository checkout. The local release gate uses the standard library only. It checks syntax, frozen data, original grants, the literal JSON/wire contract and meaningful refusal controls. It does not import a model library or fetch weights.

## Remote qualification

The manually dispatched owner workflow trains three fixed epochs on training only. Validation loss chooses a saved checkpoint with evaluation RNG isolated. That exact delta is exported and frozen before any audit score is read. A seed-matched untrained delta is exported before training. Both are checked by the full pinned parser and scored by the actual reference FP32 engine on the same untouched audit.

Only raw JSON responses are claimed. Every claimed format requires 80% audit accuracy, 70% per target-position label, five points over untrained and ten points over the strongest frozen length, overlap or answer-only shortcut. The evidence-binding lookup comparator is reported separately. It is a task-specific positive control, not universal factual proof.

A registration-ready report exists only after those real checks pass. It binds the run, commit, source inventory, package, foundation, engine, tokenizer and actual reference vectors. The serving mode uses the same executor, a five-hour restricted public forwarding lease and no operator key. Signing and endpoint activation stay inside the automatic action adapter.

## Limits

No live browsing is performed. Evidence must be present in the scored prompt and the decisive claim must survive tokenization. The raw-body default is verified from primary helper source, not asserted as a complete live daemon trace. A brief-only response is a different object and is not claimed. The node's literal token batch and bare-hash echoes are tested separately in the real remote executor.

Fine-tuning page separation does not establish independence from ModernBERT pretraining. Node is a domain and release-snapshot holdout, not a future-time guarantee. Source metadata can be wrong. A source-held-out result or registration receipt is not a canonical win.

## Terms

New competitive additions use LicenseRef-zkasuran-SAND-1.0. Python documentation retains PSF-2.0. Node documentation retains MIT. ModernBERT retains Apache-2.0 and the Telegraph reference kit retains MIT. Full upstream grants are kept in the source tree and embedded in successful and failed tensor artifacts. See DATA-SOURCES.md and NOTICE.
