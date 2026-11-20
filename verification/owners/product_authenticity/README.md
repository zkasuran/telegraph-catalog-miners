# Check archived product authenticity reports

This source-only pipeline checks reports about released medicinal product
annotations. It preserves 34 Newton2008 record units and reads all 76 explicitly
named product identifiers in Sanada2020. The latter source reports manufacturer
assessment, not a model prediction. Its three unnamed standard groups and repeat
spectral measurements are not turned into invented specimen IDs.

Run `bash verification/owners/product_authenticity/verify.sh` from the public
checkout. The same gate runs on a standard public CPU GitHub runner. It needs no
network access, external install, credentials or local neural imports.

`product_evidence.py` reads a JSON object with `product` and `answer` strings from
stdin. The answer string carries `product`, `verdict`, `authentic` and `evidence`.
Explicit archive requests use `Archived study specimen sanada2020-FV1; doi:10.3390/pharmacy9010003`.
The scorer gives fixed integer diagnostic points for an exact supported archive
report. Unknown products and new listings that copy an archive identifier remain
Inconclusive. False here means not judged authentic, not proven counterfeit.

## What this does not prove

This is not a registrable judgment adapter, trained model or semantic quality
benchmark. No reference engine inference, independent audit or registration has
run. Correct archive lookup is not proof of a new physical item's authenticity.
The live node's exact product question and raw/shaped answer contract still need
verification before any model candidate can claim compatibility.

Sanada's raw spectra and batch identity are not publicly released. Some falsified
Viagra spectra resemble genuine standards. A similar spectrum is not a universal
authenticity label. Source identifier prefixes G/F are an obvious lookup shortcut.
Do not train on exposed diagnostics and call them an independent audit.

The two source collections cannot be split into three independent source-held-out
partitions. Reused Sanada cohorts across earlier papers stay linked. The source
gate reports not ready rather than generating a registration-ready report.

Our code uses SAND. Newton's original public-domain grant, Sanada's CC BY 4.0
annotations and the kit/Pyflakes MIT portions retain their original grants. AI
assisted code and constructed response controls. No human review of our additions
is claimed. Original source-author annotations remain distinct.

## Source-grounded neural development

The `develop` workflow mode trains a real ModernBERT LoRA response-quality scorer only on the standard public CPU runner. It handles verdict consistency, fact attribution and reference-versus-new-object limits in JSON and prose. It does not universally authenticate physical products or retrieve missing issuer evidence. New preferences are constructed from original source facts and are not human-reviewed. Source identifiers with G/F label prefixes are masked as technical fixture tags.

Development contains 592 train comparisons from 74 original records and 608 validation comparisons from 76 source-held-out records. These comparisons are constructed report preferences, not independent physical specimens. Validation may select a frozen checkpoint. The independent 32-group audit remains unavailable and this mode cannot emit registration-ready.json or trigger registration. Its artifact binds actual model, parser and reference-runtime execution if those operations succeed.

## Measured development and audit admission

Development run 44603961619 at commit 5d8e9fcea5f52f53a3a1694323bb63d4422c681b produced a frozen package and actual full-parser/reference-engine results. Constructed source-held-out validation was 304/304 in JSON and 304/304 in prose. This is not independent physical-item accuracy or canonical qualification. The strongest declared validation shortcut was 62.5%. The model has not scored the new audit-source records.

The additive audit-admission gate preserves all 29 earlier tests and adds 13 source controls. It reads all 260 Nigeria rows but admits no genuine labels from quality passes or database lookups. Bangladesh's workbook contains three methods sheets, not 189 individual specimen records. The independent 32-group audit remains unverified and registration stays disabled.

## Broader development mode

The `develop_broad` workflow mode uses 1184 newly constructed training comparisons from 74 original records and 1216 validation comparisons from 76 source-held-out records. It covers JSON/prose with direct/typed questions and disjoint response formulations. Product inputs retain the actual 256-byte ceiling while the model token ceiling is 512. Source tests and live class checks precede bounded free CPU training. Full parser and reference-engine execution remain mandatory. No independent audit, registration readiness or new model result is claimed until actual artifacts are read. The earlier model's validation remains matched by a perfect structured grammar control and is not qualified.

## Selected-package recovery

Run 44630753101 completed training but failed an experiment wrapper that still expected 256 tokens for a 512-token package. The unchanged selected delta is preserved under recovery with its original data, selection and grant bindings. The recover_broad workflow performs no retraining. It must verify the full package, tensor roundtrip, native development measures and reference engine before any result is claimed. An independent audit and registration readiness remain unavailable.
