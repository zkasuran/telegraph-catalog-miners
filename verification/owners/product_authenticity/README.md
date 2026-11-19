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
