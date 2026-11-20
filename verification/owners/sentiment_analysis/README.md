# Recover and qualify the frozen sentiment scorer

This pipeline preserves the useful trained SENTIMENT_ANALYSIS scorer and repairs
registration identity without changing its learned function. The original real
source-held-out runtime measured 87.89 percent JSON, 88.67 percent prose and
91.02 percent bare-label pairwise accuracy on the same 256 Yelp originals.
These are retained source measurements, not a current node qualification claim.

Registration 118 failed before download because the endpoint echoed a 0x-prefixed
adapter hash. The reviewed front end now emits the node's literal bare hash.
This export adds only one ignored x-registration-recovery field so the reserved
old hash is not reused. Every original model field, grant, tensor descriptor and
payload byte is checked unchanged.

## Source-only verification

```sh
bash verification/owners/sentiment_analysis/verify.sh
```

The gate runs contract and failure tests without neural imports or model calls.
The HTTP tests use labelled source-only fixture scores. They are not model
inference or evidence of live node qualification. The published model bytes are
the exact licensed output of build.py and are hash-bound in frozen.json.
All public source and model artifacts carry or retain their grants.

## Bounded public CPU audit

The owned workflow is evaluator-owner-sentiment_analysis.yml. Its audit mode
reuses the frozen training/validation identity, reconstructs all 3,000 original
source rows and verifies the original grouped split hashes. It does not retrain.
It uses immutable train-fitted shortcut coefficients, not fresh float fitting.
JSON, prose and bare-label views cover all 256 originals with both source labels.

The runner validates both packages with the pinned full CheckLoRA. It loads the
real pinned foundation/engine through adapterexec and compares original and
recovery score bits on every audit view. Engines run sequentially, not as duplicate
simultaneous Foundation processes. Eight actual fixed vectors are scored twice.
The bounded front end proves three identity refusals and invalid-token refusal.
Every format must pass 80 percent overall, 70 percent per label, five points
above the original seed-matched untrained baseline and ten above the strongest
measured shortcut. The old untrained baseline is explicitly labelled PyTorch,
not falsely presented as a new untrained WASM measurement.

Only actual passing results produce registration-ready.json. It binds real
run, commit, package SHA/Keccak, foundation, engine, tokenizer, source manifest,
independent groups, all format metrics, parser/reference controls and fixed bits.
The source and serving commit remain identical. This is registration-ready
evidence, not a canonical win. No expected metric fills a missing result.

## Serving and automatic registration

After a real passing audit, the automatic adapter reserves a free existing-proxy
slot. Serving mode fetches that exact audit artifact using a step-scoped ephemeral
actions:read token. It never receives provider or wallet keys. It fetches the
published immutable package bytes and verifies actual parser/hash identity again.
The runner uploads only a restricted public lease. The automatic adapter checks
and installs the key/route and independently performs registration using only the
approved Telegraph identity and bounded test-USDC bond.

The real executor remains served on the same free public CPU infrastructure.
It checks the report's fixed vectors before forwarding and keeps an exact active
registration alive until its bounded lease ceiling. It does not replace a healthy
endpoint or dispatch a duplicate unresolved operation. Unknown write responses
are reconciled by operation key, never sent again.

## Limits and rights

Positive/Negative short English source labels only. Neutral, Mixed, reason
quality, other languages, population accuracy and backbone pretraining exclusion
are not validated. Source text/annotations and AI-assisted preference construction
are distinct. No human preference review or independent miner contributor is
invented. No hidden pool or exposed evaluation capture is used as training data.

Our competitive code and delta additions use LicenseRef-zkasuran-SAND-1.0.
Upstream ModernBERT Apache-2.0, kit MIT and UCI CC BY 4.0 grants remain attached.
Dataset statistics and coefficients retain the source grant. See LICENSE,
NOTICE, licences.json, DATA-SOURCES.md and SECURITY.md.
