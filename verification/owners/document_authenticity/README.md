# Rank document answers against measured signature evidence

This DOCUMENT_AUTHENTICITY pipeline learns to rank faithful JSON interpretations
of separately supplied mathematical PDF signature and signed-range reports. It
does not authenticate a candidate's own report or infer natural forgery. The
unknown authentic boolean is omitted. Native evidence-free inputs are unsupported
by the measured source task.

## Run the source gate

Requirements: Python 3.12 or newer, PyYAML and Bash. No model is loaded locally.

```sh
bash verification/owners/document_authenticity/verify.sh
```

The same source gate runs in publication and the public CPU workflow. It compiles
the owned Python, validates configuration and runs source-derived smoke plus
adversarial controls. Full Go compilation and the foundation engine run remotely.

## Frozen source boundary

The original collector read all eight year maps for four selected Congressional
sessions, not the first page. It selected 32 enrolled originals from 1405 eligible
package identities. Four byte views per original were measured with Poppler.
These are constructed tamper/padding controls, not naturally forged specimens.

There are 128 training pairs from 16 originals, 64 validation pairs from eight
originals and 64 reserved audit pairs from eight originals. An original and all
its views stay in one publication-session split. Source content is not used as
model input. Only identities, digests and measured report fields appear.

The failed v1 and frozen coefficients remain private historical evidence. The v2
grammar states both legal boolean values and leaves every measured fact, original
selection, split and quality threshold unchanged. Length, answer-only and token
overlap rules each score 50 percent. An exact report parser remains the source
authority. These numbers do not prove useful neural learning or universal truth.

## Remote qualification

The adapter dispatches one bounded free ubuntu-24.04 CPU job. Training sees only
training pairs. Validation selects one checkpoint. The audit is read after that
checkpoint freezes and is not used to choose another candidate. An audit failure
is preserved with the checkpoint's full grants, never hidden by another draw.

The remote audit measures trained and seed-matched untrained models. It exports
the exact selected delta then compiles the full pinned Stage 1 parser and runs
every audit pair through the foundation's reference executor. Only JSON is a
claimed format. Each supported assertion label needs 70 percent, overall needs
80 percent, five points above untrained and ten above the strongest shortcut.
Repeated score bits and three actual negative identity controls are separate
requirements. A report is emitted only after every executed gate passes.

Serving uses the identical source commit, package and executor. Its restricted
ephemeral forwarding key is generated only on the free runner. The result
artifact contains only the public lease. The automatic adapter owns endpoint
activation and Sepolia signing. This pipeline has no signer or provider key.

## Known limits

One issuer, shared templates and possible semantic clones limit independence.
Publication-session splits are not issuer-held-out or prospective acquisition
splits. Signing-time trust and production trusted-report delivery are unverified.
The neural audit is a source-conditioned diagnostic. A lookup or held-out result
is not universal factual proof. Only an independently observed canonical node
decision can establish a win.

Terms and attribution: LICENSE, NOTICE, DATA-SOURCES.md and KIT-LICENSE. The full
Apache-2.0 and MIT texts remain in licences.json. Our additions use SAND and do
not replace any upstream grant.
