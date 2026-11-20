# English and French checkpoint-only runtime qualification

This pipeline replays the exact licensed LANGUAGE_TRANSLATION package from
public CPU run 44463705810 at commit 312353893799fe90b52111ae37a079809f56fbc4.
Its native grouped scores were 82.8125% raw, 85.9375% JSON and 80.859375% prose.
Full Stage 1 passed. The original job failed because an extracted score function
was missing json in its execution namespace, not because a new quality loss was
observed. The local literal-wire regression reproduces and repairs that defect.

No retraining, coefficient fitting, source reselection or original-text correction
occurs here. The original audit is now exposed. Reference runtime replay is an
integration measurement of the same immutable candidate, not a fresh data audit
or a canonical node win.

## Source-only verification

From the public checkout:

```sh
bash verification/owners/language_translation/verify.sh
```

This standalone gate uses the Python standard library only. It verifies source,
model and original-run hashes, grants and per-sentence attribution, grouped
identity/text overlap and the exact missing-json failure and repaired wire path.
Score replies in the local smoke are synthetic; no model, server or network
request is executed. It refuses token overflow instead of silently truncating.

The preserved split has 128 train, 32 validation and 64 final source/donor blocks,
with 512/128/256 preferences and 896 original attributed sentences. Original
CC BY 2.0 FR text remains separate from SAND additions. See DATA-SOURCES.md,
NOTICE and the complete source grants in assets/rights.json.

## Actual public CPU recovery

The owned workflow has verify and serve modes on standard ubuntu-24.04 public
CPU runners. Verification downloads the pinned foundation and unmodified source
kit, runs full Stage 1, then executes every frozen raw/JSON/prose pair in the real
reference engine. It checks all repeated bits, a fresh process and three negative
identities before emitting registration-ready.json. It never fills that report
from expected values. Native untrained baselines remain bound to the original run.

Gates stay at 80% overall in every format, 70% per direction, five points over
untrained and ten points over the strongest frozen shortcut. Local source gate
success is not those remote results. Memory is bounded to min(10 GiB RSS, 75% of
runner RAM), with an explicit 9,300-second verification process-tree wall limit.

Serving downloads only the actual completed result artifact by its verified run
and hash, loads the exact package and foundation engine and uses the unchanged
public common.lease_for restrictions. It uploads only the restricted public lease.
The runner's forwarding key is ephemeral; no signer/provider key is handled.
The automatic action adapter owns publication, endpoint activation and signing.

## What remains unverified

Before the new job completes: actual reference ordering, repeated/fresh bits,
runtime throughput and activation. Node qualification is separate. Source labels
are AI-reviewed constructed preferences, not human votes or universal factual
proof. The old failed/exposed archive is not used as new training or test data.
