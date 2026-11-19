# Source-stage threat model

| Attack | Defence | Test or limit |
|---|---|---|
| Stock label or hypothesis shortcut | Balanced source hypotheses and frozen train-only prior | Prior failure retained; current real source gate passes at 50% |
| Cross-source leakage | Connected host/issuer/original groups and normalized evidence checks | Actual complete source manifests |
| Invented excerpt absence | No NotMentioned excerpt rows | Source constructor |
| Missing conditional evidence | Every annotated span retained; incomplete fragments excluded | Complete offset/index validation |
| Forged runtime identity | Literal bare hashes, not prefix-stripped comparisons | Three field mutation controls |
| Nonfinite score | uint32 and finite FP32 contract | NaN/Infinity and 500 random-bit controls |
| Local model operations | Remote-only guard and source-only gate | Denial tests |
| Fake quality pass | All format, label and improvement thresholds retained | Below-threshold failure controls |
| Slow or oversized HTTP input | Body/header byte ceilings, total input deadline and four connections | Actual local HTTP controls and remote identity/token controls |
| Wrong public model or report | Exact immutable URL/hash, full parser and reference vectors before serving | Serving preparation refuses mismatches |
| Unrestricted forwarding key | Exact existing common.lease_for with public proof hashes and pinned host | Only public restricted lease uploaded; ephemeral key stays on runner |

Known limits: source-stage checks are not neural or reference-engine results. Document labels are not performance labels. The raw scorer uses the documented 512-token interface and complete source contexts up to 768 bytes; the distinct typed miner slot remains bounded to 256 bytes. No long on-chain contract transport or link dereference is claimed. Entity aliases and legal applicability require wider evidence. All 99 inspected audit groups are permanently retired. A link-only or bare verdict answer contains no clause evidence for this narrow task. Even a completed independent reference audit is not a canonical node win. No quality or identity gate may be bypassed to register.
