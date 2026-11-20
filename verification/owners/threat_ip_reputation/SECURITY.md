# Source reference threat model

| Attack | Defense | Test |
|---|---|---|
| Delete a real rule from editable evidence metadata | Complete eight-card source selection is pinned | test_complete_source_set_and_metadata_are_required |
| Swap a neighboring IP or a CIDR boundary | Exact address predicates, including overlapping rules | test_exact_overlapping_source_predicates, test_neighbor_and_cidr_boundaries |
| Use an IPv4-mapped IPv6 alias to change identity | Normalize the address before source membership checks | test_mapped_ipv6_and_unseen_ipv6 |
| Turn missing source evidence into a clean bill of health | Current abuse remains unknown and the evidence basis stays conditional | test_current_abuse_and_packet_match_shortcuts_rejected |
| Refresh an old snapshot by editing a timestamp | Capture identity is pinned and as-of is explicit UTC | test_stale_and_future_snapshot_controls |
| Exploit duplicate JSON keys or ignored outer fields | Duplicate keys, extra fields and non-string envelope fields are refused | test_literal_cli_good_overclaim_duplicate_and_invalid |
| Crash on malformed or deeply nested input | Named byte ceilings, nesting refusal and safe JSON error exit | test_strict_parser_and_seeded_controls, test_literal_cli_good_overclaim_duplicate_and_invalid |
| Train or benchmark the wrong input template | Compare literal UTF-8 render bytes against the retained pure primary expression | test_literal_primary_render_bytes |
| Claim success from tests that never fail | Accept-all, never-stale and source-omission mutations must cause assertions to fail | verify.py |

## Known limits

This reference is not a public neural runnable endpoint. It does not execute Suricata payload conditions, fetch live threat feeds or prove current maliciousness. Rule source can retain old addresses and reassigned infrastructure. Rule edit timestamps cannot become last-seen observations.

The eight source cards cover six /24 source/entity groups from one publisher and one snapshot. There is no independent training, validation or audit split. The source-only report is not a registration-ready report and cannot satisfy the unchanged model qualification gates.

The complete source and card hashes are fixed trust anchors for the inspected bytes, not publisher signatures. A local adversary able to replace both code and its trusted hashes is outside this artifact's trust boundary.

The source-only workflow runs on standard public CPU runners with one job, an explicit ten-minute timeout and read-only repository permissions. It uses no signer, provider credential, forwarding lease or deployed service. No neural package is loaded by the release gate.

## Remote learning boundary

learn/remote.py refuses local execution before neural imports, downloads or engine startup. The source gate exercises that real refusal. Frozen train/validation hashes and whole source groups are checked before fitting. Only training fits the immutable shortcut coefficients; future audits must reuse their exported bytes. Every claimed format must pass 80% overall, 70% per label, five points over untrained and ten points over the strongest tested shortcut. Validation selects the frozen candidate. A separate fresh audit is still required. The training stage cannot emit REGISTRATION_READY. The actual reference executor must echo literal bare identities, reject three wrong identities and return repeated identical finite score bits. These runtime checks run remotely, never in the local release gate. No signer, provider credential or forwarding key is part of this stage.

## Failure recovery controls

The actual TrainingRecord defect is tested without neural imports. Repacking changes metadata only and streams tensor bytes verbatim, checking their digest and unchanged descriptors. Original licence strings and provenance remain in the header. Original parser failure and every recovery parser result retain exact stdout/stderr before another stage runs. Runtime evaluation is public CPU only. A corrected container cannot authorize registration of a weak model. No audit is used for selection.

## Failure recovery controls

The actual TrainingRecord defect is tested without neural imports. Repacking changes metadata only and streams tensor bytes verbatim, checking their digest and unchanged descriptors. Original licence strings and provenance remain in the header. Original parser failure and every recovery parser result retain exact stdout/stderr before another stage runs. Runtime evaluation is public CPU only. A corrected container cannot authorize registration of a weak model. No audit is used for selection.

Live-class integration: the exact prior parser refusal was the obsolete compiled-in catalog, not a guessed runtime failure. The public node's family/child classification is checked fresh and bound to the isolated reference tool through the upstream supported catalog.UseClasses API. No controller/shared/node catalogue is modified. The full upstream CheckLoRA parser and all original checks remain unchanged. The new Go tool has been compiled and its valid-proof required-input path plus wrong-class refusal exercised without any local model/package loading.

## Failure recovery controls

The actual TrainingRecord defect is tested without neural imports. Repacking changes metadata only and streams tensor bytes verbatim, checking their digest and unchanged descriptors. Original licence strings and provenance remain in the header. Original parser failure and every recovery parser result retain exact stdout/stderr before another stage runs. Runtime evaluation is public CPU only. A corrected container cannot authorize registration of a weak model. No audit is used for selection.

Live-class integration: the exact prior parser refusal was the obsolete compiled-in catalog, not a guessed runtime failure. The public node's family/child classification is checked fresh and bound to the isolated reference tool through the upstream supported catalog.UseClasses API. No controller/shared/node catalogue is modified. The full upstream CheckLoRA parser and all original checks remain unchanged. The new Go tool has been compiled and its valid-proof required-input path plus wrong-class refusal exercised without any local model/package loading.

Literal wire recovery: the pinned reference executor intentionally warns on requested identity mismatch and returns the actual identity. The owner front end now rejects wrong/malformed bare hashes and token arrays before forwarding, validates every actual backend reply and retains exact score bits. This is serving-contract integration, not modified neural math or weakened identity tests. Local HTTP smoke uses an explicitly fake non-neural backend; real vector and refusal checks run only on the public CPU runner.

Compact elapsed-time model formulation

The actual recovery run 44611836950 confirmed full parser, three literal identity refusals, identical repeated reference bits and tensor/grant preservation. Both trained and untrained development accuracy was 50% in JSON/prose, with current label 0% and historical label 100%. No independent audit was exposed. This new formulation keeps exactly the prior source observations, source/entity/time split and response labels, but presents exact derived observation age in compact normal text and diversifies supported assessment times. It does not add a prefilled reputation/truth flag or change observation timestamps. All new contrasts are explicitly AI-assisted constructions, not human-reviewed source labels. New development has 512 train pairs/64 components and 128 validation pairs/16 components with zero normalized overlap; length, unique-word overlap and answer-only current-state controls remain 50%. Rank 16, 256-token ceiling, three epochs, smaller accumulation and a stronger learning rate are a new development candidate, not a blind retraining/registration of unchanged bytes. Unchanged upstream data/surface, tokenizer, per-format 80%, per-label 70%, five-point untrained and ten-point shortcut improvement gates still apply. Validation selects the candidate before collecting at least 32 disjoint fresh source/entity components after real freeze. No audit or canonical win is inferred.

Bounded validation-only continuation

Actual run 44622772244 improved native JSON development accuracy to 87.5% but prose reached only 75%, with current-observation prose accuracy 50%. The unchanged full format/label gate failed; no independent audit was collected. Data, 1280 tokenizer/render checks, parser, three real identity refusals and repeated reference vectors passed. The last epoch was still learning and selected by validation loss. This phase warm-starts that exact licensed checkpoint for two additional epochs with a lower learning rate, not a fresh restart or unchanged-binary registration. Only owned LoRA and reward-head tensors are loaded; frozen base weights remain unchanged. The inverse mapping is checked against primary exporter names/transposes and must map every package tensor exactly once. Actual parent JSON/prose validation must reproduce before fitting. Optimizer state was not retained and resets explicitly. Initial untrained baseline remains a separate seed-matched initializer, never the warm parent. Frozen data, source/entity/time splits, output formats and every quality gate remain unchanged. A later disjoint fresh audit after actual candidate freeze is still required.

Independent frozen audit pipeline

Actual continuation run 44643126816 passed native JSON/prose development at 100% overall and per label. This is not independent audit quality. The exact candidate was frozen at 2026-11-20T00:41:05.166908+00:00 before audit-source reads. Audit inputs must have last_seen after that timestamp and at least 32 source/entity components disjoint from development under global old/new IPv4 /24 and ASN closure. All four raw source formats and the complete validator/publication inventory are checked. Original CC0 grant and unchanged source curation remain. The old source snapshot is used only for entity-overlap closure, never as new audit observations. Audit preferences use the frozen construction rule and are AI-assisted, not human reviewed. No protected pool material is used. Source preflight reports AUDIT_SOURCE_NOT_READY before neural work when coverage is insufficient. Once eligible, the exact licensed candidate and immutable train-fitted baseline bytes are measured once by the full parser and actual reference engine in every supported format. The result cannot be REGISTRATION_READY until every original quality/runtime/source gate passes. No audit fitting or candidate selection occurs.

Complete historical entity closure

Audit isolation now includes every intervening published JSON snapshot through the exact original source commit anchor. Inventory pagination must reach that anchor; a first-page prefix is never accepted as complete. Historical source/grant bytes are retained and checked under bounded capture limits. An ASN or IPv4 /24 bridge remains part of the entity graph even after it disappears from the newest snapshot. Historical rows are closure input only, not new training observations or current audit candidates. The original model, data splits, preference formulation and all audit quality gates remain unchanged.

Gated bounded reference serving

Serving requires the actual completed independent audit report, its exact package and reserved existing proxy slot. Native development or a constructed report is not serving evidence. Source, format, label, shortcut and pinned runtime gates remain unchanged. Preparation runs the full parser and repeats actual report vectors through the identical reference engine and literal token-only front end. Only then may the public CPU runner create an ephemeral restricted forwarding key and upload its public lease. Candidate, runtime and source proof hashes all bind the same actual combined audit report bytes. The private key never enters an artifact. Serving uses the existing proxy, fixed host key and bounded remote-only forwarding. No endpoint is reserved or activated by source tests; all local fixtures are mechanical and no neural work occurs locally.

Serving resource identity is checked at every start

The serving engine checks the exact public forwarding host pin and all pinned foundation, engine and tokenizer file hashes immediately before starting an executor. Preparation is not a durable trust boundary: changed cached files or host pins are refused on later run-mode starts. These are byte-preserving integration checks. The trained delta, source observations, split policy and all independent audit quality gates stay unchanged. Mechanical process/backend fixtures prove refusal before executor start, without local models, private keys or SSH.
