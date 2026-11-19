# CI observation consistency threat model

Own additions: LicenseRef-zkasuran-SAND-1.0.

| Attacker or failure | Defence | Test |
|---|---|---|
| Candidate supplies a forged observation | Only separately supplied trusted evidence is read | test_answer_cannot_supply_its_own_authority |
| Successful unrelated run or retry | Repository, run, attempt and optional commit binding | test_attempt_pinning_no_latest_fallback and test_source_identity_and_commit_mismatch |
| HTTP 200 confused with passing CI | Source run status and conclusion, not transport success | test_transport_200_is_not_ci_success |
| Missing source turns into failure | available=false and score=null | test_source_missing_unknown_not_failed |
| Unsupported conclusion gets invented boolean | Only completed success/failure mapping supported | test_supported_passed_mapping_and_unresolved_states |
| URL abuse | HTTPS host/path allowlist, no credentials or ports | test_url_variants_and_hostile_hosts |
| Parser exhaustion | 20000 bytes and 64 JSON container levels | test_json_duplicate_nonfinite_and_depth and test_random_parser_inputs_bounded |
| First-page job count hides a failure | Total-count paging with duplicate/change guards | test_complete_job_inventory_deep_failure_visible |
| Split relabeling | Normalized identity and source/entity/time checks | test_overlap_detector_rejects_case_whitespace_variants |
| Local model operation | Remote context guard precedes model imports | test_local_context_refuses_neural_work |
| Source diagnostic mistaken for live coverage | Explicit caller attestation gate | test_missing_or_false_caller_attestation_blocks |
| Licence lost on failed checkpoint | Embedded upstream terms before own terms and export preservation | Two container byte-preservation tests |

## Known limits

The reference trusts its caller's observation. It does not authenticate GitHub,
execute patches or certify semantic fixes. Missing job records cannot establish
test coverage. The reference currently grades a native JSON response subset.
The live schema also permits other parser inputs and pipeline descriptions.
Those formats remain unmeasured. The score checks off-chain identity fields as
well as passed. Only passed is a live scored field.

Fictional timestamps are not newly measured events. Serialization holdout is
not a genuinely independent source corpus. Token visibility, neural quality,
Torch checkpoint reload, full node parser acceptance and exported runtime bits
remain remote/controller checks. Caller-attestation structure cannot by itself
prove that the controller's claimed source behavior is true.
