# Weather scorer threat model

| Attack | Defence | Test |
|---|---|---|
| Wrong station or duplicate station-month | Exact fixed-column identity and duplicate refusal | station_mismatch_and_duplicate_record_refused |
| Source with uncleared upstream rights | Only NWS CF6 source flag 1 admitted | every_other_source_is_excluded |
| QC failure, missing or presumed zero | Exclusion without zero substitution | each_quality_failure_is_excluded, missing_is_not_zero |
| Trace precipitation treated as measurable rain | Preserve measurement T and refuse nonzero trace | trace_is_distinct_from_measured_zero |
| Out-of-range or malformed inputs | Byte, record, calendar and numeric limits | ranges_and_source_ceiling, leap_day_and_nonexistent_day |
| JSON duplicate, NaN or injected fields | Strict typed decimal parsing and duplicate/unknown-field refusal | duplicates_nonfinite_string_number_bool_refused, wrong_location_day_and_injected_fields_refused |
| Fahrenheit mistaken for Celsius | Supported Celsius or exact numeric contract only | fahrenheit_and_arbitrary_number_mining_refused |
| Source lookup masquerading as universal proof | Unsupported status with null score | unsupported_stays_unknown_and_city_alias_not_invented |
| Entity or weather-window train leakage | Disjoint stations, months and normalized prompts | frozen_splits_overlap_and_mutation_control |
| Local neural execution | Workflow guard before model imports | local_model_guard_fails_before_model_import |
| Licence stamping changes tensors | Inert safetensors metadata and ZIP comment | stamp_retains_descriptor_and_payload, reward_head_zip_comment_preserves_contents |
| Remote identity swap or batch score mismatch | Foundation/tokenizer/package identities and count checks | remote code prepared, actual neural endpoint remains unverified |

## Limits

Source flags and archive quality checks do not certify final climate truth. No arbitrary city-to-station mapping is supplied. A trained neural model could learn climatology or source distribution shortcuts. Local failure controls are not live node evidence. The controller must retain the audit even if it fails and must not select a new package using the audit result.
