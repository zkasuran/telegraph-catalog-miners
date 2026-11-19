# Source admission threat model

| Attack | Defence | Test |
|---|---|---|
| Quality pass becomes genuine | Exact original-state mapping | quality_pass_not_genuine |
| Probably falsified becomes confirmed | Uncertain state remains Unknown | probably_falsified_not_confirmed |
| Aggregate count becomes independent groups | Require released identities, no pairs created | full_workbook_not_population |
| PDF columns misbind batches | Match the analytical-method boundary | dashes_do_not_become_batches |
| Wrapped makers lose words | Recover the original manufacturer column | wrapped_manufacturer_preserved |
| Table ordinal substitutes for specimen ID | Keep QMN and split-batch suffixes | original_ids_not_table_ordinals |
| Silent partial source population | Require all 260 ordinals | missing_row_refused |
| Changed bytes or source attribution | Frozen snapshot hash and creators | frozen_snapshot_tamper_refused |
| Incompatible commercial grant | Reject NC declarations | noncommercial_original_refused |

The gate is source-only. It does not run a model, manufacture vendor identities
or authenticate another physical object. Snapshot grants and original source
hashes were verified before projection. The shipped snapshot is hash-checked,
not a replacement for independent re-reading of the original source records.
Malformed input is bounded and refused; no external entity declarations allowed.
