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
