# Keyword module threat model

| Input or attacker | Defence | Test or evidence |
|---|---|---|
| Tampered source inventory or XML | Complete Git tree hash, pinned commit binding, per-blob hash and source byte ceilings | Inventory and XML tamper tests, inventory-binding.json |
| Changed second claim for a cached source | Canonical source cached as parsed data, every claim rechecked | Real-XML repeated-grant failure test |
| Cropped prompt with uncropped source metadata | Exact complete prompt and response construction bound to source | Cropped-prompt refusal test |
| Unknown source grant | Article-specific reviewed URI, retained original attribution and notices | Unknown-grant refusal test |
| Entity expansion or oversized XML | 4 MiB source ceiling, declared XML entities refused, no external entity resolution | Entity refusal test |
| Ambiguous keyword syntax | Bounded string-list parser, bracket checking, numeric comma ambiguity requires JSON | Parser controls and 200 randomized cases |
| Keyword stuffing | Bounded term count, normalized duplicate penalty | Duplicate penalty and bounds tests |
| Leakage through donors, authors or formatting | Prefilter DOI/author/text components, topic inheritance, donor union and cross-split answer checks | Transitive author/topic and overlap refusal tests |
| Shared answer strings cross source splits | Merge source/donor components sharing normalized answer text before assigning splits | Actual SARS-CoV-2 overlap refusal and corrected assembly |
| Source capacity and assembly use different JSON layouts | Shared exact complete-annotation renderer without dropping source keywords | Complete list equality and literal wire rendering test |
| False source preference labels | Proposals remain unreviewed until explicit centrality reasons exist | Six documented original-text rejections and empty approval refusal |
| Duplicated/conflicting decisions or stale reported counts | Unique source/review IDs, explicit reasoning/origin and recomputed declared counts | Duplicate, conflicting decision, unknown proposal and count-drift controls |
| Stock negatives pass the owner checks but fail the primary source gate | Mirror non-fitted primary 80% uniqueness, 2% stock-negative, 1% identical and 65% length thresholds; original full remote gate remains | Actual failed remote gate and 2/68 source-threshold refusal test |
| Source deduplication changes an exposed held-out cohort | Omit only complete reciprocal training comparisons, preserve original conservative components and require byte-identical held-out freeze | Training-only selection contract, full source witness and frozen-byte mutation controls |
| Expansion mistakes substring or annotation membership for truth | Bounded nested-match trie plus independent primary literal check and explicit annotation witnesses | Nested Unicode, morphological, source-group and topic controls. No proposal is automatically approved |
| Packing reuses source/answer resources or stale review state | Exclude approved resources, pack distinct resources and bind source/review hashes | Synthetic resource-conflict refusal and packed-view hash checks. Packing is not proof of preference quality |
| Adding easy precision cases inflates an exposed audit | Precision cases are training-only; frozen validation/audit bytes cannot change | Held-out byte mutation, training-only donor and exact 25% cap tests |
| Donor metadata hides cross-split leakage | Original donor XML/annotation/grants verified and donor identities included in overlap checks | Donor attribution and cross-split overlap refusal controls |
| Shortcut learning | Train-only immutable integer coefficients plus unmodified primary dataset gate | Frozen coefficient test. Real dataset gate still pending |
| Wrong runnable identity or nonfinite score | Literal primary wire contract, primary Go input roots and strict FP32 decoding | Primary Go equality probe, root/prefix and nonfinite controls |
| Tiny or changed audit set | Rechecked minimum pair/component counts and frozen data/package/config/commit hashes | Minimum-count and audit-verdict failure tests |
| Report claims an unexecuted engine or identity result | Ready emitter requires completed parser/audit evidence, four actual identity probes, negative-token refusal and fixed finite vectors | Synthetic schema failure controls only. Actual remote probes remain unexecuted |
| Audit shortcut outcomes select a training candidate | Training computes shortcuts only for train/validation; independent audit retains the 65% gate across every format | Source-path inspection and audit shortcut-refusal control. Neural path is unexecuted |
| Accidental local model execution | Remote guard before model imports or weight preparation | Laptop refusal test |
| Runner environment contaminates a laptop-refusal test | Laptop tests use an explicit empty environment; a separate guard-only test covers public workflow context and wrong-repository refusal | Both clean and simulated GitHub source gates |
| Remote resource exhaustion | Workflow timeout plus process-tree RSS/time supervisor | Config checks only. Real runner resource enforcement is unverified |
| Serving trusts a source-only report or changed binary | Real passing ready report, successful actual audit run, full parser, identical executor bytes and repeated reference bits required before key creation | Synthetic report/run failures and remote-only guard tests. Actual serving remains unexecuted |
| A private key or unbounded forward reaches an artifact | Only public restricted lease.json uploads; selected owner, loopback-only port and five-hour lease, no agent forwarding | Exact lease/expiry/proof-hash controls. Actual key/forwarding operations are remote-only |
| Public wire accepts normalized hash stand-ins | Owned proxy preserves literal bare hashes and refuses prefix changes, invalid tokens and nonfinite results | Real local HTTP path with explicitly fake backend plus prefix/nonfinite controls |

## Known limits

Lexical presence is not centrality or truth. A supported term may be irrelevant and a nonliteral phrase may be a good keyword. The diagnostic reports that boundary rather than certifying source correctness.

Normalized author names are conservative grouping proxies, not resolved identities. Prior model pretraining exposure is unknown. Source rights do not certify annotation quality. Constructed reviews are not independent human judgments.

An approved source dataset now passes all local data gates, but no trained package or real neural report exists yet. The remote train/export/audit/serving paths are compiled and source-tested but have not run a model. One audit root refuses repeated measurement, and frozen held-out source bytes are required; protection against duplicate candidate audits across remote runs remains a controller gate. Real model discrimination, process memory enforcement and node qualification are unverified. The entire deployed daemon caller is not present in the inspected primary kit.
