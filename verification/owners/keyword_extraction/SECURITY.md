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
| Expansion mistakes substring or annotation membership for truth | Bounded nested-match trie plus independent primary literal check and explicit annotation witnesses | Nested Unicode, morphological, source-group and topic controls. No proposal is automatically approved |
| Packing reuses source/answer resources or stale review state | Exclude approved resources, pack distinct resources and bind source/review hashes | Synthetic resource-conflict refusal and packed-view hash checks. Packing is not proof of preference quality |
| Shortcut learning | Train-only immutable integer coefficients plus unmodified primary dataset gate | Frozen coefficient test. Real dataset gate still pending |
| Wrong runnable identity or nonfinite score | Literal primary wire contract, primary Go input roots and strict FP32 decoding | Primary Go equality probe, root/prefix and nonfinite controls |
| Tiny or changed audit set | Rechecked minimum pair/component counts and frozen data/package/config/commit hashes | Minimum-count and audit-verdict failure tests |
| Report claims an unexecuted engine or identity result | Ready emitter requires completed parser/audit evidence, four actual identity probes, negative-token refusal and fixed finite vectors | Synthetic schema failure controls only. Actual remote probes remain unexecuted |
| Audit shortcut outcomes select a training candidate | Training computes shortcuts only for train/validation; independent audit retains the 65% gate across every format | Source-path inspection and audit shortcut-refusal control. Neural path is unexecuted |
| Accidental local model execution | Remote guard before model imports or weight preparation | Laptop refusal test |
| Runner environment contaminates a laptop-refusal test | Laptop tests use an explicit empty environment; a separate guard-only test covers public workflow context and wrong-repository refusal | Both clean and simulated GitHub source gates |
| Remote resource exhaustion | Workflow timeout plus process-tree RSS/time supervisor | Config checks only. Real runner resource enforcement is unverified |

## Known limits

Lexical presence is not centrality or truth. A supported term may be irrelevant and a nonliteral phrase may be a good keyword. The diagnostic reports that boundary rather than certifying source correctness.

Normalized author names are conservative grouping proxies, not resolved identities. Prior model pretraining exposure is unknown. Source rights do not certify annotation quality. Constructed reviews are not independent human judgments.

There is no approved dataset or trained package. The remote train/export/audit path is compiled and linted but has not run. One audit root refuses repeated measurement, but protection against duplicate candidate audits across remote runs remains a controller gate. Real model discrimination, process memory enforcement and node qualification are unverified. The entire deployed daemon caller is not present in the inspected primary kit.
