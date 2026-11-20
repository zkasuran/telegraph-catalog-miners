# Source scope and granting clauses

Wikidata structured data is CC0 1.0. The primary policy says:
"All data in Wikidata has a CC0 license."
https://www.wikidata.org/w/index.php?title=Wikidata:Licensing&oldid=2548950862
The complete policy HTML retains its original CC BY-SA 4.0 notice. Its original
bytes are `CC-BY-SA-4.0-policy.html`. Linked reference pages are not copied.

CC0 section 2 permits use "for any purpose whatsoever, including without
limitation commercial, advertising or promotional purposes". Full terms are
`SOURCE-CC0.txt`, SHA-256
`a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499`.

This experiment checks consistency with the actual referenced nondeprecated
P36 statements, not universal factual truth. The archive contains the exact
observed primary entity/revision/reference/qualifier bytes. Wrong candidate
claims and verdict substitutions are AI-assisted constructions, not human
preference annotations or new source facts. Multiple applicable capitals,
missing references, ambiguous time scopes and unknown values are excluded.
Missing evidence is not labelled False. Other graded verdicts are untested.

The frozen source archive SHA-256 is
`95993d951893142e20672c1a99ccc9eb37e37ebbf41479e9a24ca4551524b754`.
The unchanged train/development/audit split hashes are in `contract.json`.
All source facts keep CC0. Our code, constructed additions and fitted parameters
use `LICENSE`, the reviewed SAND terms. Failed checkpoints retain full grants.

ModernBERT by Answer.AI and LightOn retains Apache 2.0. Its pinned model card
states: "We release the ModernBERT model architectures, model weights, training
codebase under the Apache 2.0 license." Revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`; full terms in `UPSTREAM-APACHE-2.0.txt`.

Telegraph reference kit copyright 2026 telegraphprotocol, MIT. The grant includes
the right "to deal in the Software without restriction" with copyright and
permission notice retention. Revision `1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd`;
complete terms are in `KIT-LICENSE`. Our packaging changes do not replace those
upstream permissions or relicense source text.

## Development-only successor

Run 44674965147 failed quality at 32 of 64 source audit pairs. Its audit hash
`4772203facbf9fa3f6883f71375b5348e9ad348745a1be4fda85abfc98857a01`
is exposed and cannot be reused for fitting or another audit. The original
source archive, splits and train-fitted shortcut coefficients stay unchanged.
`develop.py` reads only the original 128 training and 32 validation pairs.
Validation may select a frozen checkpoint. It is not independent audit evidence.

The original discovery used LIMIT 512. It is a bounded pilot, not a population.
`inventory.py` reads an independent aggregate count then all primary subject IDs
with monotonic keyset pages. It checks the complete count before and after.
These IDs are discovery metadata, not accepted claim evidence or preferences.
Every newly sampled subject still needs its exact revision, all relevant P36
statements, references, alternatives and qualifier/time checks. An unchanged
aggregate does not make these requests an atomic snapshot.
