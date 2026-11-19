# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Propose and review source/donor blocks before any new split is assigned."""
import argparse
import json
import re
import sqlite3
from pathlib import Path

from archive_failed import ARCHIVE, FAILED_SHA
from pilot import COUNTS, content, digest, distinct, eligible, original
from source_probe import LINKS_SHA, sha, write

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE / "source-pool-proposal.json"
REVIEW = HERE / "source-pool-review.json"
POOL = HERE / "reviewed-source-pool.json"


def propose(extend=False):
    previous = json.loads(PROPOSAL.read_text()) if PROPOSAL.exists() else None
    previous_sha = sha(PROPOSAL) if previous else None
    if previous and not extend:
        raise ValueError("review proposal exists; reuse it instead of resampling")
    old_path = ARCHIVE / "pilot-manifest.json"
    if sha(old_path) != FAILED_SHA:
        raise ValueError("failed freeze archive changed")
    old = json.loads(old_path.read_text())
    banned = {c for blocks in old["splits"].values() for b in blocks for c in b["component_roots"]}
    spec = json.loads((HERE / "general-sources.json").read_text())
    db = sqlite3.connect(HERE / "out/general.sqlite")
    representatives = {}
    query = """SELECT p.component,e.id,e.text,f.id,f.text FROM pairs p
        JOIN sentences e ON e.id=p.eng JOIN sentences f ON f.id=p.fra
        JOIN components c ON c.component=p.component
        WHERE length(CAST(e.text AS BLOB)) BETWEEN 8 AND 128
          AND length(CAST(f.text AS BLOB)) BETWEEN 8 AND 128
          AND e.owner NOT IN ('','\\N') AND f.owner NOT IN ('','\\N') AND c.originals<=1000"""
    candidates = 0
    for component, e, et, f, ft in db.execute(query):
        if component in banned or not eligible(et) or not eligible(ft):
            continue
        # Conservative review scope. These screens never substitute for review.
        if re.search(r"\b(you|your|yours|he|she|his|her|hers|they|them|their|it|its)\b", et, re.IGNORECASE):
            continue
        if not content(et) or not content(ft) or not re.match(r"^(I |We |My |Our )", et):
            continue
        candidates += 1
        key = digest(f"translation-review-pool-v2:{e}:{f}")
        item = (key, component, e, et, f, ft)
        if component not in representatives or key < representatives[component][0]:
            representatives[component] = item
    buckets, blocks, originals = {}, [], {}
    for item in sorted(representatives.values()):
        bucket = (len(item[3]) // 16, len(item[5]) // 16, item[3][-1], item[5][-1])
        waiting = buckets.setdefault(bucket, [])
        match = next((i for i, a in enumerate(waiting) if distinct(a[3], item[3]) and distinct(a[5], item[5])), None)
        if match is None:
            if len(waiting) < 64:
                waiting.append(item)
            continue
        a, b = waiting.pop(match), item
        ids = sorted((a[2], a[4], b[2], b[4]))
        block = {"id": digest("reviewed-translation-block-v2:" + json.dumps(ids)),
                 "proposal_index": len(blocks), "component_roots": [a[1], b[1]],
                 "links": [[a[2], a[4]], [b[2], b[4]]], "length_bucket": list(bucket),
                 "component_membership_sha256": [digest(json.dumps([sid for sid, in db.execute("SELECT id FROM membership WHERE component=? ORDER BY id", (c,))])) for c in (a[1], b[1])]}
        for sid in ids:
            originals[str(sid)] = original(db, sid, spec)
        blocks.append(block)
        if len(blocks) == (400 if extend else 300):
            break
    db.close()
    if len(blocks) < 224:
        raise ValueError(f"only {len(blocks)} candidate review blocks, fixed pilot needs 224")
    if previous and (blocks[:len(previous["blocks"])] != previous["blocks"] or any(originals[k] != v for k, v in previous["originals"].items())):
        raise ValueError("source-pool extension changed an already reviewed original/block")
    write(PROPOSAL, {"schema_version": 2, "status": "UNREVIEWED_NO_SPLITS", "blocks": blocks, "originals": originals,
        "extended_previous_proposal_sha256": previous_sha,
        "source_spec_sha256": sha(HERE / "general-sources.json"), "links_sha256": LINKS_SHA,
        "failed_archive_manifest_sha256": FAILED_SHA, "excluded_old_components": sorted(banned),
        "candidate_links": candidates, "candidate_components": len(representatives),
        "source_license": "CC-BY-2.0-FR", "review_scope": "First-person everyday short statements. No unresolved second/third-person pronoun number, referent or ownership. AI bilingual review still required for each link and donor."})
    lines = []
    for block in blocks:
        texts = [f"{e}/{f}: {originals[str(e)]['text']} => {originals[str(f)]['text']}" for e, f in block["links"]]
        lines.append(f"{block['proposal_index']:03} {block['id'][:12]} | " + " || ".join(texts))
    (HERE / "out/new-pool-review.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"proposed_blocks": len(blocks), "candidate_links": candidates, "candidate_components": len(representatives), "proposal_sha256": sha(PROPOSAL)}))


def approve():
    proposal, review = json.loads(PROPOSAL.read_text()), json.loads(REVIEW.read_text())
    if review["proposal_sha256"] != sha(PROPOSAL) or review["review_kind"] != "AI-assisted source-grounded bilingual review before splitting":
        raise ValueError("source review identity or disclosure mismatch")
    rejected = {r["proposal_index"]: r for r in review["rejected_blocks"]}
    accepted = set(review["approved_indices"])
    if accepted.intersection(rejected) or accepted | set(rejected) != set(range(len(proposal["blocks"]))):
        raise ValueError("each proposed block needs exactly one actual review decision")
    selected = []
    for block in proposal["blocks"]:
        if block["proposal_index"] not in accepted:
            continue
        (ea, fa), (eb, fb) = block["links"]
        original_rows = proposal["originals"]
        proof = {"review": "AI-reviewed predicates, tense, polarity, number, referents and register preserve the supplied proposition in both linked directions.",
                 "source_a": original_rows[str(ea)]["text"], "linked_a": original_rows[str(fa)]["text"],
                 "source_b": original_rows[str(eb)]["text"], "linked_b": original_rows[str(fb)]["text"],
                 "donor_contrast": "Swapping these originals substitutes the other quoted proposition, not an alternate wording of this source. Both English and French source propositions were read before approval.",
                 "human_review_claimed": False}
        selected.append({**block, "bilingual_review": proof})
    if len(selected) < 224:
        raise ValueError(f"only {len(selected)} reviewed blocks, fixed pilot needs 224")
    write(POOL, {**proposal, "status": "AI_REVIEWED_BEFORE_FREEZE", "blocks": selected,
          "review_sha256": sha(REVIEW), "proposal_sha256": sha(PROPOSAL), "reviewed_blocks": len(selected),
          "source_review_before_split": True, "rejected_blocks": review["rejected_blocks"]})
    print(json.dumps({"reviewed_blocks": len(selected), "rejected_blocks": len(rejected), "reviewed_pool_sha256": sha(POOL)}))


def freeze():
    pool = json.loads(POOL.read_text())
    if pool["status"] != "AI_REVIEWED_BEFORE_FREEZE" or not pool["source_review_before_split"] or pool["review_sha256"] != sha(REVIEW):
        raise ValueError("only a fully reviewed source pool may be split")
    blocks = sorted(pool["blocks"], key=lambda b: digest("new-sealed-translation-v2:" + b["id"]))[:224]
    splits, offset = {}, 0
    ids = set()
    for split, count in COUNTS.items():
        splits[split] = blocks[offset:offset + count]
        offset += count
        ids.update(sid for b in splits[split] for p in b["links"] for sid in p)
    manifest = {"schema_version": 2, "family": "LANGUAGE_TRANSLATION", "source_license": "CC-BY-2.0-FR",
        "source_spec_sha256": pool["source_spec_sha256"], "links_sha256": LINKS_SHA, "group_counts": COUNTS,
        "splits": splits, "originals": {str(sid): pool["originals"][str(sid)] for sid in sorted(ids)},
        "reviewed_pool_sha256": sha(POOL), "semantic_review": "PASS_AI_REVIEW_BEFORE_FREEZE",
        "source_review_before_split": True, "failed_exposed_archive": "archive/failed-" + FAILED_SHA[:12],
        "excluded_failed_components": pool["excluded_old_components"],
        "selection": "New deterministic hash split from approved source/donor blocks only. All old exposed components excluded. No final scores or post-freeze source repair.",
        "construction": "Programmatically constructed reciprocal donor preferences after AI bilingual review of exact CC BY originals. Not human votes.",
        "sealed_final_policy": "Final groups selected only after complete source-pool review. No model choice or source repair from final audit."}
    target = HERE / "pilot-manifest.json"
    if target.exists() and sha(target) != FAILED_SHA:
        raise ValueError("new frozen manifest already exists; do not overwrite")
    write(target, manifest)
    write(HERE / "semantic-review.json", {"schema_version": 2, "status": "PASS", "review_kind": "AI-assisted reviewed source pool before new deterministic freeze; no human review",
        "pilot_manifest_sha256": sha(target), "reviewed_pool_sha256": sha(POOL), "source_pool_review_sha256": sha(REVIEW),
        "reviewed_source_links": len(pool["blocks"]) * 2, "reviewed_donor_blocks": len(pool["blocks"]),
        "known_failures": [], "old_failures_retained": "archive/failed-" + FAILED_SHA[:12],
        "new_final_material_not_repaired_after_freeze": True, "workflow_ready": False, "model_execution": False})
    print(json.dumps({"new_freeze_sha256": sha(target), "groups": COUNTS, "source_review_preceded_freeze": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("propose", "extend", "approve", "freeze"))
    {"propose": propose, "extend": lambda: propose(True), "approve": approve, "freeze": freeze}[parser.parse_args().phase]()
