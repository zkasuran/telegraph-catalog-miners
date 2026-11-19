# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Freeze original linked source/donor groups and run source-only data gates."""
import argparse
import hashlib
import importlib.util
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from source_checks import KIT_REVISION, TOKENIZER_SHA
from source_probe import LINKS_SHA, norm, sha, write

HERE = Path(__file__).resolve().parent
COUNTS = {"train": 128, "validation": 32, "test": 64}
WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
STOP = {
    "a", "an", "the", "is", "are", "be", "to", "of", "for", "from", "in", "at", "on",
    "it", "he", "she", "we", "they", "you", "i", "my", "me", "his", "her", "our", "their",
    "this", "that", "un", "une", "le", "la", "les", "de", "du", "des", "et", "est", "sont",
    "en", "au", "aux", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "ce",
    "cet", "cette", "ces", "mon", "ma", "mes", "son", "sa", "ses",
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def content(text):
    return set(WORDS.findall(text.casefold())) - STOP


def distinct(a, b):
    x, y = content(a), content(b)
    return bool(x and y) and len(x & y) / len(x | y) <= 0.5


def eligible(text):
    # A short everyday-sentence scope, not a universal translation benchmark.
    # This is fixed before any validation/final audit and never pads originals.
    return (8 <= len(text.encode()) <= 128 and len(WORDS.findall(text)) >= 3
            and not any(c in text for c in "\n\r\t<>@")
            and not re.search(r"https?://|www\.|\b(?:Tom|Mary)\b", text)
            and not re.search(r"\b[A-Z][a-z]+\b", text[1:])
            and text[-1:] in ".?!")


def original(db, sid, spec):
    row = db.execute("SELECT id,lang,text,owner,created,modified,line,detail_line FROM sentences WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise ValueError("original identity absent")
    sid, lang, text, owner, created, modified, line, detail_line = row
    if not owner or owner == "\\N" or not eligible(text):
        raise ValueError("original lacks source attribution or is out of fixed scope")
    return {"id": sid, "language": lang, "text": text,
            "credited_contributor": owner, "credit_basis": "Primary detailed export Username (sentence owner), not an independently verified creation history",
            "sentence_url": f"https://tatoeba.org/en/sentences/show/{sid}",
            "contributor_url": f"https://tatoeba.org/en/user/profile/{owner}",
            "created": created, "modified": modified,
            "source_row": line, "detailed_source_row": detail_line,
            "source_sha256": spec["files"][f"{lang}_sentences.tsv.bz2"]["sha256"],
            "detailed_source_sha256": spec["files"][f"{lang}_sentences_detailed.tsv.bz2"]["sha256"],
            "license": "CC-BY-2.0-FR", "license_url": spec["grant_url"],
            "original_text_changed": False}


def freeze(out):
    manifest_path = HERE / "pilot-manifest.json"
    if manifest_path.exists():
        raise ValueError("pilot is already frozen; audit existing bytes instead of selecting again")
    spec = json.loads((HERE / "general-sources.json").read_text())
    graph = json.loads((out / "general-graph.json").read_text())
    if graph["links_sha256"] != LINKS_SHA or graph["general_spec_sha256"] != sha(HERE / "general-sources.json") or graph["missing_reciprocals"]:
        raise ValueError("source graph binding mismatch")
    db = sqlite3.connect(out / "general.sqlite")
    query = """SELECT p.component,e.id,e.text,f.id,f.text FROM pairs p
               JOIN sentences e ON e.id=p.eng JOIN sentences f ON f.id=p.fra
               JOIN components c ON c.component=p.component
               WHERE length(CAST(e.text AS BLOB)) BETWEEN 8 AND 128
                 AND length(CAST(f.text AS BLOB)) BETWEEN 8 AND 128
                 AND e.owner NOT IN ('','\\N') AND f.owner NOT IN ('','\\N')
                 AND c.originals<=1000"""
    representatives = {}
    eligible_links = 0
    for component, e, et, f, ft in db.execute(query):
        if not eligible(et) or not eligible(ft) or not content(et) or not content(ft):
            continue
        eligible_links += 1
        key = digest(f"translation-original-v1:{e}:{f}")
        candidate = (key, component, e, et, f, ft)
        if component not in representatives or key < representatives[component][0]:
            representatives[component] = candidate
    pool = sorted(representatives.values())
    buckets = {}
    blocks = []
    # Match source/target length and punctuation before labels are constructed.
    # Each target occurs once on each side in its direction, within one group.
    for item in pool:
        _, component, e, et, f, ft = item
        bucket = (len(et) // 16, len(ft) // 16, et[-1], ft[-1])
        options = buckets.setdefault(bucket, [])
        matched = None
        for i, other in enumerate(options):
            if component != other[1] and distinct(et, other[3]) and distinct(ft, other[5]):
                matched = options.pop(i)
                break
        if matched is None:
            if len(options) < 64:
                options.append(item)
            continue
        a, b = matched, item
        ids = sorted((a[2], a[4], b[2], b[4]))
        blocks.append({"id": digest("translation-block-v1:" + json.dumps(ids)),
                       "component_roots": [a[1], b[1]], "links": [[a[2], a[4]], [b[2], b[4]]],
                       "length_bucket": list(bucket)})
        if len(blocks) == sum(COUNTS.values()):
            break
    if len(blocks) != 224:
        raise ValueError(f"only {len(blocks)} matched original/donor blocks, fixed pilot needs 224")
    blocks.sort(key=lambda b: digest("split-translation-v1:" + b["id"]))
    originals = {}
    splits, offset = {}, 0
    for split, count in COUNTS.items():
        splits[split] = blocks[offset:offset + count]
        offset += count
        for block in splits[split]:
            for pair in block["links"]:
                for sid in pair:
                    originals[str(sid)] = original(db, sid, spec)
            block["component_membership_sha256"] = [
                digest(json.dumps([sid for sid, in db.execute("SELECT id FROM membership WHERE component=? ORDER BY id", (component,))]))
                for component in block["component_roots"]]
    manifest = {"schema_version": 1, "family": "LANGUAGE_TRANSLATION", "source_spec_sha256": sha(HERE / "general-sources.json"),
                "links_sha256": LINKS_SHA, "graph_report": graph, "group_counts": COUNTS,
                "selection": "Fixed hash-order original representative per full normalized/link component, matched length/punctuation, disjoint reciprocal donor blocks. No final audit used for selection.",
                "eligible_original_links": eligible_links, "eligible_components": len(representatives),
                "splits": splits, "originals": originals,
                "construction": "Original community translation links, programmatically constructed reciprocal unrelated-original donor preferences. Not human preference votes or certified translation truth.",
                "error_family": "unrelated_original_donor", "source_license": "CC-BY-2.0-FR",
                "sealed_final_policy": "Final groups and bytes frozen before all gate measurements. No reselection or model/epoch choice from final results.",
                "semantic_review": "pending independent bilingual review of linked originals and donor meaning difference"}
    write(manifest_path, manifest)
    db.close()
    print(json.dumps({"frozen_groups": COUNTS, "originals": len(originals), "manifest_sha256": sha(manifest_path),
                      "eligible_original_links": eligible_links, "eligible_components": len(representatives)}), flush=True)


def render_pairs(manifest, split, style="raw"):
    originals = manifest["originals"]
    rows = []
    for block in manifest["splits"][split]:
        (ea, fa), (eb, fb) = block["links"]
        for source, preferred, donor, direction in ((ea, fa, fb, "eng-fra"), (eb, fb, fa, "eng-fra"),
                                                   (fa, ea, eb, "fra-eng"), (fb, eb, ea, "fra-eng")):
            original_source, chosen, rejected = (originals[str(sid)] for sid in (source, preferred, donor))
            target = "French" if direction == "eng-fra" else "English"
            prompt = json.dumps({"text": original_source["text"], "target_language": target}, ensure_ascii=False, separators=(",", ":"))
            def answer(text):
                if style == "json":
                    return json.dumps({"translation": text}, ensure_ascii=False, separators=(",", ":"))
                if style == "prose":
                    return "Translation: " + text
                return text
            rows.append({"prompt": prompt, "chosen": answer(chosen["text"]), "rejected": answer(rejected["text"]),
                         "margin": 0.0, "label_kind": "preference_pair", "direction": direction, "label": direction,
                         "group_id": block["id"], "error_family": "unrelated_original_donor", "style": style,
                         "source_id": source, "chosen_id": preferred, "donor_id": donor,
                         "source_license": "CC-BY-2.0-FR", "source_reference": original_source["sentence_url"],
                         "chosen_reference": chosen["sentence_url"], "donor_reference": rejected["sentence_url"],
                         "construction": "Choose the direct linked original target over an intact target from a different complete equivalence component. Constructed source-bound preference, not a human vote."})
    return rows


def checks(manifest, splits, tokenizer, render):
    seen_components, seen_texts, seen_ids = set(), set(), set()
    report = {}
    for split, rows in splits.items():
        blocks = manifest["splits"][split]
        if len(blocks) != COUNTS[split] or len(rows) != 4 * COUNTS[split]:
            raise ValueError("fixed group/row budget changed")
        comps = [component for b in blocks for component in b["component_roots"]]
        ids = {sid for b in blocks for pair in b["links"] for sid in pair}
        texts = {norm(manifest["originals"][str(sid)]["text"]) for sid in ids}
        if len(comps) != len(set(comps)) or seen_components.intersection(comps) or seen_texts.intersection(texts) or seen_ids.intersection(ids):
            raise ValueError("source/donor identity, component or visible normalized text crosses splits")
        seen_components.update(comps)
        seen_texts.update(texts)
        seen_ids.update(ids)
        count = Counter(row["direction"] for row in rows)
        if count["eng-fra"] != count["fra-eng"]:
            raise ValueError("direction balance changed")
        ranges, tokens = [], []
        for direction in count:
            subset = [r for r in rows if r["direction"] == direction]
            for measure in (lambda s: len(s), lambda s: len(s.encode())):
                fraction = sum(0.5 if measure(r["chosen"]) == measure(r["rejected"]) else float(measure(r["chosen"]) > measure(r["rejected"])) for r in subset) / len(subset)
                if not 0.4 <= fraction <= 0.6:
                    raise ValueError("direction/length control failed")
                ranges.append(fraction)
        for style in ("raw", "json", "prose"):
            for row in render_pairs(manifest, split, style):
                if len(row["prompt"].encode()) > 256:
                    raise ValueError("original request exceeds 256 bytes")
                encoded = [tokenizer.encode(render(row["prompt"], row[side])).ids for side in ("chosen", "rejected")]
                if max(map(len, encoded)) > 256 or encoded[0] == encoded[1]:
                    raise ValueError("whole original distinction lost or above 256 tokens")
                tokens.extend(map(len, encoded))
        report[split] = {"source_donor_blocks": len(blocks), "distinct_equivalence_components": len(comps),
                         "rows": len(rows), "direction_counts": dict(count), "length_control_fractions": ranges,
                         "rendered_texts_all_formats": len(tokens), "maximum_tokens": max(tokens),
                         "cross_split_normalized_text_identity_component_overlap": 0,
                         "chosen_role_histogram_equals_rejected": Counter(r["chosen"] for r in rows) == Counter(r["rejected"] for r in rows)}
    return report


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validated_corpus(out):
    """Require an existing pinned corpus; never create a database while auditing."""
    out = out.resolve()
    spec_path = HERE / "general-sources.json"
    spec = json.loads(spec_path.read_text())
    required = {"general.sqlite", "general-index.json", "general-graph.json", "links.tar.bz2", *spec["files"]}
    if not out.is_dir() or any(not (out / name).is_file() for name in required):
        raise ValueError("SOURCE_OUT must contain an existing complete validated corpus cache")
    index = json.loads((out / "general-index.json").read_text())
    graph = json.loads((out / "general-graph.json").read_text())
    if (index["source_spec_sha256"] != sha(spec_path) or index["missing_detailed_identities"] != 0
            or graph["general_spec_sha256"] != sha(spec_path) or graph["links_sha256"] != LINKS_SHA
            or graph["missing_reciprocals"] != 0):
        raise ValueError("SOURCE_OUT corpus reports do not match the frozen source pins")
    return out


def audit(out, kit, tokenizer_path):
    out = validated_corpus(out)
    if sha(tokenizer_path) != TOKENIZER_SHA:
        raise ValueError("released tokenizer pin mismatch")
    import subprocess
    trainer_source = kit / "float_runtime/train_adapter.py"
    pinned = subprocess.run(["git", "show", KIT_REVISION + ":float_runtime/train_adapter.py"], cwd=kit, capture_output=True, timeout=15, check=True).stdout
    if trainer_source.read_bytes() != pinned:
        raise ValueError("upstream kit gate source changed")
    sys.path.insert(0, str(kit))
    from float_runtime import train_adapter as trainer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vectors = json.loads((kit / "float_runtime/testdata/tokenizer_vectors.json").read_text())["vectors"]
    if len(vectors) != 88 or any(tokenizer.encode(v["text"], add_special_tokens=v["add_special_tokens"]).ids != v["ids"] for v in vectors):
        raise ValueError("tokenizer conformance failed")
    manifest = json.loads((HERE / "pilot-manifest.json").read_text())
    if manifest.get("schema_version") == 2:
        pool_path, review_path = HERE / "reviewed-source-pool.json", HERE / "source-pool-review.json"
        if manifest.get("reviewed_pool_sha256") != sha(pool_path) or not manifest.get("source_review_before_split"):
            raise ValueError("source review did not precede this freeze")
        pool = json.loads(pool_path.read_text())
        if (pool["review_sha256"] != sha(review_path) or not pool["source_review_before_split"]
                or pool["proposal_sha256"] != sha(HERE / "source-pool-proposal.json")):
            raise ValueError("reviewed pool binding changed")
        reviewed = {b["id"]: b for b in pool["blocks"]}
        banned = set(manifest["excluded_failed_components"])
        for blocks in manifest["splits"].values():
            for block in blocks:
                if reviewed.get(block["id"]) != block or banned.intersection(block["component_roots"]):
                    raise ValueError("unreviewed block or old exposed component entered the new freeze")
    spec = json.loads((HERE / "general-sources.json").read_text())
    if manifest["source_spec_sha256"] != sha(HERE / "general-sources.json") or manifest["links_sha256"] != LINKS_SHA:
        raise ValueError("frozen source grant/hash binding changed")
    for filename, info in spec["files"].items():
        if sha(out / filename) != info["sha256"]:
            raise ValueError("frozen primary source bytes changed")
    if sha(out / "links.tar.bz2") != LINKS_SHA:
        raise ValueError("frozen link bytes changed")
    db = sqlite3.connect((out / "general.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        for original_row in manifest["originals"].values():
            if original(db, original_row["id"], spec) != original_row:
                raise ValueError("frozen original text, attribution or identity changed")
        for blocks in manifest["splits"].values():
            for block in blocks:
                for (e, f), component, membership_hash in zip(block["links"], block["component_roots"], block["component_membership_sha256"]):
                    row = db.execute("SELECT component FROM pairs WHERE eng=? AND fra=?", (e, f)).fetchone()
                    if row != (component,):
                        raise ValueError("frozen positive is not the original reciprocal link/component")
                    members = [sid for sid, in db.execute("SELECT id FROM membership WHERE component=? ORDER BY id", (component,))]
                    if digest(json.dumps(members)) != membership_hash:
                        raise ValueError("frozen all-language normalized component changed")
    finally:
        db.close()
    splits = {split: render_pairs(manifest, split) for split in COUNTS}
    source_report = checks(manifest, splits, tokenizer, trainer.render)
    data = load_helper("unchanged_training_data", HERE.parent / "training/data.py")
    baselines = data.shortcut_baselines(splits)
    gates = {split: trainer.gate_dataset(rows, trainer.Hyperparams(max_length=256)) for split, rows in splits.items()}
    ready = all(g["ok"] for g in gates.values()) and all(value <= 0.65 for stats in baselines["by_split"].values() for value in stats.values())
    directory = HERE / "out/pilot"
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in splits.items():
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        files[split] = {"sha256": sha(path), "rows": len(rows)}
    if any(n in sys.modules for n in ("torch", "transformers", "peft", "onnxruntime")):
        raise RuntimeError("source-only audit imported neural tooling")
    report = {"family": "LANGUAGE_TRANSLATION", "pilot_manifest_sha256": sha(HERE / "pilot-manifest.json"),
              "source_spec_sha256": sha(HERE / "general-sources.json"), "tokenizer_sha256": TOKENIZER_SHA,
              "kit_revision": KIT_REVISION, "trainer_sha256": sha(trainer_source), "tokenizer_vectors": len(vectors),
              "source_checks": source_report, "dataset_gates": gates, "shortcut_baselines": baselines, "files": files,
              "numeric_data_gates_pass": ready, "semantic_review": manifest["semantic_review"],
              "workflow_ready": False, "model_execution": False,
              "sealed_final_audit": "Single source-only audit after frozen groups; never used for model choice or reselection"}
    semantic_path = HERE / "semantic-review.json"
    if semantic_path.exists():
        semantic = json.loads(semantic_path.read_text())
        if semantic["pilot_manifest_sha256"] != report["pilot_manifest_sha256"]:
            raise ValueError("semantic audit is not bound to this frozen pilot")
        for failure in semantic["known_failures"]:
            block = manifest["splits"][failure["split"]][failure["block_index"]]
            if [failure["english_id"], failure["french_id"]] not in block["links"]:
                raise ValueError("semantic failure references the wrong frozen link")
            if manifest["originals"][str(failure["english_id"])]["text"] != failure["english_evidence"] or manifest["originals"][str(failure["french_id"])]["text"] != failure["french_evidence"]:
                raise ValueError("semantic evidence is not verbatim original text")
        report["semantic_review"] = semantic["status"]
        report["semantic_review_sha256"] = sha(semantic_path)
        report["known_failed_source_links"] = len(semantic["known_failures"])
        report["workflow_ready"] = ready and semantic["status"] == "PASS"
        report["blocker"] = None if report["workflow_ready"] else "Frozen bilingual audit or numeric data gates failed. No training workflow enabled."
    report["answer_only_control"] = {"original_target_role_histograms_balanced_in_every_split": True,
        "reciprocal_pair_bound": 0.5,
        "explanation": "For every target pair the group contains both preference directions. An answer-only ranking wins one and loses one, with half credit for ties. The kit's random row holdout separates those reversed roles, so its unusually low surface scores are not independent semantic evidence."}
    write(HERE / "pilot-checks.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "audit", "final-gate"))
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--kit", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.out)
    elif args.phase == "audit":
        audit(args.out, args.kit, args.tokenizer)
    else:
        report = json.loads((HERE / "pilot-checks.json").read_text())
        if report["pilot_manifest_sha256"] != sha(HERE / "pilot-manifest.json") or report.get("semantic_review_sha256") != sha(HERE / "semantic-review.json"):
            raise ValueError("final gate evidence changed")
        if not report["workflow_ready"]:
            print("NUMERIC DATA VERIFIED. FROZEN BILINGUAL AUDIT FAILED. TRAINING BLOCKED.")
            raise SystemExit(1)
        print("REVIEWED SOURCE POOL AND FIXED NEW DATA GATES PASS. REMOTE MODEL EXECUTION NOT RUN.")


if __name__ == "__main__":
    main()
