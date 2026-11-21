# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Fresh identity-quarantined source splits and explicit task-authoritative preferences."""
import argparse
from collections import Counter
import hashlib
from contextlib import contextmanager
import json
import random
from pathlib import Path

from contracts import FAMILY, FROZEN, HERE, fetch, file_sha, strict_json
from retrain_data import SOURCE_FILE, download, dump
from source_data import group_id, inventory
import v3_data as origin

COUNTS = {"train": 144, "validation": 36, "audit": 36}
QUARANTINE = HERE / "v4-quarantine.json"
LABELS = origin.LABELS
BASELINE = origin.BASELINE
EXPLANATIONS = {"Positive": "The text expresses a positive overall tone.", "Negative": "The text expresses a negative overall tone.",
                "Neutral": "The text expresses a neutral overall tone.", "Mixed": "The text combines positive and negative tones."}


@contextmanager
def grouping_policy():
    previous = origin.COUNTS, origin.QUARANTINE
    origin.COUNTS, origin.QUARANTINE = COUNTS, QUARANTINE
    try:
        yield
    finally:
        origin.COUNTS, origin.QUARANTINE = previous


def pairs(groups, style=None, training=False):
    """No answer copies the task. Untrusted quoted excerpts are balanced across labels."""
    by_label = {label: sorted((row for row in groups if row["label"] == label), key=lambda row: row["group_id"]) for label in LABELS}
    n = len(by_label[LABELS[0]])
    if n < 4 or n % 4 or any(len(rows) != n for rows in by_label.values()):
        raise ValueError("balanced bounded source labels required")
    result = []
    for label in LABELS:
        for index, group in enumerate(by_label[label]):
            text = group["text"]
            prompt = (json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) if index % 4 == 0 else
                      text if index % 4 == 1 else "Classify the overall sentiment of this text: " + text if index % 4 == 2 else
                      "Choose Positive, Negative, Neutral or Mixed for this text.\n" + text)
            selected = (2 if index % 64 == 0 else index % 2) if training and style is None else style
            if selected not in (0, 1, 2):
                raise ValueError("all supported response styles are explicit")
            # The exact same unrelated excerpt is used once with each task label.
            # Its class is uniform and independent of the task class, including
            # same-label controls. It never reveals a cyclic correct-label mapping.
            distractor = by_label[LABELS[index % 4]][(index + 1) % n]
            if distractor["group_id"] == group["group_id"]:
                raise ValueError("unrelated excerpt must not be the task")
            words = distractor["text"].split()
            excerpt = " ".join(words[:4] + (["..."] + words[-4:] if len(words) > 8 else words[4:]))
            for condition in ("task_only", "untrusted_excerpt"):
                for wrong in LABELS:
                    if wrong != label:
                        references = [distractor]
                        reference_reason = None
                        if training:
                            # Meaningful, independent source contexts, not hashes,
                            # answer IDs or padding used to evade repetition checks.
                            references = []
                            for offset in (0, 1):
                                kind = LABELS[(index + offset) % 4]
                                seed = (group["group_id"] + wrong + condition + str(offset)).encode()
                                picked = int(hashlib.sha256(seed).hexdigest(), 16) % n
                                if kind == label and picked == index:
                                    picked = (picked + 1) % n
                                references.append(by_label[kind][picked])
                            snippets = []
                            for ref in references:
                                words = ref["text"].split()
                                snippets.append(" ".join(words[:4] + (["..."] + words[-4:] if len(words) > 8 else words[4:])))
                            reference_reason = ("Independent reference: " if condition == "task_only" else "Quoted excerpt: ") + snippets[0] + "; contrast reference: " + snippets[1]

                        def answer(value):
                            if selected == 2:
                                return value
                            reason = reference_reason or (EXPLANATIONS[value] if condition == "task_only" else "Quoted excerpt: " + excerpt)
                            if selected == 1:
                                return f"Sentiment: {value}. Reason: {reason}"
                            return json.dumps({"sentiment": value, "score": {"Positive": 1, "Negative": -1, "Neutral": 0, "Mixed": 0}[value],
                                               "reason": reason}, ensure_ascii=False, separators=(",", ":"))
                        built = {"prompt": prompt, "chosen": answer(label), "rejected": answer(wrong), "label": label,
                                       "wrong_label": wrong, "group_id": group["group_id"], "condition": condition,
                                       "support_group_id": distractor["group_id"] if condition == "untrusted_excerpt" else group["group_id"],
                                       "label_kind": "preference_pair", "margin": 0.0}
                        if training:
                            built.update(condition="reference_context" if condition == "task_only" else condition,
                                         support_group_id=references[0]["group_id"], support_group_ids=[ref["group_id"] for ref in references])
                        result.append(built)
    return result


def training_structure(rows, eval_fraction=.15, seed=42):
    """Exact primary non-neural checks. Its learned surface probe remains remote."""
    n = len(rows)
    if not n:
        raise ValueError("empty source preferences")
    rejected = Counter(row["rejected"].strip() for row in rows)
    indexes = list(range(n))
    random.Random(seed).shuffle(indexes)
    n_eval = max(1, int(n * eval_fraction)) if n > 1 else 0
    held = indexes[:n_eval]
    seen = {rows[i]["rejected"].strip() for i in indexes[n_eval:]}
    leak = sum(rows[i]["rejected"].strip() in seen for i in held)/len(held) if held else 0
    longer = sum(len(row["chosen"]) > len(row["rejected"]) for row in rows)
    tied = sum(len(row["chosen"]) == len(row["rejected"]) for row in rows)
    length = (longer + tied*.5)/n
    metrics = {"unique_rejected_frac": len(rejected)/n, "top_rejected_share": max(rejected.values())/n,
               "eval_negative_leak": leak, "length_shortcut_acc": max(length, 1-length),
               "identical_fraction": sum(row["chosen"].strip() == row["rejected"].strip() for row in rows)/n}
    return {"structural_passed": metrics["unique_rejected_frac"] >= .8 and metrics["top_rejected_share"] <= .02 and
                                 metrics["eval_negative_leak"] <= .1 and metrics["length_shortcut_acc"] <= .65 and metrics["identical_fraction"] <= .01,
            "metrics": metrics, "scope": "Primary repetition/split/length checks only, not its learned surface probe or neural quality."}


def prepare(root, raw_root=None):
    root.mkdir(parents=True, exist_ok=True)
    raw_root = raw_root or root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    spec = strict_json(SOURCE_FILE.read_bytes())
    if spec.get("stage") != "verified":
        raise ValueError("original complete source grants required")
    for name, item in spec["files"].items():
        download(item["url"], raw_root / name, item["sha256"])
    rows, inventory_report = origin.unanimous_inventory(raw_root, strict_json((raw_root / "sentiment_mapping.json").read_bytes()))
    excluded = root / "excluded-uci.zip"
    fetch(FROZEN["source_url"], excluded, FROZEN["source_sha256"], 1 << 20)
    uci, _ = inventory(excluded)
    with grouping_policy():
        groups, selection = origin.select_groups(rows, {group_id(row["text"]) for row in uci})
    files = {}
    structural = None
    for split, values in groups.items():
        dump(root / (split + "-groups.json"), values)
        built = pairs(values, training=True) if split == "train" else pairs(values, 0)
        if split == "train":
            structural = training_structure(built)
            if not structural["structural_passed"]:
                raise ValueError("primary training uniqueness/reuse/leak/length checks failed")
        if any(support not in {group["group_id"] for group in values} for row in built for support in row.get("support_group_ids", [row["support_group_id"]])):
            raise ValueError("quoted source escaped the grouped split")
        (root / (split + ".jsonl")).write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in built))
        files[split] = {"sha256": file_sha(root / (split + ".jsonl")), "groups_sha256": file_sha(root / (split + "-groups.json")),
                        "pairs": len(built), "independent_groups": len(values)}
    (root / "LICENSE").write_bytes((raw_root / "LICENSE").read_bytes())
    manifest = {"schema_version": 1, "version": 4, "family": FAMILY, "sources_sha256": file_sha(SOURCE_FILE),
                "quarantine_sha256": file_sha(QUARANTINE), "baseline_sha256": file_sha(BASELINE),
                "baseline_scope": "Immutable coefficients fitted only on original v3 training, reused unchanged. No fresh floating-point fit.",
                "inventory": inventory_report, "selection": selection, "files": files, "normalized_overlap": 0,
                "training_structure": structural,
                "audit_used_for_selection": False, "grants_verified": True, "independent_groups": len(groups["audit"]),
                "construction": "Original human emotion annotations, programmatic unanimous valence/composition labels and task-authoritative preferences. No human sentiment-preference review.",
                "scope": "Short English emotion-valence proxy, constructed Mixed and some other compositions. Both answers share untrusted excerpts; category correctness, not explanation truth, determines preferences."}
    dump(root / "source-manifest.json", manifest)
    return groups, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    _, report = prepare(args.root, args.raw_root)
    print(json.dumps(report, indent=2))
