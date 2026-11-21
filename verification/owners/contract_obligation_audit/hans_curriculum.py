# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Original synthetic training only. Non-entailment is not factual contradiction."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import math

from contracts import FORMATS, file_sha, render, sha, strict_json, write_json
from data import download, normalized

REVISION = "7299f6f657089ce06a0f98e7e81f8d0f5b7741ce"
TRAIN_SHA256 = "440040b6790c6359b56496375b87e6cedfa13a20f52bc3223dad31bd28b77faf"
URL = "https://raw.githubusercontent.com/tommccoy1/hans/" + REVISION + "/heuristics_train_set.jsonl"
LABELS = ("entailment", "non-entailment")
CURRICULUM_ROWS = 320


def check_row(row):
    if not isinstance(row, dict) or row.get("gold_label") not in LABELS:
        raise ValueError("original synthetic entailment/non-entailment labels required")
    for key in ("sentence1", "sentence2", "pairID", "heuristic", "subcase", "template"):
        if not isinstance(row.get(key), str) or not row[key] or len(row[key].encode()) > 4096:
            raise ValueError("bounded original source row required")


def source_rows(path):
    if path.is_symlink() or path.stat().st_size > 32 << 20 or file_sha(path) != TRAIN_SHA256:
        raise ValueError("exact original HANS training source required")
    rows, ids, counts, subcases = [], set(), Counter(), Counter()
    for line in path.read_bytes().splitlines():
        row = strict_json(line)
        if not isinstance(row, dict) or row.get("gold_label") not in LABELS or row.get("pairID") in ids:
            raise ValueError("original training source identity or label differs")
        for key in ("sentence1", "sentence2", "pairID", "heuristic", "subcase", "template"):
            if not isinstance(row.get(key), str) or not row[key] or len(row[key].encode()) > 4096:
                raise ValueError("bounded original training source text required")
        ids.add(row["pairID"])
        counts[row["gold_label"]] += 1
        subcases[row["subcase"]] += 1
        rows.append(row)
    if len(rows) != 30000 or counts != {label: 15000 for label in LABELS} or len(subcases) != 30 or any(n != 1000 for n in subcases.values()):
        raise ValueError("complete original HANS training inventory differs")
    return rows


def select(rows, maximum=CURRICULUM_ROWS):
    if not isinstance(rows, list) or type(maximum) is not int or maximum < 2 or maximum % 2:
        raise ValueError("even original training curriculum size required")
    by_hypothesis = defaultdict(lambda: defaultdict(list))
    for row in rows:
        check_row(row)
        by_hypothesis[normalized(row["sentence2"])][row["gold_label"]].append(row)
    selected, used = [], set()
    for hypothesis in sorted(by_hypothesis, key=lambda text: sha(text.encode())):
        groups = by_hypothesis[hypothesis]
        if any(not groups[label] for label in LABELS):
            continue
        choices = []
        for label in LABELS:
            candidates = sorted(groups[label], key=lambda row: sha((row["pairID"] + row["sentence1"]).encode()))
            candidate = next((row for row in candidates if normalized(row["sentence1"]) not in used
                              and all(normalized(row["sentence1"]) != normalized(other["sentence1"]) for other in choices)), None)
            if candidate is None:
                break
            choices.append(candidate)
        if len(choices) != 2:
            continue
        selected.extend(choices)
        used.update(normalized(row["sentence1"]) for row in choices)
        if len(selected) == maximum:
            break
    if len(selected) != maximum:
        raise ValueError("no complete balanced source-hypothesis curriculum at requested size")
    return selected


def preference(row, style):
    check_row(row)
    if row.get("gold_label") not in LABELS or style not in FORMATS:
        raise ValueError("original synthetic support labels and formats required")
    premise, hypothesis = row["sentence1"], row["sentence2"]
    positive = "The supplied text establishes that " + hypothesis
    negative = "The supplied text does not establish that " + hypothesis
    # A non-entailment annotation does not assert not-hypothesis. It includes
    # neutral. Both preferences concern textual support, not world performance.
    chosen, rejected = (positive, negative) if row["gold_label"] == "entailment" else (negative, positive)

    def answer(statement):
        if style == "json":
            return json.dumps({"verdict": "Unclear", "obligations": [{"claim": statement, "status": "source_supported"}],
                               "clauses": [premise], "summary": statement + " Contractual performance is not assessed."}, separators=(",", ":"))
        return "Verdict: Unclear. Source-supported inference: " + statement + " Clause: " + premise + " Contractual performance is not assessed."

    return {"prompt": premise, "chosen": answer(chosen), "rejected": answer(rejected), "margin": 0., "label_kind": "preference_pair",
            "original_label": row["gold_label"], "source_revision": REVISION, "source_pair_id": row["pairID"],
            "source_partition": "original_training", "source_group": "hans-premise:" + sha(normalized(premise).encode()),
            "hypothesis_id": "hans-hypothesis:" + sha(normalized(hypothesis).encode()), "evidence": premise,
            "evidence_sha256": sha(premise.encode()),
            "authorship": "Original programmatically generated HANS labels; AI-assisted support preferences, not human review. Non-entailment is not factual contradiction."}


def shortcuts(rows):
    coefficients = Counter()
    prior = defaultdict(Counter)
    words = lambda text: Counter(re.findall(r"[a-z0-9']+", text.casefold()))
    for row in rows:
        coefficients.update(words(row["chosen"]))
        coefficients.subtract(words(row["rejected"]))
        prior[row["hypothesis_id"]][row["original_label"]] += 1
    values = defaultdict(list)
    for row in rows:
        chosen, rejected, prompt = [words(row[key]) for key in ("chosen", "rejected", "prompt")]
        expected = max(LABELS, key=lambda label: prior[row["hypothesis_id"]][label])
        differences = {"length": len(row["chosen"]) - len(row["rejected"]),
            "unigram_overlap": sum((chosen & prompt).values()) - sum((rejected & prompt).values()),
            "answer_only_bow": sum((chosen[word] - rejected[word]) * count for word, count in coefficients.items()),
            "exact_quote": int(row["evidence"] in row["chosen"]) - int(row["evidence"] in row["rejected"]),
            "hypothesis_prior": 1 if row["original_label"] == expected else -1}
        for key, difference in differences.items():
            values[key].append(.5 if difference == 0 else float(difference > 0))
    measured = {key: sum(value) / len(value) for key, value in values.items()}
    directions = {key: 1 if value >= .5 else -1 for key, value in measured.items()}
    return {key: max(value, 1 - value) for key, value in measured.items()}, {
        "fit_split": "synthetic_original_training_only", "rows": len(rows),
        "bow_integer_coefficients": {word: count for word, count in sorted(coefficients.items()) if count},
        "train_fitted_directions": directions,
        "hypothesis_original_label_counts": {key: dict(value) for key, value in sorted(prior.items())}}


def prepare(root, source=None):
    root.mkdir(exist_ok=True)
    path = source if source is not None else root / "original-training.jsonl"
    if source is None:
        download(URL, path, TRAIN_SHA256, 32 << 20)
    originals = source_rows(path)
    chosen = select(originals)
    views = [preference(row, style) for row in chosen for style in FORMATS]
    measured = {}
    coefficients = {}
    for style in FORMATS:
        rows = [preference(row, style) for row in chosen]
        measured[style], coefficients[style] = shortcuts(rows)
    if any(max(value.values()) > .65 for value in measured.values()):
        raise ValueError("synthetic training shortcut exceeds unchanged source integrity gate")
    if len({row["source_group"] for row in views}) != len(chosen):
        raise ValueError("normalized curriculum premise overlap")
    report = {"schema_version": 1, "source_revision": REVISION, "source_sha256": TRAIN_SHA256,
        "original_training_rows_read": len(originals), "selected_originals": len(chosen), "selected_views": len(views),
        "original_labels": dict(Counter(row["gold_label"] for row in chosen)),
        "heuristics": dict(Counter(row["heuristic"] for row in chosen)), "shortcuts": measured,
        "coefficients": coefficients, "evaluation_rows_read": False, "synthetic_auxiliary_training_only": True,
        "human_preference_reviews": False, "new_audited_contract_claim": False,
        "selected_pair_ids": [row["pairID"] for row in chosen],
        "selected_normalized_premise_sha256": [sha(normalized(row["sentence1"]).encode()) for row in chosen],
        "selected_originals_sha256": sha(json.dumps(chosen, sort_keys=True, separators=(",", ":")).encode()),
        "licence": "MIT, copyright 2019 R. Thomas McCoy; retain original notice"}
    write_json(root / "curriculum-manifest.json", report)
    return chosen, views, report


def require_curriculum(freeze, manifest, manifest_hash):
    if freeze.get("syntactic_curriculum_originals", 0) == 0:
        return
    if (freeze.get("syntactic_curriculum_originals") != CURRICULUM_ROWS or not isinstance(manifest, dict)
            or manifest.get("source_revision") != REVISION or manifest.get("source_sha256") != TRAIN_SHA256
            or manifest.get("original_training_rows_read") != 30000 or manifest.get("selected_originals") != CURRICULUM_ROWS
            or manifest.get("selected_views") != 2 * CURRICULUM_ROWS or manifest.get("synthetic_auxiliary_training_only") is not True
            or manifest.get("evaluation_rows_read") is not False or manifest.get("human_preference_reviews") is not False
            or manifest.get("new_audited_contract_claim") is not False
            or any(manifest.get(key) != freeze.get(key) for key in ("family", "run_id", "commit", "source_manifest_sha256"))
            or not isinstance(manifest_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", manifest_hash)
            or freeze.get("syntactic_curriculum_manifest_sha256") != manifest_hash):
        raise ValueError("actual original-training curriculum does not bind the frozen candidate")
    ids = manifest.get("selected_pair_ids")
    if (not isinstance(ids, list) or len(ids) != CURRICULUM_ROWS or any(not isinstance(value, str) for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("exact unique original training curriculum identities required")
    hashes = manifest.get("selected_normalized_premise_sha256")
    if (not isinstance(hashes, list) or len(hashes) != CURRICULUM_ROWS
            or any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value) for value in hashes)
            or len(set(hashes)) != len(hashes)):
        raise ValueError("exact normalized original curriculum premises required")
    if manifest.get("original_labels") != {label: CURRICULUM_ROWS // 2 for label in LABELS}:
        raise ValueError("original balanced entailment/non-entailment classes required")
    probes = manifest.get("shortcuts")
    if not isinstance(probes, dict) or set(probes) != set(FORMATS):
        raise ValueError("both curriculum source formats required")
    for metrics in probes.values():
        if (not isinstance(metrics, dict) or set(metrics) != {"length", "unigram_overlap", "answer_only_bow", "exact_quote", "hypothesis_prior"}
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= .65 for value in metrics.values())):
            raise ValueError("unchanged curriculum source-integrity cutoff failed")
