# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Deterministic source-group subsets and immutable integer shortcut coefficients."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from acquire import sha, write
from build import check_splits, shortcuts
from groups import components


def token_counts(text):
    return Counter(re.findall(r"\w+", text.casefold()))


def freeze(data, out):
    report = json.loads((data / "data-report.json").read_text())
    splits = {}
    sizes = {"train": 128, "validation": 32, "audit": 64}
    for split, count in sizes.items():
        raw = (data / (split + ".jsonl")).read_bytes()
        if sha(raw) != report["files"][split]["sha256"]:
            raise ValueError("data hash mismatch")
        rows = [json.loads(line) for line in raw.splitlines()]
        ids = sorted({r["event_id"] for r in rows}, key=lambda x: sha(x.encode()))[:count]
        if len(ids) != count:
            raise ValueError("insufficient frozen source groups")
        splits[split] = [r for r in rows if r["event_id"] in set(ids)]
    check_splits(splits)
    # A train-only additive answer-text baseline. Integers serialize exactly.
    # This is a shortcut diagnostic, not local neural fitting or inference.
    weights = Counter()
    for row in splits["train"]:
        weights.update(token_counts(row["chosen"]))
        weights.subtract(token_counts(row["rejected"]))
    coefficients = {k: v for k, v in sorted(weights.items()) if v}
    out.mkdir(parents=True, exist_ok=True)
    fitted_raw = b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode() for r in splits["train"])
    write(out / "baseline.json", {"training_sha256": sha(fitted_raw),
          "train_universe_sha256": report["files"]["train"]["sha256"],
          "rule": "sum(train chosen token counts minus rejected token counts); tied scores count half",
          "coefficients": coefficients, "length_direction": 1, "overlap_direction": 1,
          "direction_tie_policy": "positive; selected using train only"})
    files = {}
    for split, rows in splits.items():
        raw = b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode() for r in rows)
        (out / (split + ".jsonl")).write_bytes(raw)
        def score_answer(answer):
            return sum(coefficients.get(k, 0)*v for k, v in token_counts(answer).items())
        acc = sum((score_answer(r["chosen"]) > score_answer(r["rejected"])) +
                  0.5*(score_answer(r["chosen"]) == score_answer(r["rejected"])) for r in rows)/len(rows)
        probes = shortcuts(rows)
        probes["train_fitted_answer_unigram"] = acc
        if any(max(v, 1-v) > 0.65 for v in probes.values() if isinstance(v, float)):
            raise ValueError("frozen shortcut gate failed")
        files[split] = {"sha256": sha(raw), "pairs": len(rows), "events": sizes[split], "shortcuts": probes,
                        "independent_groups": components(rows)}
    if files["audit"]["independent_groups"] < 32:
        raise ValueError("audit has fewer than 32 independent event/entity components")
    manifest = {"schema_version": 1, "family": "EVENT_OUTCOME_RESOLUTION", "files": files,
                "source_report_sha256": sha((data / "data-report.json").read_bytes()),
                "source_receipts_sha256": report["source_receipts_sha256"],
                "baseline_sha256": sha((out / "baseline.json").read_bytes()),
                "scope": report["source_scope"], "cross_split_overlap": 0,
                "selection": "smallest SHA256(event_id) within each previously separated time split",
                "audit_neural_status": "never measured; controller independent measurement after validation freeze"}
    write(out / "frozen.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    freeze(args.data, args.out)
