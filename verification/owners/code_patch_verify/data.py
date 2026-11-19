# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Original fictional CI consistency pairs. No source logs or protected rows."""
import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from ci_reference import render, score

HERE = Path(__file__).resolve().parent
SPLITS = {"train": (128, 0, "2026-11-01"), "validation": (32, 1000, "2026-11-10"),
          "audit": (64, 10000, "2026-11-18")}
NON_SUCCESS = ("failure",)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text):
    return re.findall(r"[a-z0-9_]+", normalized(text))


def fixture(split, index):
    _, start, date = SPLITS[split]
    seed = start + index
    repository = f"cpv-fixture/{split}-{seed:05d}"
    run_id = 900000 + seed
    commit = digest(f"original-{split}-{seed}".encode())[:40]
    attempt = index % 3 + 1
    url = f"https://github.com/{repository}/actions/runs/{run_id}"
    # Source outcomes are balanced, with distinct attempts and no real event claim.
    other = NON_SUCCESS[(index // 2) % len(NON_SUCCESS)]
    conclusion = "success" if index % 2 == 0 else other
    run = {"id": run_id, "run_attempt": attempt, "repository": {"full_name": repository},
           "head_sha": commit, "html_url": url, "status": "queued" if conclusion == "pending" else "completed",
           "conclusion": None if conclusion == "pending" else conclusion}
    evidence = {"transport_status": 200, "observed_at": date + "T12:00:00Z", "run": run}
    request = {"patch": url + f"/attempts/{attempt}"}
    chosen = {"passed": conclusion == "success", "conclusion": conclusion, "commit": commit, "pipeline": url}
    rejected = {**chosen, "passed": conclusion != "success", "conclusion": other if conclusion == "success" else "success"}
    # Both claims are internally coherent. Only the prompt distinguishes them.
    if split == "train":
        prompt = json.dumps({"patch": request["patch"], "ci_observation": evidence}, separators=(",", ":"))
        answers = [json.dumps(a, separators=(",", ":")) for a in (chosen, rejected)]
    elif split == "validation":
        prompt = json.dumps({"ci_observation": evidence, "patch": request["patch"]}, sort_keys=True)
        answers = [json.dumps(a, sort_keys=True) for a in (chosen, rejected)]
    else:
        prompt = json.dumps({"patch": request["patch"], "ci_observation": evidence}, indent=1)
        answers = [json.dumps(a, indent=1) for a in (chosen, rejected)]
    return {"SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0", "schema_version": 1,
            "label_kind": "preference_pair", "family": "CODE_PATCH_VERIFY",
            "original_id": f"{repository}:{run_id}", "repository": repository, "commit": commit,
            "run_id": run_id, "run_attempt": attempt, "observed_at": evidence["observed_at"],
            "seed": seed, "source_kind": "original_fictional_observation",
            "preference_authorship": "constructed_by_AI_assisted_code_not_human_reviewed",
            "mutation": "flip_passed_and_conclusion_together_preserving_format_and_identity",
            "prompt": prompt, "chosen": answers[0], "rejected": answers[1], "margin": 0.0}


def check_splits(splits):
    if set(splits) != set(SPLITS):
        raise ValueError("all independent splits required")
    seen = {key: set() for key in ("repository", "commit", "run_id", "original_id", "prompt", "answers")}
    report = {}
    for name in SPLITS:
        rows = splits[name]
        if len(rows) != SPLITS[name][0]:
            raise ValueError("unexpected original count")
        current = {key: set() for key in seen}
        for row in rows:
            for key in ("repository", "commit", "original_id"):
                current[key].add(normalized(row[key]))
            current["run_id"].add(str(row["run_id"]))
            for key in ("repository", "commit", "run_id", "original_id"):
                if current[key] & seen[key]:
                    raise ValueError("normalized split overlap: " + key)
            current["prompt"].add(normalized(row["prompt"]))
            current["answers"].update(normalized(row[side]) for side in ("chosen", "rejected"))
            if normalized(row["chosen"]) == normalized(row["rejected"]):
                raise ValueError("identical preference")
            source = json.loads(row["prompt"])
            run = source["ci_observation"]["run"]
            if (row["commit"] != run["head_sha"] or row["run_id"] != run["id"]
                    or row["run_attempt"] != run["run_attempt"] or row["repository"] != run["repository"]["full_name"]
                    or row["observed_at"] != source["ci_observation"]["observed_at"]
                    or not row["observed_at"].startswith(SPLITS[name][2])):
                raise ValueError("source/entity/time provenance mismatch")
            request = {"patch": source["patch"]}
            good = score(request, row["chosen"], source["ci_observation"])
            bad = score(request, row["rejected"], source["ci_observation"])
            if good["score"] != 1 or bad["score"] != 0 or "passed" not in bad["mismatches"]:
                raise ValueError("independent reference disagrees with fixture construction")
        for key in current:
            if current[key] & seen[key]:
                raise ValueError("normalized split overlap: " + key)
            seen[key].update(current[key])
        if len(current["original_id"]) != len(rows) or len(current["prompt"]) != len(rows):
            raise ValueError("duplicate original within split")
        longer = sum(len(row["chosen"]) > len(row["rejected"]) for row in rows)
        ties = sum(len(row["chosen"]) == len(row["rejected"]) for row in rows)
        report[name] = {"rows": len(rows), "original_run_groups": len(current["original_id"]),
                        "chosen_longer": longer, "chosen_shorter": len(rows) - longer - ties,
                        "length_ties": ties, "maximum_rendered_utf8_bytes": max(
                            len(render(row["prompt"], row[side]).encode()) for row in rows for side in ("chosen", "rejected"))}
    return {"splits": report, "normalized_overlap": {key: 0 for key in seen},
            "scope": "Source-visible consistency, not opaque-URL live retrieval"}


def freeze_controls(train):
    """Exact integer frequency differences from train only, never floating fitting."""
    coefficients = Counter()
    longer_twice = 0
    for row in train:
        coefficients.update(tokens(row["chosen"]))
        coefficients.subtract(tokens(row["rejected"]))
        longer_twice += 2 * (len(row["chosen"]) > len(row["rejected"])) + (len(row["chosen"]) == len(row["rejected"]))
    return {"SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0", "schema_version": 1,
            "fit_split": "train", "train_sha256": digest(jsonl(train)),
            "answer_only_coefficients": dict(sorted(coefficients.items())),
            "length_direction": 1 if longer_twice >= len(train) else -1,
            "tie_credit": "1/2", "method": "train_only_integer_term_frequency_difference"}


def control_metrics(rows, controls):
    result = {}
    for control in ("answer_only", "length"):
        twice = 0
        for row in rows:
            if control == "length":
                difference = controls["length_direction"] * (len(row["chosen"]) - len(row["rejected"]))
            else:
                weights = controls["answer_only_coefficients"]
                difference = sum(weights.get(t, 0) for t in tokens(row["chosen"])) - sum(
                    weights.get(t, 0) for t in tokens(row["rejected"]))
            twice += 2 if difference > 0 else int(difference == 0)
        result[control] = {"wins_twice": twice, "pairs": len(rows), "accuracy": twice / (2 * len(rows))}
    return result


def jsonl(rows):
    return ("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)).encode()


def prepare(root):
    root.mkdir(exist_ok=True)
    old = root / "data-report.json"
    if old.is_file():
        destination = root / ("previous-" + digest(old.read_bytes())[:16])
        if not destination.exists():
            destination.mkdir()
            for path in root.iterdir():
                if path.is_file():
                    shutil.copyfile(path, destination / path.name)
    splits = {name: [fixture(name, i) for i in range(count)] for name, (count, _, _) in SPLITS.items()}
    report = check_splits(splits)
    report["generator_sha256"] = digest(Path(__file__).read_bytes())
    report["files"] = {}
    for name, rows in splits.items():
        raw = jsonl(rows)
        (root / (name + ".jsonl")).write_bytes(raw)
        report["files"][name] = {"sha256": digest(raw), "rows": len(rows)}
    controls = freeze_controls(splits["train"])
    write_json(root / "frozen-controls.json", controls)
    report["controls_sha256"] = digest((root / "frozen-controls.json").read_bytes())
    report["control_metrics"] = {name: control_metrics(rows, controls) for name, rows in splits.items()}
    report["training_allowed"] = False
    report["template_limit"] = "Different serialization layouts are held out, but all fictional observations share one semantic generator. This is not independently collected source-held-out factual evidence."
    report["blocker"] = "Controller must verify trusted CI observation presence in literal live scorer prompt"
    write_json(root / "data-report.json", report)
    first = splits["train"][0]
    source = json.loads(first["prompt"])
    smoke = {"request": {"patch": source["patch"]}, "answer": first["chosen"], "evidence": source["ci_observation"]}
    write_json(root / "smoke.json", smoke)
    write_json(root / "tampered.json", {**smoke, "answer": first["rejected"]})
    print(json.dumps({"originals": {name: len(rows) for name, rows in splits.items()},
                      "overlap": report["normalized_overlap"], "controls": report["control_metrics"],
                      "training_allowed": False}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "data")
    prepare(parser.parse_args().out)
