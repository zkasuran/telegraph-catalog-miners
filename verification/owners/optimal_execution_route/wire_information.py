# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal-render collision diagnostic. Source checks, never model inference."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from oracle import QUESTION, Oracle, score_wire, state_identity

MAX_ROWS = 4096
MAX_TEXT_BYTES = 16384
MAX_TOTAL_BYTES = 2 << 20
FIELDS = {"source_group", "question", "answer_a", "answer_b", "preferred"}
GROUP = re.compile(r"[0-9a-f]{64}\Z")


def render(question, answer):
    """Literal pinned KIT render expression. Do not normalize the scoring wire."""
    return f"### Task:\n{question}\n\n### Answer:\n{answer}"


def text_bytes(value):
    if not isinstance(value, str):
        raise ValueError("text field must be a string")
    try:
        raw = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("text is not valid UTF-8") from error
    if len(raw) > MAX_TEXT_BYTES:
        raise ValueError("text byte ceiling")
    return raw


def oriented(a, b, label):
    return ((a, b), label) if a <= b else ((b, a), -label)


def pair_hash(pair):
    # Framed lengths avoid concatenation ambiguity. This hashes literal bytes.
    return hashlib.sha256(b"".join(len(p).to_bytes(8, "big") + p for p in pair)).hexdigest()


def ceiling(groups):
    correct = sum(counts.get(0, 0) if pair[0] == pair[1] else max(counts.values())
                  for pair, counts in groups.items())
    return correct


def assess(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("row count ceiling or empty input")
    wire_groups, answer_groups, seen, source_groups = {}, {}, set(), set()
    duplicates, byte_count, same_length = 0, 0, 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("strict diagnostic row schema")
        group = row["source_group"]
        if not isinstance(group, str) or not GROUP.fullmatch(group):
            raise ValueError("source group must be a normalized SHA-256 identity")
        label = row["preferred"]
        if type(label) is not int or label not in (-1, 0, 1):
            raise ValueError("preference must be -1, 0 or 1, not Boolean")
        question, a, b = (text_bytes(row[k]) for k in ("question", "answer_a", "answer_b"))
        byte_count += len(question) + len(a) + len(b)
        if byte_count > MAX_TOTAL_BYTES:
            raise ValueError("aggregate input byte ceiling")
        wire, direction = oriented(render(row["question"], row["answer_a"]).encode(),
                                   render(row["question"], row["answer_b"]).encode(), label)
        identity = (group, wire, direction)
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        source_groups.add(group)
        same_length += len(a) == len(b)
        wire_groups.setdefault(wire, Counter())[direction] += 1
        answer, answer_direction = oriented(a, b, label)
        answer_groups.setdefault(answer, Counter())[answer_direction] += 1
    total = len(seen)
    wire_correct, answer_correct = ceiling(wire_groups), ceiling(answer_groups)
    conflicts = [{"wire_pair_sha256": pair_hash(pair), "counts": {str(k): v for k, v in sorted(counts.items())},
                  "identical_answers": pair[0] == pair[1]}
                 for pair, counts in sorted(wire_groups.items())
                 if len(counts) > 1 or (pair[0] == pair[1] and any(k != 0 for k in counts))]
    return {"schema_version": 1, "family": "OPTIMAL_EXECUTION_ROUTE", "status": "LITERAL_WIRE_DIAGNOSTIC",
            "input_rows": len(rows), "unique_group_pair_label_rows": total, "duplicate_rows_removed": duplicates,
            "declared_source_groups": len(source_groups), "source_groups_independently_verified": False,
            "wire_pair_groups": len(wire_groups), "conflicting_wire_pairs": len(conflicts), "conflicts": conflicts,
            "same_byte_length_pairs": same_length,
            "optimistic_wire_correct_ceiling": {"numerator": wire_correct, "denominator": total},
            "optimistic_answer_only_correct_ceiling": {"numerator": answer_correct, "denominator": total},
            "unavoidable_wire_errors": total - wire_correct,
            "registration_ready": False, "model_operations": 0,
            "limits": ["Exact rendered UTF-8 collisions imply identical tokenizer inputs; tokenization/truncation may cause additional collisions that this source-only check does not measure.",
                       "Ceiling is per-pair optimistic. Scalar-score transitivity can further restrict achievable performance. It is not a measured model accuracy or universal impossibility claim.",
                       "Source-group identities and labels are supplied diagnostic annotations. This function does not verify independent truth or authorize fitting, validation or audit use."]}


def diagnostic(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name in ("evidence/STATE.json", "evidence/STATE-OLDER.json", "primary/render_fragment.py.txt"):
        matching = [r for r in manifest["records"] + manifest["fixtures"] if r["path"] == name]
        if len(matching) != 1:
            raise ValueError("diagnostic source binding absent or ambiguous")
        row = matching[0]
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("diagnostic source checksum mismatch")
    namespace = {}
    exec(compile((root / "primary/render_fragment.py.txt").read_text(), "pinned-primary-render", "exec"), namespace)
    policy = json.loads((root / "policy.json").read_text())
    states = [s for name in ("STATE.json", "STATE-OLDER.json")
              for s in json.loads((root / "evidence" / name).read_text())["states"]]
    oracles = [Oracle(policy, s) for s in states]
    for oracle in oracles:
        oracle.verify_router_observations()
    amounts = sorted({o.best_amount for o in oracles})
    if len(amounts) != 2:
        raise ValueError("original diagnostic requires the retained two quote outputs")
    candidates = [json.dumps({"amount_out": a}) for a in amounts]
    if any(render(QUESTION, a) != namespace["render"](QUESTION, a) for a in candidates):
        raise ValueError("literal primary render contract mismatch")
    rows = []
    for state, oracle in zip(states, oracles):
        scores = [score_wire(QUESTION, a, oracle)["score_ppm"] for a in candidates]
        rows.append({"source_group": state_identity(state), "question": QUESTION,
                     "answer_a": candidates[0], "answer_b": candidates[1],
                     "preferred": (scores[0] > scores[1]) - (scores[0] < scores[1])})
    result = assess(rows)
    result["origin"] = "Retained exposed two-state regression only, not new training, validation or independent audit data."
    result["policy_sha256"] = hashlib.sha256((root / "policy.json").read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-unambiguous", action="store_true")
    args = parser.parse_args()
    try:
        report = diagnostic()
    except (ValueError, KeyError, OSError, UnicodeError) as error:
        print(json.dumps({"status": "DIAGNOSTIC_REFUSED", "error": str(error)}))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 2 if args.require_unambiguous and report["unavoidable_wire_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
