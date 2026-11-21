# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded training/development structure and optional literal-wire diagnostics."""
import argparse
from collections import Counter
from pathlib import Path

from contracts import FAMILY, FORMATS, FOUNDATION, LABELS, check_wire, decode_bits, file_sha, render, sha, strict_json, write_json
from data import normalized, pair

MAX_ROWS = 5000
MAX_BYTES = 16 << 20
MAX_TEXT_BYTES = 16000


def load_rows(path, split):
    if split not in ("train", "validation") or path.stat().st_size > MAX_BYTES:
        raise ValueError("training/development-only bounded input required")
    raw = path.read_bytes()
    rows = [strict_json(line) for line in raw.splitlines() if line.strip()]
    if not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("development row ceiling")
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != split:
            raise ValueError("audit, renamed or mixed-split rows are refused")
    return rows


def checked_pair(row, style, split):
    if not isinstance(row, dict) or split not in ("train", "validation") or row.get("split") != split or row.get("label") not in LABELS:
        raise ValueError("source diagnostics accept training/development labels only")
    for key in ("evidence", "hypothesis", "source_group", "hypothesis_id"):
        if not isinstance(row.get(key), str) or not row[key] or len(row[key].encode()) > MAX_TEXT_BYTES:
            raise ValueError("bounded source identity/text required")
    if row.get("evidence_sha256") != sha(row["evidence"].encode()):
        raise ValueError("literal source evidence identity mismatch")
    if row.get("prompt") is not None and row["prompt"] != row["evidence"]:
        raise ValueError("stored prompt differs from original evidence")
    result = pair(row, style)
    if result["prompt"] != row["evidence"] or result["chosen"] == result["rejected"]:
        raise ValueError("prompt/answer alignment failed")
    if style == "json":
        chosen, rejected = [strict_json(result[side]) for side in ("chosen", "rejected")]
        if chosen["verdict"] != "Unclear" or rejected["verdict"] != "Unclear" or chosen["clauses"] != rejected["clauses"]:
            raise ValueError("a contrast changed the verdict or source quote")
        if len(chosen["obligations"]) != 1 or len(rejected["obligations"]) != 1:
            raise ValueError("one source claim per contrast required")
        if chosen["obligations"][0]["claim"] == rejected["obligations"][0]["claim"]:
            raise ValueError("a source claim contrast is absent")
        for side in ("chosen", "rejected"):
            if row.get(side) is not None and row[side] != result[side]:
                raise ValueError("stored source contrast differs from literal reconstruction")
    return result


def structure(train, validation, encoder=None):
    split_rows = {"train": train, "validation": validation}
    if any(not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS for rows in split_rows.values()):
        raise ValueError("bounded nonempty training/development required")
    for split, rows in split_rows.items():
        for row in rows:
            checked_pair(row, "json", split)
    groups = {name: {row.get("source_group") for row in rows} for name, rows in split_rows.items()}
    evidence = {name: {normalized(row.get("evidence", "")) for row in rows} for name, rows in split_rows.items()}
    if any(len(evidence[name]) != len(rows) for name, rows in split_rows.items()):
        raise ValueError("duplicate normalized source evidence")
    if groups["train"] & groups["validation"] or evidence["train"] & evidence["validation"]:
        raise ValueError("source/entity or normalized evidence overlap")
    report = {"schema_version": 1, "family": FAMILY, "fit_role": "training_and_development_structure_only",
              "independent_audit": False, "neural_operations": "none", "splits": {}}
    for split, rows in split_rows.items():
        labels = Counter(row.get("label") for row in rows)
        if set(labels) != set(LABELS):
            raise ValueError("both original source labels must be covered")
        counts = Counter((row["hypothesis_id"], row["label"]) for row in rows)
        formats = {}
        for style in FORMATS:
            lengths, tokens, checked = [], [], 0
            for row in rows:
                result = checked_pair(row, style, split)
                for side in ("chosen", "rejected"):
                    text = render(result["prompt"], result[side])
                    if len(text.encode()) > MAX_TEXT_BYTES:
                        raise ValueError("literal render byte ceiling")
                    lengths.append(len(text.encode()))
                    if encoder is not None:
                        ids = encoder(text)
                        if not isinstance(ids, list) or not 1 <= len(ids) <= 512 or any(type(t) is not int or not 0 <= t < 50368 for t in ids):
                            raise ValueError("token vector invalid or render would truncate")
                        tokens.append(len(ids))
                    checked += 1
            formats[style] = {"literal_answers_checked": checked, "minimum_render_bytes": min(lengths),
                              "maximum_render_bytes": max(lengths), "tokens_measured": encoder is not None,
                              "maximum_tokens": max(tokens) if tokens else None}
        report["splits"][split] = {"pairs": len(rows), "independent_source_groups": len(groups[split]),
                                  "per_label": dict(labels), "per_hypothesis": {
                                      key: {label: counts[(key, label)] for label in LABELS}
                                      for key in sorted({row["hypothesis_id"] for row in rows})},
                                  "citation_training_pairs": sum(row.get("training_render_policy") == "citation" for row in rows),
                                  "formats": formats}
    return report


def wire_diagnostics(validation, wires, package_keccak):
    if not isinstance(package_keccak, str) or len(package_keccak) != 64 or any(c not in "0123456789abcdef" for c in package_keccak):
        raise ValueError("literal bare package identity required")
    if not isinstance(wires, dict) or set(wires) != set(FORMATS) or not isinstance(validation, list) or not 1 <= len(validation) <= MAX_ROWS:
        raise ValueError("both actual reported wire formats required")
    formats, signs = {}, {}
    for style in FORMATS:
        rows = [checked_pair(row, style, "validation") for row in validation]
        response = wires[style]
        if (not isinstance(response, dict) or not isinstance(response.get("results"), list)
                or any(not isinstance(item, dict) or "score" not in item for item in response["results"])):
            raise ValueError("malformed literal wire response")
        bits = check_wire(wires[style], len(rows) * 2, package_keccak)
        margins = [decode_bits(bits[2 * i]) - decode_bits(bits[2 * i + 1]) for i in range(len(rows))]
        signs[style] = [1 if value > 0 else -1 if value < 0 else 0 for value in margins]
        labels = {}
        for label in LABELS:
            selected = [margins[i] for i, row in enumerate(rows) if row["label"] == label]
            if not selected:
                raise ValueError("wire diagnosis requires both source labels")
            labels[label] = {"pairs": len(selected), "strict_correct": sum(m > 0 for m in selected),
                             "ties": sum(m == 0 for m in selected)}
        formats[style] = {"pairs": len(rows), "strict_correct": sum(m > 0 for m in margins),
                          "strict_accuracy": sum(m > 0 for m in margins) / len(rows),
                          "ties": sum(m == 0 for m in margins), "per_label": labels}
    return {"schema_version": 1, "family": FAMILY, "fit_role": "validation_reported_wire_only",
            "independent_audit": False, "new_inference_performed": False, "package_keccak": package_keccak,
            "formats": formats, "format_sign_disagreements": sum(a != b for a, b in zip(signs["json"], signs["prose"])),
            "limits": ["Input scores are reported wire data, not a fresh verified reference-engine run.",
                       "This diagnostic neither selects a model nor produces registration-ready status."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--wire", type=Path)
    parser.add_argument("--package-keccak")
    args = parser.parse_args()
    train, validation = load_rows(args.train, "train"), load_rows(args.validation, "validation")
    encoder = None
    if args.tokenizer:
        if file_sha(args.tokenizer) != FOUNDATION["tokenizer.json"]:
            raise ValueError("pinned source tokenizer identity mismatch")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        encoder = lambda text: tokenizer.encode(text).ids
    report = structure(train, validation, encoder)
    report["train_sha256"], report["validation_sha256"] = file_sha(args.train), file_sha(args.validation)
    if args.wire:
        if args.wire.stat().st_size > MAX_BYTES:
            raise ValueError("reported wire byte ceiling")
        report["reported_wire"] = wire_diagnostics(validation, strict_json(args.wire.read_bytes()), args.package_keccak)
    elif args.package_keccak:
        raise ValueError("package identity without reported wire data is refused")
    write_json(args.out, report)
    print("Training/development diagnostic bound to actual source bytes. No audit or new inference.")


if __name__ == "__main__":
    main()
