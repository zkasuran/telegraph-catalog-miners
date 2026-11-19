# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Immutable diagnostic for archived records, not a live judgment adapter.

No trained model, no counterfeit inference from seller claims and no external I/O.
The lookup authenticates the interpretation of a published specimen record only.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
MAX_PRODUCT_BYTES = 256
MAX_ANSWER_BYTES = 4096
DOI = "10.1371/journal.pmed.0050032.st001"
PREFIX = "Archived study specimen "
SCOPE = "Archived record only. This does not authenticate a new physical item."
NO_EVIDENCE = "No specimen-bound evidence is available. False means not judged authentic, not proven counterfeit."
ALLOWED = ("Authentic", "Likely authentic", "Likely counterfeit", "Counterfeit", "Inconclusive")


def strict_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key: " + key)
        out[key] = value
    return out


def reject_constant(value):
    raise ValueError("non-finite JSON: " + value)


def decode_answer(answer):
    if not isinstance(answer, str) or len(answer.encode("utf-8")) > MAX_ANSWER_BYTES:
        raise ValueError("answer exceeds string/byte limit")
    try:
        obj = json.loads(answer, object_pairs_hook=strict_object, parse_constant=reject_constant)
    except (RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON answer") from exc
    if not isinstance(obj, dict) or set(obj) != {"product", "verdict", "authentic", "evidence"}:
        raise ValueError("answer must have the exact live response fields")
    if not isinstance(obj["product"], str) or obj["verdict"] not in ALLOWED or type(obj["authentic"]) is not bool:
        raise ValueError("invalid live field types")
    evidence = obj["evidence"]
    if not isinstance(evidence, list) or len(evidence) > 12 or any(not isinstance(v, str) for v in evidence):
        raise ValueError("evidence must be a bounded string list")
    return obj


def records():
    inventory = json.loads((ROOT / "SOURCE-INVENTORY.json").read_text())
    return {r["record_id"]: r for r in inventory["records"]}


def product_for(record_id):
    return PREFIX + record_id + "; doi:" + DOI


def resolve(product, source_records):
    if not isinstance(product, str) or len(product.encode("utf-8")) > MAX_PRODUCT_BYTES:
        raise ValueError("product exceeds the live 256-byte limit")
    # This deliberately does not normalise the string the kit receives.
    match = re.fullmatch(re.escape(PREFIX) + r"(newton2008-st001-row[0-9]+); doi:" + re.escape(DOI), product)
    return source_records.get(match.group(1)) if match else None


def expectation(product, source_records):
    record = resolve(product, source_records)
    if record is None:
        return {"product": product, "verdict": "Inconclusive", "authentic": False, "evidence": [NO_EVIDENCE]}
    # S18 combines different physical packages. Never erase that source limit.
    if record["specific_limits"]:
        return {"product": product, "verdict": "Inconclusive", "authentic": False,
                "evidence": [*record["specific_limits"], SCOPE]}
    label = record["source_label"]
    return {"product": product, "verdict": label, "authentic": label == "Authentic",
            "evidence": ["Source: doi:" + DOI, "HPLC record: " + record["hplc"], SCOPE]}


def score(product, answer, source_records):
    """Return fixed integer diagnostic points. No fitting or floating point."""
    want = expectation(product, source_records)
    try:
        got = decode_answer(answer)
    except (ValueError, UnicodeError, TypeError):
        return {"score": 0, "reason": "invalid_response", "mode": "diagnostic_only"}
    if got["product"] != product:
        return {"score": 0, "reason": "wrong_product_echo", "mode": "diagnostic_only"}
    if got["verdict"] != want["verdict"] or got["authentic"] != want["authentic"]:
        return {"score": 0, "reason": "unsupported_verdict", "mode": "diagnostic_only"}
    # Exact evidence is a narrow source-extraction check, not semantic judgment.
    if got["evidence"] != want["evidence"]:
        return {"score": 0, "reason": "unbound_evidence_or_scope", "mode": "diagnostic_only"}
    return {"score": 10000, "reason": "source_record_interpretation", "mode": "diagnostic_only"}


def main():
    import sys
    data = sys.stdin.buffer.read(MAX_PRODUCT_BYTES + MAX_ANSWER_BYTES + 2048)
    if len(data) >= MAX_PRODUCT_BYTES + MAX_ANSWER_BYTES + 2048:
        raise SystemExit("input envelope exceeds byte limit")
    try:
        obj = json.loads(data, object_pairs_hook=strict_object, parse_constant=reject_constant)
        if not isinstance(obj, dict) or set(obj) != {"product", "answer"}:
            raise ValueError("expected product and answer strings")
        result = score(obj["product"], obj["answer"], records())
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
