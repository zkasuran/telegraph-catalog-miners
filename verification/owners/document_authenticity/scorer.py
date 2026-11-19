# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Deterministic report-consistency scorer. No model or network operations."""
import argparse
import hashlib
import json
import re
from pathlib import Path

MAX_JSON = 8192
FIELDS = ("integrity", "whole_document")
HASH = re.compile(r"[0-9a-f]{64}")
PACKAGE = re.compile(r"BILLS-(\d+)(?:hr|s|hjres|sjres|hconres|sconres)\d+enr")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def strict_json(raw):
    if not isinstance(raw, str) or len(raw.encode()) > MAX_JSON:
        raise ValueError("invalid JSON byte ceiling")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite JSON value")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def validate_report(report):
    if not isinstance(report, dict) or set(report) != {"package", "sha256", "integrity", "whole_document", "trust"}:
        raise ValueError("report field contract")
    if not isinstance(report["package"], str) or not PACKAGE.fullmatch(report["package"]):
        raise ValueError("invalid package identity")
    if not isinstance(report["sha256"], str) or not HASH.fullmatch(report["sha256"]):
        raise ValueError("invalid byte identity")
    if any(type(report[field]) is not bool for field in FIELDS):
        raise ValueError("measured fields must be booleans")
    if report["trust"] != "unchecked":
        raise ValueError("this verifier cannot establish signing-time trust")


def score(prompt, raw_answer):
    """Score only against the caller-supplied report, never answer self-evidence.

    The controller must authenticate report delivery. This function checks binding
    but cannot establish report origin. Native evidence-free tasks are unsupported.
    """
    try:
        task = strict_json(prompt)
        if set(task) not in ({"document", "report"}, {"document", "report", "claim_values"}):
            raise ValueError("trusted report missing or unexpected fields")
        if "claim_values" in task and (task["claim_values"] != [True, False] or any(type(v) is not bool for v in task["claim_values"])):
            raise ValueError("boolean claim grammar")
        report = task["report"]
        validate_report(report)
        expected_url = f"https://www.govinfo.gov/content/pkg/{report['package']}/pdf/{report['package']}.pdf"
        if task["document"] != expected_url:
            raise ValueError("literal document binding differs")
        answer = strict_json(raw_answer)
        if set(answer) != {"document", "verdict", "evidence"} or answer["document"] != task["document"]:
            raise ValueError("answer contract or document identity")
        if answer["verdict"] != "Inconclusive":
            raise ValueError("unverified authenticity assertion")
        evidence = answer["evidence"]
        if not isinstance(evidence, list) or len(evidence) != 1:
            raise ValueError("one evidence claim required")
        claim = evidence[0]
        if not isinstance(claim, dict) or set(claim) != {"sha256", "field", "value"}:
            raise ValueError("claim contract")
        if claim["sha256"] != report["sha256"] or claim["field"] not in FIELDS or type(claim["value"]) is not bool:
            raise ValueError("claim identity, field or type")
        return {"supported": True, "score": int(claim["value"] == report[claim["field"]]),
                "scope": "report_consistency", "authentic": None}
    except (ValueError, TypeError, KeyError, RecursionError):
        return {"supported": False, "score": 0, "scope": "unsupported", "authentic": None}


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def make_pair(report, field, group, split, variant):
    validate_report(report)
    url = f"https://www.govinfo.gov/content/pkg/{report['package']}/pdf/{report['package']}.pdf"
    # State both legal claim values as protocol grammar. Otherwise token overlap
    # solves homogeneous reports without binding a value to the selected field.
    # This is grammar, not a second source observation or invented fact.
    prompt = compact({"document": url, "claim_values": [True, False], "report": report})
    def answer(value):
        return compact({"document": url, "verdict": "Inconclusive", "evidence": [
            {"sha256": report["sha256"], "field": field, "value": value}]})
    return {"prompt": prompt, "chosen": answer(report[field]), "rejected": answer(not report[field]),
            "label_kind": "preference_pair", "group_id": group, "source_id": report["package"],
            "split": split, "variant": variant, "field": field,
            "construction": "AI-assisted programmatic report interpretation; not human reviewed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    if args.fixture.stat().st_size > 32768:
        raise ValueError("fixture too large")
    fixture = json.loads(args.fixture.read_text())
    print(compact(score(fixture["prompt"], fixture["chosen"])))


if __name__ == "__main__":
    main()
