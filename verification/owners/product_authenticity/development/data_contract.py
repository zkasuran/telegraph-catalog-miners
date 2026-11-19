# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-grounded constructed preferences. No neural imports or fitting here."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from source_admission import extract as metformin_extract
from source_inventory import sanada_records

HERE = Path(__file__).resolve().parent
FAMILY = "PRODUCT_AUTHENTICITY"
CHILD = FAMILY + "_GENERAL"
FORMATS = ("json", "prose")
KINDS = ("verdict", "fact", "scope", "binding")


def canonical(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def source_rows():
    raw = (HERE / "sources/newton-annotations.json").read_bytes()
    if digest(raw) != "45f02de315d9a2757b67b0d215c9e9238bc216e2d713327b4f85deb0ed48ed18":
        raise ValueError("Newton original source annotations changed")
    original = json.loads(raw)["records"]
    metformin = metformin_extract((HERE / "sources/metformin.xml").read_bytes())["records"]
    sanada_raw = (HERE / "sources/sanada.xml").read_bytes()
    if digest(sanada_raw) != "fccbe8928d93a261927661ff907483cbc420cdef1b3988efde3591131a4192e1":
        raise ValueError("Sanada original source changed")
    sanada = sanada_records(sanada_raw)
    train, validation = [], []
    for row in original:
        status = "Inconclusive" if row["specific_limits"] else row["source_label"]
        train.append({"id": row["record_id"], "source": row["source_doi"].removesuffix(".st001"),
                      "entity": row["entity_group"], "time": row["acquisition_event"], "status": status,
                      "source_annotation": row["source_label"], "fact": "batch " + row["batch_dates"].split()[0]})
    for row in metformin:
        train.append({"id": row["record_id"], "source": row["source_doi"], "entity": row["entity_group"],
                      "time": row["acquisition_time_group"], "status": row["diagnostic_verdict"],
                      "source_annotation": row["source_label"], "fact": "site " + str(row["source_site_no"])})
    for row in sanada:
        validation.append({"id": row["record_id"], "source": row["source_doi"], "entity": row["entity_group"],
                           "time": row["cohort_group"], "status": row["source_label"],
                           "source_annotation": row["source_label"], "fact": row["brand"] + " " + row["dose"]})
    return {"train": train, "validation": validation}


def tag(row):
    # A technical fixture tag, not an invented real serial or a source label.
    return "u" + digest((row["source"] + ":" + row["id"]).encode())[:10]


def alternate_status(status):
    return "Authentic" if status != "Authentic" else "Counterfeit"


def answer(style, prompt, status, first, second, reverse, kind, unit, contrast_unit, requested_same=True):
    if kind == "verdict":
        verdict = alternate_status(status) if reverse else status
        a, b = (alternate_status(status), status) if reverse else (status, alternate_status(status))
        evidence = f"For {unit}, the source status is {a}, not {b}."
    elif kind == "fact":
        verdict = status
        a, b = (second, first) if reverse else (first, second)
        evidence = f"For {unit}, the recorded fact is {a}, not {b}."
    elif kind == "scope":
        verdict = status if requested_same else "Inconclusive"
        a, b = (contrast_unit, unit) if reverse else (unit, contrast_unit)
        evidence = f"The source statement concerns {a}, not {b}."
    else:
        verdict = status
        a, b = (contrast_unit, unit) if reverse else (unit, contrast_unit)
        evidence = f"The reported fact {first} belongs to {a}, not {b}."
    if style == "json":
        return json.dumps({"product": "Described medicinal specimen", "verdict": verdict, "authentic": verdict == "Authentic", "evidence": [evidence]},
                          ensure_ascii=False, separators=(",", ":"))
    # Both answers share the same template and vocabulary. No stock bad style.
    return f"Product Described medicinal specimen\nVerdict {verdict}\nAuthentic {'true' if verdict == 'Authentic' else 'false'}\nEvidence {evidence}"


def make_pairs(rows, split):
    pairs = []
    for index, row in enumerate(rows):
        alternatives = [r for r in rows if r["fact"] != row["fact"]]
        if not alternatives:
            raise ValueError("source facts cannot form contrasting reports")
        contrast = alternatives[index % len(alternatives)]
        unit, other = tag(row), tag(contrast)
        for kind in KINDS:
            requested_same = kind != "scope" or index % 2 == 0
            scope = "record-backed inquiry"
            prompt = json.dumps({"unit": unit, "status": row["status"], "fact": row["fact"],
                                 "contrast": other, "contrast_fact": contrast["fact"], "requested": unit if requested_same else other, "scope": scope},
                                ensure_ascii=False, separators=(",", ":"))
            if len(prompt.encode()) > 256:
                raise ValueError("literal product exceeds live byte contract")
            for style in FORMATS:
                chosen = answer(style, prompt, row["status"], row["fact"], contrast["fact"], False, kind, unit, other, requested_same)
                rejected = answer(style, prompt, row["status"], row["fact"], contrast["fact"], True, kind, unit, other, requested_same)
                # Match lengths by JSON/prose whitespace, not by changing claims.
                length = max(len(chosen.encode()), len(rejected.encode()))
                chosen += " " * (length - len(chosen.encode()))
                rejected += " " * (length - len(rejected.encode()))
                pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "margin": 0.0,
                              "format": style, "kind": kind, "label": row["status"],
                              "group_id": row["source"] + ":" + row["id"], "source_group": row["source"],
                              "entity_group": row["entity"], "time_group": row["time"], "split": split,
                              "source_annotation": row["source_annotation"], "source_record_id": row["id"],
                              "preference_provenance": "New AI-assisted constructed report preference from primary source facts, not human-reviewed; no new physical-item truth."})
    return pairs


def check_splits(splits):
    if set(splits) != {"train", "validation"} or any(not rows for rows in splits.values()):
        raise ValueError("nonempty train and validation required; audit is separately unavailable")
    for field in ("source_group", "entity_group", "time_group", "group_id"):
        a = {canonical(r[field]) for r in splits["train"]}
        b = {canonical(r[field]) for r in splits["validation"]}
        if a & b:
            raise ValueError("group overlap: " + field)
    for side in ("prompt", "chosen", "rejected"):
        a = {canonical(r[side]) for r in splits["train"]}
        b = {canonical(r[side]) for r in splits["validation"]}
        if a & b:
            raise ValueError("normalized train/validation overlap: " + side)
    for rows in splits.values():
        for row in rows:
            if row["chosen"] == row["rejected"] or len(row["prompt"].encode()) > 256:
                raise ValueError("invalid literal contract or no preference signal")
            if row["source_record_id"] in row["prompt"]:
                raise ValueError("source identifier is exposed to the model")
            if len(row["chosen"].encode()) != len(row["rejected"].encode()):
                raise ValueError("length mismatch")
    return True


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sources = source_rows()
    splits = {name: make_pairs(rows, name) for name, rows in sources.items()}
    check_splits(splits)
    files = {}
    for name, rows in splits.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        (directory / (name + ".jsonl")).write_bytes(raw)
        files[name] = {"sha256": digest(raw), "pairs": len(rows), "groups": len({r["group_id"] for r in rows})}
    report = {"schema_version": 1, "family": FAMILY, "scope": "source-grounded answer faithfulness and uncertainty development, not physical product certification",
              "files": files, "source_counts": {name: len(rows) for name, rows in sources.items()},
              "labels": {name: dict(Counter(r["status"] for r in rows)) for name, rows in sources.items()},
              "normalized_overlap": 0, "source_identifier_masked": True, "answer_formats": list(FORMATS),
              "preference_types": list(KINDS), "audit_available": False, "audit_pairs": 0,
              "registration_ready": False, "source_annotations_not_preferences": True,
              "limits": ["Input is an explicit archived evidence snapshot. It is not proof about another physical item.",
                         "Source train/validation groups are disjoint. This is not a chronological-generalization claim.",
                         "Audit groups and the deployed caller observation delivery are still unverified."]}
    (directory / "data-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report, splits


if __name__ == "__main__":
    import sys
    print(json.dumps(build(sys.argv[1])[0], sort_keys=True))
