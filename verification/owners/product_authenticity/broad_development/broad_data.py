# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Broader source-grounded preferences from original facts, never model audits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "development"
sys.path.insert(0, str(DEV))
from data_contract import source_rows, tag, alternate_status  # noqa: E402

FAMILY = "PRODUCT_AUTHENTICITY"
CHILD = FAMILY + "_GENERAL"
FORMATS = ("json", "prose")
DESCRIPTION = "Product, serial number or listing"
MAX_PRODUCT_BYTES = 256
MAX_QUESTION_BYTES = 1024
MAX_ANSWER_BYTES = 2048
MAX_TOKENS = 512
TRAIN_FORMS = (
    "The archive assigns {a} to {u}; {b} is not its recorded assessment.",
    "For {u}, read the recorded status as {a}, not {b}.",
    "{b} is not the archived finding for {u}; the finding is {a}.",
    "The status attached to {u} is {a}. Substituting {b} would change the record.",
)
VALIDATION_FORMS = (
    "According to the retained assessment, {u} falls under {a} rather than {b}.",
    "Calling the archived specimen {u} {b} would contradict its {a} assessment.",
    "The record supports {a} for {u}. It does not support replacing that with {b}.",
    "For this reference, {a} describes {u}; {b} does not describe its assessed status.",
)
FACT_FORMS = {
    "train": ("For {u}, the retained detail is {a}; {b} is a different detail.",
              "Attribute {a}, not {b}, to the record identified as {u}.",
              "Do not substitute {b} for {a} in the archived description of {u}.",
              "{u} is associated with {a}; reporting {b} would misstate that detail."),
    "validation": ("Reading the archive literally, {u} has {a} rather than {b}.",
                   "The entry for {u} supports the detail {a}. The detail {b} is not that entry's value.",
                   "For the specimen designated {u}, replacing {a} with {b} changes the source fact.",
                   "The retained value for {u} is {a}; the alternative value {b} belongs elsewhere.")}
SCOPE_FORMS = {
    "train": ("The source statement covers {a}, not {b}; it cannot certify another physical object.",
              "{b} is outside the source statement's scope. The archived statement applies to {a}.",
              "Keep the assessment attached to {a}; do not transfer it to {b} on identifier similarity alone.",
              "A finding about {a} is not a finding about {b}. Matching identifiers do not bind two objects."),
    "validation": ("This evidence concerns {a} alone. It does not establish the identity or authenticity of {b}.",
                   "The archived assessment of {a} cannot be reused as an assessment of {b} without specimen linkage.",
                   "For {b}, this source supplies no specimen-bound determination; its determination concerns {a}.",
                   "The named reference {a} is the subject of the source conclusion, whereas {b} remains outside it.")}
BINDING_FORMS = {
    "train": ("Assign the detail {f} to {a}, not {b}.",
              "{b} is not the entry carrying {f}; that detail is attached to {a}.",
              "The detail {f} is part of the {a} record. Moving it to {b} would misattribute the source.",
              "Keep {f} bound to {a}, rather than attaching it to {b}."),
    "validation": ("In the retained record, {f} belongs under {a} instead of {b}.",
                   "The source associates {a} with {f}; it does not assign that detail to {b}.",
                   "Attributing {f} to {b} would confuse the entries: the relevant entry is {a}.",
                   "The reference for {f} is {a}, while {b} is a different reference.")}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def typed_question(product):
    if len(product.encode()) > MAX_PRODUCT_BYTES:
        raise ValueError("typed product exceeds live 256-byte contract")
    return f"{CHILD} request. product ({DESCRIPTION}): {product}."


def task_from_prompt(prompt):
    prefix = f"{CHILD} request. product ({DESCRIPTION}): "
    raw = prompt[len(prefix):-1] if prompt.startswith(prefix) and prompt.endswith(".") else prompt
    return json.loads(raw)


def answer(style, task, first, second, kind, formulation, split, reverse):
    u = task["unit"]
    expected = task["status"] if task["requested"] == u else "Inconclusive"
    verdict = alternate_status(expected) if reverse and kind == "verdict" else expected
    a, b = (second, first) if reverse else (first, second)
    if kind == "verdict":
        forms = TRAIN_FORMS if split == "train" else VALIDATION_FORMS
        evidence = forms[formulation].format(a=a, b=b, u=u)
    elif kind == "fact":
        evidence = FACT_FORMS[split][formulation].format(a=a, b=b, u=u)
    elif kind == "scope":
        evidence = SCOPE_FORMS[split][formulation].format(a=a, b=b)
    else:
        evidence = BINDING_FORMS[split][formulation].format(a=a, b=b, f=task["fact"])
    product = "Archived medicinal unit " + task["requested"]
    if style == "json":
        return json.dumps({"product": product, "verdict": verdict, "authentic": verdict == "Authentic", "evidence": [evidence]},
                          ensure_ascii=False, separators=(",", ":"))
    return f"Product {product}\nVerdict {verdict}\nAuthentic {'true' if verdict == 'Authentic' else 'false'}\nEvidence {evidence}"


def make_pairs(records, split):
    pairs = []
    for index, row in enumerate(records):
        alternatives = [r for r in records if r["fact"] != row["fact"]]
        contrast = alternatives[index % len(alternatives)]
        u, other = tag(row), tag(contrast)
        for kind_index, kind in enumerate(("verdict", "fact", "scope", "binding")):
            same = kind != "scope" or index % 2 == 0
            task = {"unit": u, "status": row["status"], "fact": row["fact"], "contrast": other,
                    "contrast_fact": contrast["fact"], "requested": u if same else other, "scope": "record-backed inquiry"}
            product = json.dumps(task, ensure_ascii=False, separators=(",", ":"))
            if len(product.encode()) > MAX_PRODUCT_BYTES:
                raise ValueError("actual product payload exceeds live byte cap")
            first, second = (row["status"], alternate_status(row["status"])) if kind == "verdict" else (row["fact"], contrast["fact"]) if kind == "fact" else (u, other)
            for view in ("direct", "typed"):
                formulation = (index + kind_index + (2 if view == "typed" else 0)) % 4
                template = ((TRAIN_FORMS if split == "train" else VALIDATION_FORMS) if kind == "verdict" else FACT_FORMS[split] if kind == "fact" else SCOPE_FORMS[split] if kind == "scope" else BINDING_FORMS[split])[formulation]
                prompt = typed_question(product) if view == "typed" else product
                for style in FORMATS:
                    good = answer(style, task, first, second, kind, formulation, split, False)
                    bad = answer(style, task, first, second, kind, formulation, split, True)
                    length = max(len(good.encode()), len(bad.encode()))
                    good += " " * (length - len(good.encode()))
                    bad += " " * (length - len(bad.encode()))
                    pairs.append({"prompt": prompt, "product_input": product, "chosen": good, "rejected": bad, "margin": 0.0,
                                  "format": style, "kind": kind, "label": row["status"] if same else "Inconclusive", "view": view,
                                  "formulation_group": kind + ":" + digest(template.encode()),
                                  "source_record_id": row["id"], "source_group": row["source"], "entity_group": row["entity"], "time_group": row["time"],
                                  "group_id": row["source"] + ":" + row["id"], "source_annotation": row["source_annotation"], "split": split,
                                  "preference_provenance": "New AI-assisted constructed report comparison from original source facts. Not human-reviewed and not new physical-item truth."})
    return pairs


def check_splits(splits):
    if set(splits) != {"train", "validation"} or any(not rows for rows in splits.values()):
        raise ValueError("nonempty development split required; independent audit remains unavailable")
    for field in ("source_group", "entity_group", "time_group", "group_id", "formulation_group"):
        a = {canonical(r[field]) for r in splits["train"]}
        b = {canonical(r[field]) for r in splits["validation"]}
        if a & b:
            raise ValueError("group overlap: " + field)
    for field in ("prompt", "chosen", "rejected"):
        a = {canonical(r[field]) for r in splits["train"]}
        b = {canonical(r[field]) for r in splits["validation"]}
        if a & b:
            raise ValueError("normalized overlap: " + field)
    for rows in splits.values():
        for row in rows:
            if len(row["product_input"].encode()) > MAX_PRODUCT_BYTES or len(row["prompt"].encode()) > MAX_QUESTION_BYTES:
                raise ValueError("literal product/question boundary")
            if row["prompt"] != (typed_question(row["product_input"]) if row["view"] == "typed" else row["product_input"]):
                raise ValueError("literal primary request helper expression changed")
            if row["source_record_id"] in row["prompt"] or row["chosen"] == row["rejected"]:
                raise ValueError("source ID leakage or empty preference")
            if len(row["chosen"].encode()) != len(row["rejected"].encode()) or len(row["chosen"].encode()) > MAX_ANSWER_BYTES:
                raise ValueError("answer byte matching/ceiling")
        rejected = Counter(r["rejected"] for r in rows)
        if len(rejected) / len(rows) < .8 or max(rejected.values()) / len(rows) > .02:
            raise ValueError("stock-answer surface shortcut")
    return True


def response_verdict(answer_text):
    if answer_text.lstrip().startswith("{"):
        return json.loads(answer_text)["verdict"]
    match = re.search(r"^Verdict (.+)$", answer_text, re.M)
    if not match:
        raise ValueError("missing verdict in control input")
    return match.group(1)


def source_copy_score(prompt, answer_text):
    task = task_from_prompt(prompt)
    expected = task["status"] if task["requested"] == task["unit"] else "Inconclusive"
    return int(response_verdict(answer_text).strip() == expected)


def overlap_score(prompt, answer_text):
    # Fixed lexical set overlap, no fitting or entity-label metadata.
    words = lambda s: set(re.findall(r"[a-z0-9]+", s.casefold()))
    return len(words(prompt) & words(answer_text))


def control_accuracy(rows, scorer):
    scores = [(scorer(r["prompt"], r["chosen"]), scorer(r["prompt"], r["rejected"])) for r in rows]
    return sum(1 if a > b else .5 if a == b else 0 for a, b in scores) / len(rows)


def source_controls(rows):
    controls = {}
    for style in FORMATS:
        chosen = [r for r in rows if r["format"] == style]
        controls[style] = {"source_status_copy": control_accuracy(chosen, source_copy_score),
                           "lexical_overlap": control_accuracy(chosen, overlap_score),
                           "byte_length": control_accuracy(chosen, lambda p, a: len(a.encode()))}
    return controls


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sources = source_rows()
    splits = {name: make_pairs(rows, name) for name, rows in sources.items()}
    check_splits(splits)
    files = {}
    for split, rows in splits.items():
        raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        (directory / (split + ".jsonl")).write_bytes(raw)
        files[split] = {"sha256": digest(raw), "pairs": len(rows), "groups": len({r["group_id"] for r in rows})}
    report = {"schema_version": 1, "family": FAMILY, "scope": "Broader source-grounded report faithfulness and source/request binding. Not universal physical product authentication.",
              "files": files, "source_counts": {name: len(rows) for name, rows in sources.items()}, "formats": list(FORMATS),
              "views": ["direct", "typed"], "max_product_bytes": MAX_PRODUCT_BYTES, "max_question_bytes": MAX_QUESTION_BYTES,
              "max_tokens": MAX_TOKENS, "normalized_overlap": 0, "formulation_groups_disjoint": True,
              "source_annotations_not_preferences": True, "audit_available": False, "audit_pairs": 0, "registration_ready": False,
              "source_controls": {split: source_controls(rows) for split, rows in splits.items()},
              "limits": ["Typed question expression is verified against the primary library, not the deployed judgment caller.",
                         "Validation formulations and sources are held out from training. This is development selection, not an independent audit.",
                         "Original source annotations/grants stay attached. New response preferences are AI-assisted, not human-reviewed."]}
    (directory / "data-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report, splits


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output)[0], sort_keys=True))
