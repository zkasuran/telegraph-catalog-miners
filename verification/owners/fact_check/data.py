# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Frozen primary capital assertions and transparent claim contrasts, no models."""
import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
COUNTS = {"train": 128, "validation": 32, "audit": 64}
AS_OF = date(2026, 11, 18)
SOURCE_REVISION = "wikidata-p36-v1"
TOKENIZER_SHA = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"
TOKENIZER_URL = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/tokenizer.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def label(entity):
    labels = entity.get("labels", {})
    text = next((labels[k]["value"] for k in ("en", "mul") if k in labels), None)
    if not isinstance(text, str) or not text.strip() or len(text) > 80 or re.search(r"[\r\n]", text):
        raise ValueError("missing or oversized primary entity label")
    return text


def applicable(statement):
    qualifiers = statement.get("qualifiers", {})
    if set(qualifiers) - {"P580", "P582"}:
        raise ValueError("unresolved qualifier scope")
    dates = {}
    for field, snaks in qualifiers.items():
        if len(snaks) != 1:
            raise ValueError("multiple time qualifiers")
        snak = snaks[0]
        value = snak.get("datavalue", {}).get("value", {})
        if (snak.get("snaktype") != "value" or value.get("precision") != 11
                or value.get("calendarmodel") != "http://www.wikidata.org/entity/Q1985727"
                or value.get("before") != 0 or value.get("after") != 0 or value.get("timezone") != 0):
            raise ValueError("ambiguous calendar or time precision")
        dates[field] = date.fromisoformat(value["time"][1:11])
        if dates[field] == AS_OF:
            raise ValueError("exact transition day has ambiguous applicable scope")
    if dates.get("P580", date.min) > dates.get("P582", date.max):
        raise ValueError("inverted time scope")
    return dates.get("P580", date.min) <= AS_OF <= dates.get("P582", date.max)


def source_assertion(entity, objects, file_pin):
    ident = entity.get("id")
    if not re.fullmatch(r"Q[1-9][0-9]*", ident or "") or type(entity.get("lastrevid")) is not int:
        raise ValueError("missing exact primary entity/revision")
    subject = label(entity)
    claims = entity.get("claims", {}).get("P36", [])
    alternatives, active, all_labels = set(), [], set()
    for statement in claims:
        if statement.get("rank") == "deprecated":
            continue
        if statement.get("rank") not in ("normal", "preferred"):
            raise ValueError("unknown statement rank")
        snak = statement.get("mainsnak", {})
        value = snak.get("datavalue", {}).get("value", {})
        if (snak.get("property") != "P36" or snak.get("snaktype") != "value"
                or not isinstance(value, dict) or value.get("id") not in objects):
            raise ValueError("unknown or missing capital alternative")
        object_id = value["id"]
        if type(objects[object_id].get("lastrevid")) is not int:
            raise ValueError("missing exact capital label revision")
        object_label = label(objects[object_id])
        alternatives.add(object_id)
        all_labels.add(norm(object_label))
        for aliases in objects[object_id].get("aliases", {}).values():
            all_labels.update(norm(alias["value"]) for alias in aliases)
        if not applicable(statement):
            continue
        references = statement.get("references", [])
        if (not isinstance(statement.get("id"), str) or not statement["id"].startswith(ident + "$")
                or not references or any(not r.get("hash") or not r.get("snaks") for r in references)):
            raise ValueError("active statement lacks bound references")
        active.append({"statement_id": statement["id"], "rank": statement["rank"], "object_id": object_id,
                       "object_label": object_label, "mainsnak": snak,
                       "qualifiers": statement.get("qualifiers", {}), "references": references,
                       "object_revision": objects[object_id].get("lastrevid"),
                       "object_source": objects[object_id].get("_source_pin")})
    if len({s["object_id"] for s in active}) != 1:
        raise ValueError("missing or multiple applicable capitals; never labelled false")
    return {"entity_id": ident, "entity_revision": entity["lastrevid"], "subject_label": subject,
            "relation": "P36", "active_statements": active, "all_non_deprecated_objects": sorted(alternatives),
            "all_accepted_label_aliases": sorted(all_labels), "group_id": ident + "/P36",
            "as_of": AS_OF.isoformat(), "time_scope": "unqualified snapshot assertion" if all(not s["qualifiers"] for s in active) else "explicit day-precision applicable scope",
            "entity_url": f"https://www.wikidata.org/w/index.php?title={ident}&oldid={entity['lastrevid']}",
            "raw_source": file_pin}


def source_pool(source):
    index = json.loads((source / "index.json").read_text())
    if index.get("complete") is not True:
        raise ValueError("source acquisition is incomplete")
    subjects, objects, pins = {}, {}, {}
    for record in index["records"]:
        path = source / record["file"]
        if path.parent != source or path.is_symlink() or path.stat().st_size > 4 << 20:
            raise ValueError("source must be a bounded local regular file")
        raw = path.read_bytes()
        if digest(raw) != record["sha256"]:
            raise ValueError("original source response hash mismatch")
        body = json.loads(raw)
        if path.name.startswith("subjects-"):
            subjects.update(body["entities"])
            pins.update({ident: record for ident in body["entities"]})
        elif path.name.startswith("objects-"):
            objects.update({ident: {**entity, "_source_pin": record} for ident, entity in body["entities"].items()})
    accepted, rejected = [], Counter()
    for ident, entity in subjects.items():
        if ident in {"Q42", "Q22923830"}:
            rejected["previously exposed development entity"] += 1
            continue
        try:
            accepted.append(source_assertion(entity, objects, pins[ident]))
        except (ValueError, KeyError) as exc:
            rejected[str(exc)] += 1
    return accepted, {"observed_subjects": len(subjects), "accepted_source_groups": len(accepted),
                      "exclusions": dict(rejected), "index_sha256": digest((source / "index.json").read_bytes())}


def pair(assertion, candidate, truth):
    subject = assertion["subject_label"]
    correct = assertion["active_statements"][0]["object_label"]
    claim = f"The capital of {subject} is {candidate['object_label']}."
    prompt = (f"Wikidata {assertion['entity_id']} P36, revision {assertion['entity_revision']}: {subject} has capital {correct}.\n"
              f"Scope: {assertion['time_scope']}; as-of {assertion['as_of']}. Claim: {claim}\n"
              "Check consistency with this source only. Use True or False.")
    good = "True" if truth else "False"
    bad = "False" if truth else "True"
    return {"prompt": prompt, "chosen": claim + " Verdict: " + good + ".",
            "rejected": claim + " Verdict: " + bad + ".", "group_id": assertion["group_id"],
            "label": good, "claim_object_id": candidate["object_id"], "claim_object_label": candidate["object_label"],
            "source_domain": "Wikidata", "source_revision": SOURCE_REVISION,
            "chosen_rank": 0, "rejected_rank": 1, "margin": 0.0,
            "source_binding": assertion,
            "construction": "AI-assisted deterministic claim/value and verdict substitution; not human preference labels"}


def allocate(assertions, tokenize):
    pools = defaultdict(list)
    seen_names = set()
    for assertion in sorted(assertions, key=lambda a: digest(("allocation-v1" + a["group_id"]).encode())):
        name_key = norm(assertion["subject_label"])
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        bucket = int(digest(("split-v1" + assertion["group_id"]).encode()), 16) % 20
        name = "train" if bucket < 12 else "validation" if bucket < 15 else "audit"
        pools[name].append(assertion)
    rows, availability = {}, {}
    for name, count in COUNTS.items():
        candidates = pools[name]
        bins = {True: [], False: []}
        for index, assertion in enumerate(candidates):
            truth = index % 2 == 0
            correct = assertion["active_statements"][0]
            if truth:
                candidate = correct
            else:
                ordered_wrong = sorted(candidates, key=lambda other: digest(("negative-v1" + assertion["group_id"] + other["active_statements"][0]["object_id"]).encode()))
                candidate = next((other["active_statements"][0] for other in ordered_wrong
                                  if other["active_statements"][0]["object_id"] not in assertion["all_non_deprecated_objects"]
                                  and norm(other["active_statements"][0]["object_label"]) not in assertion["all_accepted_label_aliases"]), None)
                if candidate is None:
                    continue
            row = pair(assertion, candidate, truth)
            ids = [tokenize(render(row["prompt"], row[side])) for side in ("chosen", "rejected")]
            if max(map(len, ids)) > 256 or ids[0] == ids[1]:
                continue
            bins[truth].append(row)
        availability[name] = {"True": len(bins[True]), "False": len(bins[False]), "required_each": count // 2}
        rows[name] = sorted(bins[True][:count // 2] + bins[False][:count // 2], key=lambda r: r["group_id"])
    complete = all(len(rows[name]) == count for name, count in COUNTS.items())
    return rows, availability, complete


def check_splits(splits):
    seen = set()
    seen_text = set()
    for rows in splits.values():
        groups = {r["group_id"] for r in rows}
        if len(groups) != len(rows) or seen & groups:
            raise ValueError("entity/relation cross-split leakage")
        seen.update(groups)
        texts = {norm(row[field]) for row in rows for field in ("prompt", "chosen", "rejected")}
        if seen_text & texts:
            raise ValueError("normalized factual text cross-split leakage")
        seen_text.update(texts)
        for row in rows:
            source = row["source_binding"]
            if row["group_id"] != source["entity_id"] + "/P36":
                raise ValueError("source group identity mismatch")
            good = row["label"]
            if good not in ("True", "False") or not row["chosen"].endswith(f"Verdict: {good}."):
                raise ValueError("candidate label direction mismatch")
            active = {s["object_id"] for s in source["active_statements"]}
            if (row["claim_object_id"] in active) != (good == "True"):
                raise ValueError("source-bound claim direction wrong")
            if good == "False" and (row["claim_object_id"] in source["all_non_deprecated_objects"]
                                     or norm(row["claim_object_label"]) in source["all_accepted_label_aliases"]):
                raise ValueError("wrong claim is an accepted alternative")


def prepare(out, cache=None):
    """Replay frozen outputs only. Never reselect examples during a model job."""
    del cache
    from source_pack import unpack
    unpack()
    contract = json.loads((HERE / "contract.json").read_text())
    bundle = HERE / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in contract["freeze_sha256"].items():
        if name == "sources.json":
            path = HERE / name
        else:
            path = bundle / name
        raw = path.read_bytes()
        if digest(raw) != expected:
            raise ValueError("frozen preparation artifact changed: " + name)
        if name != "sources.json":
            target = out / name
            if target.is_file() and target.read_bytes() != raw:
                raise ValueError("existing frozen output changed")
            target.write_bytes(raw)
    (out / "sources").mkdir(exist_ok=True)
    for name in ("tokenizer.json",):
        raw = (HERE / "sources" / name).read_bytes()
        if digest(raw) != TOKENIZER_SHA:
            raise ValueError("primary tokenizer pin mismatch")
        (out / "sources" / name).write_bytes(raw)
    (out / "LICENSE").write_bytes((HERE / "SOURCE-CC0.txt").read_bytes())
    (out / "NOTICE").write_bytes((HERE / "NOTICE").read_bytes())
    return read_splits(out)[0]


def read_splits(out):
    report = json.loads((out / "data-report.json").read_text())
    splits = {}
    for name, pin in report["files"].items():
        raw = (out / (name + ".jsonl")).read_bytes()
        if digest(raw) != pin["sha256"] or len(raw) > 8 << 20:
            raise ValueError("frozen split hash or byte limit")
        splits[name] = [json.loads(line) for line in raw.decode().split("\n") if line]
        if len(splits[name]) != pin["rows"]:
            raise ValueError("frozen split count mismatch")
    check_splits(splits)
    return report, splits


def freeze(source):
    from tokenizers import Tokenizer
    if (HERE / "contract.json").exists():
        raise ValueError("freeze already exists; no adaptive reselection")
    assertions, recount = source_pool(source)
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    if digest((source / "tokenizer.json").read_bytes()) != TOKENIZER_SHA:
        raise ValueError("tokenizer not primary pinned bytes")
    tokenizer.no_padding()
    tokenizer.no_truncation()
    rows, availability, complete = allocate(assertions, lambda text: tokenizer.encode(text).ids)
    check_splits(rows)
    bundle = HERE / "bundle"
    bundle.mkdir(exist_ok=True)
    files = {}
    for name, selected in rows.items():
        raw = "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in selected).encode()
        (bundle / (name + ".jsonl")).write_bytes(raw)
        files[name] = {"rows": len(selected), "groups": len(selected), "sha256": digest(raw),
                       "label_counts": dict(Counter(r["label"] for r in selected))}
    report = {"status": "SOURCE_ONLY" if complete else "INSUFFICIENT_REAL_GROUPS",
              "training_allowed": complete, "scope": "Consistency with referenced nondeprecated Wikidata P36 assertions, not universal fact-checking",
              "files": files, "available_groups": availability, "source_recount": recount,
              "as_of": AS_OF.isoformat(), "final_audit_frozen_before_training": True,
              "source_manifest_sha256": digest((HERE / "sources.json").read_bytes()),
              "source_pins": json.loads((HERE / "sources.json").read_text()),
              "sampling_policy_sha256": digest((HERE / "freeze-policy.json").read_bytes()),
              "wrong_claims_excluded_from_all_alternatives": True, "synthetic_source_facts": False}
    write_json(bundle / "data-report.json", report)
    if not complete:
        print(json.dumps({key: value for key, value in report.items() if key != "source_pins"}, indent=2))
        raise ValueError("precise source insufficiency; no training permission, padding or lower gates")
    # Only train contributes fitted coefficients. Audit is not scored here.
    from gates import fit_controls
    _, controls = fit_controls(rows["train"], tokenizer)
    write_json(bundle / "frozen-controls.json", controls)
    token_binding = {name: [{"group_id": row["group_id"], **{side: {
        "tokens": len(tokenizer.encode(render(row["prompt"], row[side])).ids),
        "render_sha256": digest(render(row["prompt"], row[side]).encode())} for side in ("chosen", "rejected")}}
        for row in selected] for name, selected in rows.items()}
    write_json(bundle / "token-visible-binding.json", {"splits": token_binding, "max_tokens": 256, "cropped": False})
    pins = {path.name: digest(path.read_bytes()) for path in bundle.iterdir() if path.is_file()}
    pins["sources.json"] = digest((HERE / "sources.json").read_bytes())
    contract = {"splits": files, "freeze_sha256": pins,
                "hyperparameters": {"max_length": 256, "lora_rank": 8, "lora_alpha": 32, "batch_size": 1,
                                    "grad_accum": 16, "epochs": 3, "eval_batch": 1, "seed": 42},
                "quality": {"minimum_accuracy": .80, "minimum_subset_accuracy": .70, "minimum_untrained_gain": .05,
                            "minimum_shortcut_gain": .10, "maximum_shortcut_accuracy": .65}}
    write_json(HERE / "contract.json", contract)
    baselines = HERE / "baselines"
    baselines.mkdir(exist_ok=True)
    (baselines / "frozen-controls.json").write_bytes((bundle / "frozen-controls.json").read_bytes())
    write_json(baselines / "manifest.json", {"schema_version": 1, "artifact": "frozen-controls.json",
               "artifact_sha256": pins["frozen-controls.json"], "fit_split": "train", "training_rows": 128,
               "training_sha256": files["train"]["sha256"], "source": "Wikidata", "source_revision": SOURCE_REVISION,
               "source_manifest_sha256": pins["sources.json"], "tokenizer_sha256": TOKENIZER_SHA})
    print(json.dumps({key: value for key, value in report.items() if key != "source_pins"}, indent=2))


def tokenizer(source):
    target = source / "tokenizer.json"
    request = urllib.request.Request(TOKENIZER_URL, headers={"User-Agent": "curl/8.0 FactcheckTokenizer/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read((8 << 20) + 1)
    if len(raw) > 8 << 20 or digest(raw) != TOKENIZER_SHA:
        raise ValueError("released tokenizer byte/hash mismatch")
    target.write_bytes(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tokenizer", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.tokenizer:
        tokenizer(args.source)
    elif args.source:
        freeze(args.source)
    else:
        print(json.dumps({key: value for key, value in prepare(args.out).items() if key != "source_pins"}, indent=2))
