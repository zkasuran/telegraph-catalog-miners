# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Human-source grouping and prompt-conditioned moderation preference controls."""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORMATS = ("json", "prose", "bare_label")
MAX_ROWS = 100000


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def group_id(text):
    return sha(" ".join(unicodedata.normalize("NFKC", text).casefold().split()).encode())


def source_spec():
    value = json.loads((HERE / "aegis-sources.json").read_bytes())
    if value.get("dataset_id") != "nvidia/Aegis-AI-Content-Safety-Dataset-2.0" or value.get("upstream_license") != "CC-BY-4.0" or value.get("revision") != "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c":
        raise ValueError("primary source/grant revision differs")
    expected = {"train.json", "refusals_train.json", "validation.json", "refusals_validation.json", "test.json"}
    if len(value["files"]) != 5 or {r["path"] for r in value["files"]} != expected:
        raise ValueError("complete five-file source inventory required")
    for row in value["files"]:
        if not re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) or not 0 < row["bytes"] <= 32 << 20:
            raise ValueError("bounded exact original file identity required")
    return value


def grouped(rows, excluded, max_bytes=512):
    states, admitted, counts = {}, {}, Counter()
    scanned = 0
    for file, split, raw in rows:
        scanned += 1
        if scanned > MAX_ROWS:
            raise ValueError("complete source scan ceiling")
        counts["file:" + file] += 1
        counts["prompt_annotation:" + str(raw.get("prompt_label_source"))] += 1
        counts["response_annotation:" + str(raw.get("response_label_source"))] += 1
        text = raw.get("prompt")
        if not isinstance(text, str) or raw.get("reconstruction_id_if_redacted") not in (None, "") or text.strip() == "REDACTED":
            counts["redacted_or_missing"] += 1
            continue
        text = text.strip()
        label = {"safe": "Safe", "unsafe": "Flagged"}.get(raw.get("prompt_label")) if raw.get("prompt_label_source") == "human" else None
        group = group_id(text)
        states.setdefault(group, set()).add((split, label))
        if label is None or group in excluded or not 8 <= len(text.encode()) <= max_bytes:
            counts["not_admitted"] += 1
            continue
        row = {"text": text, "label": label, "group_id": group, "split": split, "source_id": str(raw.get("id")),
               "source_file": file, "annotation_source": "human", "original_prompt_label": raw["prompt_label"],
               "violated_categories": raw.get("violated_categories")}
        admitted.setdefault(group, row)
    clean = [r for g, r in admitted.items() if states[g] == {(r["split"], r["label"])}]
    return clean, {"scanned_rows": scanned, "counts": dict(counts), "normalized_groups": len(states), "clean_groups": len(clean),
                   "cross_split_or_annotation_conflict": len(admitted) - len(clean), "old_exposed_audit_groups": len(excluded)}


def select(rows, sizes, salts, eligible):
    selected, rejected = {}, Counter()
    for split, n in sizes.items():
        official = "test" if split == "audit" else split
        result = []
        for label in ("Safe", "Flagged"):
            ordered = sorted((r for r in rows if r["split"] == official and r["label"] == label), key=lambda r: sha((salts[split] + ":" + r["group_id"]).encode()))
            chosen = []
            for row in ordered:
                if not eligible(row):
                    rejected[split] += 1
                    continue
                chosen.append(row)
                if len(chosen) == n // 2:
                    break
            if len(chosen) != n // 2:
                raise ValueError("insufficient clean independent human sources: " + split + " " + label)
            result.extend(chosen)
        selected[split] = sorted(result, key=lambda r: r["group_id"])
    seen = set()
    for rows in selected.values():
        ids = {r["group_id"] for r in rows}
        if len(ids) != len(rows) or seen & ids:
            raise ValueError("source/entity split overlap")
        seen |= ids
    return selected, dict(rejected)


def single(row, style):
    label = row["label"]
    opposite = {"Safe": "Flagged", "Flagged": "Safe"}[label]
    def answer(verdict):
        if style == 0:
            return json.dumps({"verdict": verdict, "flagged": verdict == "Flagged"}, separators=(",", ":"))
        return "Verdict: " + verdict + "." if style == 1 else verdict
    return {"prompt": json.dumps({"text": row["text"]}, ensure_ascii=False, separators=(",", ":")),
            "chosen": answer(label), "rejected": answer(opposite), "label": label, "margin": 0.0, "label_kind": "preference_pair",
            "group_id": row["group_id"], "source_group_ids": [row["group_id"]], "annotation_source": "human",
            "preference_authorship": "programmatically constructed from original human source labels", "style": style, "source_id": row["source_id"]}


def contrast(first, second, style, selected):
    if first["label"] == second["label"] or first["group_id"] == second["group_id"]:
        raise ValueError("two independent opposite human-labelled passages required")
    target = (first, second)[selected]
    def answer(verdict):
        references = "Compared passages: " + first["text"][:120] + " / " + second["text"][:120] + ". Verdict applies only to the requested selection."
        if style == 0:
            return json.dumps({"verdict": verdict, "flagged": verdict == "Flagged", "reason": references}, ensure_ascii=False, separators=(",", ":"))
        return ("Verdict: " + verdict + ". " if style == 1 else verdict + ". ") + references
    prompt = json.dumps({"task": "Moderate only the selected passage under the content safety policy. Safe means no violation; Flagged means a violation.",
                         "selected": "first" if selected == 0 else "second", "first": first["text"], "second": second["text"]}, ensure_ascii=False, separators=(",", ":"))
    label = target["label"]
    return {"prompt": prompt, "chosen": answer(label), "rejected": answer({"Safe": "Flagged", "Flagged": "Safe"}[label]),
            "label": label, "margin": 0.0, "label_kind": "preference_pair", "style": style,
            "group_id": sha((first["group_id"] + second["group_id"] + str(selected)).encode()),
            "source_group_ids": [first["group_id"], second["group_id"]], "annotation_source": "human",
            "preference_authorship": "AI-assisted programmatic prompt-selection contrast, not human-reviewed preferences"}


def training_pairs(rows, epoch=0):
    by_label = {label: sorted((r for r in rows if r["label"] == label), key=lambda r: r["group_id"]) for label in ("Safe", "Flagged")}
    if len(rows) != 1584 or any(len(values) != 792 for values in by_label.values()):
        raise ValueError("frozen 1584 independent training sources required")
    pairs = []
    for i in range(768):
        first, second = by_label["Safe"][i], by_label["Flagged"][i]
        if (i // 3) % 2:
            first, second = second, first
        style = (i + epoch) % 3
        pairs.extend(contrast(first, second, style, selected) for selected in (0, 1))
    for label in by_label:
        pairs.extend(single(row, i // 8) for i, row in enumerate(by_label[label][768:]))
    return pairs


def validation_views(rows, context=False):
    if not context:
        return {name: [single(row, style) for row in rows] for style, name in enumerate(FORMATS)}
    safe = sorted((r for r in rows if r["label"] == "Safe"), key=lambda r: r["group_id"])
    flagged = sorted((r for r in rows if r["label"] == "Flagged"), key=lambda r: r["group_id"])
    if len(safe) != len(flagged):
        raise ValueError("balanced prompt-selection validation required")
    return {name: [contrast(*( (a, b) if i % 2 == 0 else (b, a)), style, selected)
                   for i, (a, b) in enumerate(zip(safe, flagged, strict=True)) for selected in (0, 1)]
            for style, name in enumerate(FORMATS)}
