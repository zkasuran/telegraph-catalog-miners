# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure data and wire helpers. No model imports or model execution."""
import hashlib
import json
import math
import os
import re
import unicodedata
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = "AI_TEXT_DETECTION"
COUNTS = {"train": 128, "validation": 32, "test": 64}
MAX_TEXT_BYTES = 256
MAX_TOKENS = 256
WORDS = re.compile(r"[a-z0-9']+")
FORMATS = ("json", "prose", "bare_label")
SHORTCUTS = {"length", "overlap", "answer_only", "source_origin_length"}
SCOPE = ("Published scientific human controls versus GPT-2-family synthetic continuations. "
         "DistilGPT2 is a held-out checkpoint distilled from GPT-2, not an independent generator family. "
         "No modern-model detection, Mixed/likely calibration or proof of a person's authorship.")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def identity(text):
    return sha(normalized(text).encode())


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def lines(raw):
    if len(raw) > 16 << 20:
        raise ValueError("JSONL byte ceiling")
    return [json.loads(s) for s in raw.decode("utf-8").split("\n") if s.strip()]


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or
            os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners" or
            not os.environ.get("GITHUB_RUN_ID") or not os.environ.get("GITHUB_SHA")):
        raise RuntimeError("Model operations require the bounded controller-owned public CPU workflow")


def fetch(url, path, expected=None, ceiling=16 << 20):
    """Bounded anonymous public download. Model callers must guard first."""
    if expected and path.exists() and path.stat().st_size <= ceiling:
        if sha(path.read_bytes()) == expected:
            return path.read_bytes()
    result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                             "--proto", "=https", "--proto-redir", "=https",
                             "--connect-timeout", "30", "--max-time", "120",
                             "--max-filesize", str(ceiling), "--retry", "2", "--retry-max-time", "300", url],
                            capture_output=True, timeout=330, check=False)
    if result.returncode:
        raise OSError("public download failed: " + result.stderr.decode(errors="replace")[:200])
    raw = result.stdout
    if len(raw) > ceiling:
        raise ValueError("download byte ceiling")
    if expected and sha(raw) != expected:
        raise ValueError("source hash mismatch: " + path.name)
    part = path.with_suffix(path.suffix + ".partial")
    part.write_bytes(raw)
    part.replace(path)
    return raw


def render(prompt, answer):
    # The pinned reference trainer's literal expression. Tested against its AST.
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def audit_excerpts(human, decoded):
    """Symmetric whole-word byte fit. No score, label or retry chooses a view.

    Source human controls already have collapsed whitespace. Apply that same
    transformation to generated text then keep the largest shared word count
    from 16 through 24 fitting both literal 256-byte request slots.
    """
    human_words, generated_words = human.split(), decoded.split()
    if len(human_words) < 24 or len(generated_words) < 24:
        raise ValueError("fixed generation/source has fewer than 24 words")
    for count in range(24, 15, -1):
        h, g = " ".join(human_words[:count]), " ".join(generated_words[:count])
        if 32 <= len(h.encode()) <= MAX_TEXT_BYTES and 32 <= len(g.encode()) <= MAX_TEXT_BYTES:
            if identity(h) == identity(g):
                raise ValueError("identical production histories cannot supply an origin preference")
            return h, g, {"word_count": count, "human_bytes": len(h.encode()), "synthetic_bytes": len(g.encode()),
                          "changes": "Whitespace collapsed on both origins; equal-count whole-word prefixes selected for the literal request ceiling."}
    raise ValueError("both origins cannot fit the unchanged request ceiling at 16 words")


def validation_gate(formats, baselines):
    if set(formats) != set(FORMATS) or set(baselines) != set(FORMATS):
        return {"passed": False, "failures": ["missing validation format"]}
    if any(not SHORTCUTS <= set(v) or any(not math.isfinite(x) or not 0 <= x <= 1 for x in v.values()) for v in baselines.values()):
        return {"passed": False, "failures": ["missing or invalid frozen shortcut, including source-origin length"]}
    failures = []
    for style, metrics in formats.items():
        labels = metrics.get("accuracy_by_label", {})
        if (metrics["n_eval"] != 2 * COUNTS["validation"] or metrics["pairwise_accuracy"] < 0.80 or
                set(labels) != {"Human", "AI-generated"} or min(labels.values()) < 0.70 or
                metrics["pairwise_accuracy"] < max(baselines[style].values()) + 0.10):
            failures.append(style + " failed unchanged validation quality/label/shortcut gate")
    return {"passed": not failures, "failures": failures}


def pair_data_shortcuts_ok(baselines):
    """The original 65% answer-surface data gate, distinct from quality margins."""
    names = {"length", "overlap", "answer_only"}
    return bool(baselines) and all(names <= set(v) and all(v[k] <= 0.65 for k in names) for v in baselines.values())


def validation_selection_key(formats, baselines):
    """Train-independent selection on validation only, with no metric overwrite."""
    gate = validation_gate(formats, baselines)
    if set(formats) != set(FORMATS):
        raise ValueError("selection requires every claimed validation format")
    if set(baselines) != set(FORMATS) or any(not SHORTCUTS <= set(v) for v in baselines.values()):
        raise ValueError("selection requires every frozen shortcut")
    for metrics in formats.values():
        labels = metrics.get("accuracy_by_label", {})
        if (metrics.get("n_eval") != 2 * COUNTS["validation"] or set(labels) != {"Human", "AI-generated"} or
                any(not math.isfinite(v) or not 0 <= v <= 1 for v in [metrics["pairwise_accuracy"], *labels.values()]) or
                not math.isfinite(metrics["val_loss"]) or metrics["val_loss"] < 0):
            raise ValueError("selection requires complete finite actual validation metrics")
    margin = min(min(m["pairwise_accuracy"] - 0.80,
                     min(m["accuracy_by_label"].values()) - 0.70,
                     m["pairwise_accuracy"] - max(baselines[s].values()) - 0.10)
                 for s, m in formats.items())
    loss = sum(m["val_loss"] for m in formats.values()) / len(formats)
    return (int(gate["passed"]), margin, -loss)


def pair(row, style="json"):
    if style not in FORMATS or row["origin"] not in ("human_control", "synthetic"):
        raise ValueError("unsupported origin or answer format")
    text = row["text"]
    if not isinstance(text, str) or not 32 <= len(text.encode()) <= MAX_TEXT_BYTES:
        raise ValueError("text outside live 256-byte request ceiling")
    good = "Human" if row["origin"] == "human_control" else "AI-generated"
    bad = "AI-generated" if good == "Human" else "Human"
    def answer(label):
        if style == "bare_label":
            return label
        if style == "prose":
            return f"Verdict: {label}. Text assessed: {text}"
        return json.dumps({"verdict": label, "cues": ["Text assessed: " + text]},
                          ensure_ascii=False, separators=(",", ":"))
    return {"prompt": json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")),
            "chosen": answer(good), "rejected": answer(bad), "margin": 0.0,
            "label_kind": "preference_pair", "label": good, "origin": row["origin"],
            "source_id": row["source_id"], "group_id": row["group_id"],
            "text_id": identity(text), "style": style, "split": row["split"],
            "generator": row.get("generator", "publisher"), "year": row["year"],
            "view_id": row.get("view_id", "original:" + row["group_id"])}


def check_splits(splits):
    if set(splits) != set(COUNTS):
        raise ValueError("missing split")
    seen_groups, seen_text = set(), set()
    for split, rows in splits.items():
        if ((split != "train" and len(rows) != 2 * COUNTS[split]) or
                (split == "train" and not 2 * COUNTS[split] <= len(rows) <= 6 * COUNTS[split])):
            raise ValueError("wrong split size: " + split)
        groups = {}
        ids = set()
        for r in rows:
            text = json.loads(r["prompt"])["text"]
            if r["split"] != split or r["text_id"] != identity(text):
                raise ValueError("split or normalized identity mismatch")
            if r["group_id"] != identity(r["source_id"]):
                raise ValueError("article-group identity mismatch")
            if not ((split == "train" and 2012 <= r["year"] <= 2014) or
                    (split == "validation" and r["year"] == 2015) or
                    (split == "test" and 2016 <= r["year"] <= 2018)):
                raise ValueError("publication time-block violation")
            if r["chosen"] == r["rejected"] or r["text_id"] in ids:
                raise ValueError("identical or duplicate source text")
            ids.add(r["text_id"])
            groups.setdefault(r["group_id"], []).append(r)
            if r["label"] != ("Human" if r["origin"] == "human_control" else "AI-generated"):
                raise ValueError("origin label mismatch")
            expected_generator = "distilgpt2" if split == "test" else "gpt2"
            if r["origin"] == "synthetic" and r["generator"] != expected_generator:
                raise ValueError("held-out generator violation")
        if len(groups) != COUNTS[split]:
            raise ValueError("independent article group count changed")
        for rs in groups.values():
            views = {}
            for row in rs:
                views.setdefault(row["view_id"], []).append(row)
            if any(len(v) != 2 or {r["origin"] for r in v} != {"human_control", "synthetic"} for v in views.values()):
                raise ValueError("article/view derivatives not paired as one group")
        if seen_groups.intersection(groups) or seen_text.intersection(ids):
            raise ValueError("cross-split article or normalized text overlap")
        seen_groups.update(groups)
        seen_text.update(ids)


def fit_baselines(train):
    """Integer train-only answer-surface counts and frozen rule directions.

    No fresh floating-point fitting is needed to reproduce the baseline bytes.
    The answer-only log likelihood is computed from immutable integer counts.
    """
    counts = {s: {} for s in ("chosen", "rejected")}
    for r in train:
        for side in counts:
            for w in WORDS.findall(r[side].casefold()):
                counts[side][w] = counts[side].get(w, 0) + 1
    directions = {}
    for name in ("length", "overlap"):
        diffs = [rule_diff(r, name) for r in train]
        directions[name] = 1 if sum((d > 0) + 0.5 * (d == 0) for d in diffs) >= len(diffs)/2 else -1
    return {"schema_version": 1, "fit_split": "train", "integer_word_counts": counts,
            "direction": directions, "smoothing": 1, "source_origin_length": source_length_fit(train)}


def source_length_fit(train):
    from origin_contract import raw_origin_view, fit_length
    return fit_length([raw_origin_view(r) for r in train])


def rule_diff(row, name):
    if name == "length":
        return len(row["chosen"].encode()) - len(row["rejected"].encode())
    prompt = set(WORDS.findall(row["prompt"].casefold()))
    def overlap(s):
        answer = set(WORDS.findall(s.casefold()))
        return len(prompt.intersection(answer)) / max(1, len(prompt.union(answer)))
    return overlap(row["chosen"]) - overlap(row["rejected"])


def baseline_report(rows, fitted):
    counts = fitted["integer_word_counts"]
    vocab = set(counts["chosen"]).union(counts["rejected"])
    totals = {s: sum(c.values()) + len(vocab) for s, c in counts.items()}
    def surface(text):
        return sum(math.log((counts["chosen"].get(w, 0) + 1) / totals["chosen"]) -
                   math.log((counts["rejected"].get(w, 0) + 1) / totals["rejected"])
                   for w in WORDS.findall(text.casefold()) if w in vocab)
    diffs = {name: [rule_diff(r, name) * direction for r in rows]
             for name, direction in fitted["direction"].items()}
    diffs["answer_only"] = [surface(r["chosen"]) - surface(r["rejected"]) for r in rows]
    report = {name: sum(0.5 if abs(d) < 1e-12 else float(d > 0) for d in ds) / len(ds)
              for name, ds in diffs.items()}
    from origin_contract import raw_origin_view, length_accuracy
    report["source_origin_length"] = length_accuracy([raw_origin_view(r) for r in rows], fitted["source_origin_length"])
    if "original_source_origin_length" in fitted:
        report["original_source_origin_length"] = length_accuracy([raw_origin_view(r) for r in rows], fitted["original_source_origin_length"])
    return report


def quality(metrics, untrained, baseline):
    failures = []
    if set(metrics) != set(FORMATS) or set(untrained) != set(FORMATS) or set(baseline) != set(FORMATS):
        return {"verdict": "FAIL", "failures": ["missing literal response format"]}
    if any(not SHORTCUTS <= set(v) or any(not math.isfinite(x) or not 0 <= x <= 1 for x in v.values()) for v in baseline.values()):
        return {"verdict": "FAIL", "failures": ["missing or invalid frozen shortcut including source-origin length"]}
    for style in FORMATS:
        m = metrics[style]
        if m["n_eval"] != 2 * COUNTS["test"] or m["pairwise_accuracy"] < 0.80:
            failures.append(style + " below 80% or wrong audit count")
        if set(m["accuracy_by_label"]) != {"Human", "AI-generated"} or min(m["accuracy_by_label"].values()) < 0.70:
            failures.append(style + " below per-origin gate")
        if m["pairwise_accuracy"] < untrained[style]["pairwise_accuracy"] + 0.05:
            failures.append(style + " did not improve seed-matched untrained model")
        if m["pairwise_accuracy"] < max(baseline[style].values()) + 0.10:
            failures.append(style + " did not beat strongest frozen shortcut")
    return {"verdict": "FAIL" if failures else "PASS", "failures": failures}
