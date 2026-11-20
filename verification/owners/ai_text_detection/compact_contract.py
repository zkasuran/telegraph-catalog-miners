# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Training-only short source quotations, no invented cue or source identifier padding."""
import json


def compact_training(rows):
    result = []
    for row in rows:
        if row["split"] != "train":
            raise ValueError("compact source-quote layout may fit training only")
        if row["style"] == "bare_label":
            result.append(dict(row))
            continue
        text = json.loads(row["prompt"])["text"]
        words = text.split()
        if len(words) < 8:
            raise ValueError("source too short for fixed quote anchors")
        opening, ending = " ".join(words[:4]), " ".join(words[-4:])
        cues = ["Opening words: " + opening, "Ending words: " + ending]
        good = row["label"]
        bad = "AI-generated" if good == "Human" else "Human"
        def answer(label):
            if row["style"] == "json":
                return json.dumps({"verdict": label, "cues": cues}, ensure_ascii=False, separators=(",", ":"))
            if row["style"] == "prose":
                return f"Verdict: {label}. Opening words: {opening}. Ending words: {ending}."
            raise ValueError("unsupported compact answer format")
        result.append({**row, "chosen": answer(good), "rejected": answer(bad), "layout": "compact_source_quotes",
                       "quote_anchor_words": 8})
    return result
