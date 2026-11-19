# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only contracts shared by the public gate and remote learner."""
import hashlib
import json
import math
import os
import re
from pathlib import Path

FAMILY = "THREAT_IP_REPUTATION"
REPO = "zkasuran/telegraph-catalog-miners"
TRAIN_SHA = "8546f4a520965eba5a5feda407aaedadd8eda5b7670e5d6bfa21b38be1e5b6bd"
VALIDATION_SHA = "1606218fb15deb0f8b6ffe70bf5360f1530fa51cd35c8cda7b2781a1cdfe4819"
V2_TRAIN_SHA = "8a074d8b374ac02117a84d6708fcf739b130831a116fe40186717a98e725143d"
V2_VALIDATION_SHA = "b537d108aa76a3e06f46822419cfb319ceb0fd49c547cb8e72c3d46473a2e686"
KIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
KIT_SHA = "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243"
BASE_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
FOUNDATION = {
    "foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
    "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
    "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456",
}
BASE_FILES = {
    "model.safetensors": "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
    "config.json": "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
    "tokenizer_config.json": "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
    "special_tokens_map.json": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049",
}
FORMATS = {"json", "prose"}
LABELS = {"publisher_observed_in_window", "historical_publisher_observation"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != REPO or
            not re.fullmatch(r"[1-9][0-9]*", os.environ.get("GITHUB_RUN_ID", "")) or
            not re.fullmatch(r"[a-f0-9]{40}", os.environ.get("GITHUB_SHA", ""))):
        raise RuntimeError("model work runs only in the bounded owned public CPU workflow")


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def norm(text):
    return " ".join(text.casefold().split())


def words(text):
    return re.findall(r"[a-z0-9']+", text.casefold())


def read_development(root, variant="original"):
    result = {}
    specs = (("train", TRAIN_SHA, 256, 64), ("validation", VALIDATION_SHA, 64, 16))
    if variant == "compact-elapsed-v2":
        specs = (("train", V2_TRAIN_SHA, 512, 64), ("validation", V2_VALIDATION_SHA, 128, 16))
    elif variant != "original":
        raise ValueError("unknown frozen development variant")
    for name, digest, count, groups in specs:
        raw = (root / (name + ".jsonl")).read_bytes()
        if len(raw) > 2 << 20 or sha(raw) != digest:
            raise ValueError("frozen development identity changed")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != count or len({row["group_id"] for row in rows}) != groups:
            raise ValueError("development group or pair count changed")
        if any(row["format"] not in FORMATS or row["label"] not in LABELS or row["observation_modified"] is not False for row in rows):
            raise ValueError("source preference lineage changed")
        result[name] = [{**row, "margin": 0.0} for row in rows]
    groups = [{row["group_id"] for row in result[name]} for name in ("train", "validation")]
    text = [{norm(render(row["prompt"], row[side])) for row in result[name] for side in ("chosen", "rejected")}
            for name in ("train", "validation")]
    if groups[0] & groups[1] or text[0] & text[1]:
        raise ValueError("normalized source group or rendered overlap")
    return result


def basic_margins(row):
    chosen, rejected = row["chosen"], row["rejected"]
    q = set(words(row["prompt"]))
    return {"length": len(chosen) - len(rejected),
            "overlap": len(q & set(words(chosen))) - len(q & set(words(rejected))),
            "prefer_current": int("publisher_observed_in_window" in chosen) - int("publisher_observed_in_window" in rejected)}


def margin_accuracy(values):
    return sum(1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in values) / len(values)


def validation_gate(trained, untrained, shortcuts):
    failures = []
    if set(trained) != FORMATS or set(untrained) != FORMATS or set(shortcuts) != FORMATS:
        return {"passed": False, "failures": ["every claimed response format is required"]}
    for fmt in FORMATS:
        metrics = trained[fmt]
        if set(metrics["per_label"]) != LABELS:
            failures.append(fmt + " lacks a claimed label")
            continue
        values = [metrics["accuracy"], untrained[fmt]["accuracy"], shortcuts[fmt], *metrics["per_label"].values()]
        if not all(type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1 for value in values):
            failures.append(fmt + " invalid metrics")
        elif (metrics["accuracy"] < 0.8 or min(metrics["per_label"].values()) < 0.7 or
              metrics["accuracy"] + 1e-9 < untrained[fmt]["accuracy"] + 0.05 or
              metrics["accuracy"] + 1e-9 < shortcuts[fmt] + 0.10):
            failures.append(fmt + " fails unchanged accuracy/label/improvement gates")
    return {"passed": not failures, "failures": failures}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
