# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure literal-wire and finite-score contracts. No neural imports."""
import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = "CONTRACT_OBLIGATION_AUDIT"
FORMATS = ("json", "prose")
LABELS = ("Entailment", "Contradiction")
REPO = "zkasuran/telegraph-catalog-miners"
FOUNDATION = {
    "foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
    "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
    "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    def invalid(_):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != REPO
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or os.environ.get("OWNER_SOURCE_ONLY") == "1"):
        raise RuntimeError("neural/runtime operations require the bounded public GitHub workflow")


def decode_bits(bits):
    if type(bits) is not int or not 0 <= bits < 1 << 32:
        raise ValueError("score must be a uint32 bit pattern")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite reference score")
    return value


def check_wire(response, count, package_keccak):
    if (not isinstance(response, dict) or response.get("foundation_hash") != FOUNDATION["foundation_lora_fp32.onnx"]
            or response.get("tokenizer_hash") != FOUNDATION["tokenizer.json"]
            or response.get("adapter_hash") != package_keccak):
        raise ValueError("literal bare-hash reference identity mismatch")
    results = response.get("results")
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("reference result count mismatch")
    bits = [item["score"] for item in results]
    for value in bits:
        decode_bits(value)
    return bits


def quality(formats):
    if set(formats) != set(FORMATS):
        raise ValueError("a claimed response format was not audited")
    failures = []
    for name, item in formats.items():
        if set(item["per_label"]) != set(LABELS):
            raise ValueError("missing source label in quality report")
        values = [item["accuracy"], item["untrained_accuracy"], item["shortcut_accuracy"], *item["per_label"].values()]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("quality metrics must be finite actual fractions")
        if (item["accuracy"] < .80 or min(item["per_label"].values()) < .70
                or item["accuracy"] + 1e-9 < item["untrained_accuracy"] + .05
                or item["accuracy"] + 1e-9 < item["shortcut_accuracy"] + .10):
            failures.append(name)
    return {"passed": not failures, "failed_formats": failures}


def read_jsonl(path):
    return [strict_json(line) for line in path.read_bytes().split(b"\n") if line.strip()]


def check_config(value):
    fields = {"SPDX-License-Identifier", "schema_version", "family", "source_revision", "archive_sha256", "archive_url", "limits", "max_evidence_bytes", "max_tokens", "epochs", "seed"}
    if set(value) != fields or value["schema_version"] != 1 or value["family"] != FAMILY:
        raise ValueError("pipeline config identity/schema mismatch")
    if not re.fullmatch(r"[a-f0-9]{40}", value["source_revision"]) or not re.fullmatch(r"[a-f0-9]{64}", value["archive_sha256"]):
        raise ValueError("source identity must be pinned")
    if value["limits"] != {"train": 64, "validation": 16, "audit": 64}:
        raise ValueError("frozen split ceilings changed")
    if value["max_evidence_bytes"] != 768 or value["max_tokens"] != 512 or value["epochs"] != 32 or value["seed"] != 42:
        raise ValueError("frozen pipeline bounds changed")
    return value
