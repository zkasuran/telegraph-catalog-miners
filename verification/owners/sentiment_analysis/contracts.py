# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure contracts and bounded public downloads. Never import a neural framework."""
import hashlib
import json
import math
import os
import re
import struct
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = "SENTIMENT_ANALYSIS"
CHILD = FAMILY + "_GENERAL"
OWNER = "0x8b224783fe5b3c52b7db0cb9b1754f8812b75287"
REPO = "zkasuran/telegraph-catalog-miners"
NODE = "https://devnode.telegraphprotocol.com"
FORMATS = ("json", "prose", "bare_label")
LABELS = {"Positive", "Negative"}
FOUNDATION = {
    "foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
    "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
    "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456",
}
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"
SPEC = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
        "tokenizer_hash": FOUNDATION["tokenizer.json"], "foundation_version": "modernbert-base-lora16-v1",
        "r_max": 16, "num_layers": 22, "hidden_size": 768,
        "targets": {"attn.Wqkv": [768, 2304], "attn.Wo": [768, 768],
                    "mlp.Wi": [768, 2304], "mlp.Wo": [1152, 768]}}
MAX_BODY = 1 << 20
MAX_PACKAGE = 64 << 20
HASH = re.compile(r"[0-9a-f]{64}")
FROZEN = json.loads((HERE / "frozen.json").read_text())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(MAX_BODY), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json(raw):
    def fields(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value

    def constant(value):
        raise ValueError("nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=fields, parse_constant=constant)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != REPO or
            os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or os.environ.get("OWNER_SOURCE_ONLY") == "1"):
        raise RuntimeError("model/runtime operations require the bounded public workflow")


def bare_hash(value):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError("literal bare lowercase hash required")
    return value


def immutable_url(value):
    prefix = "https://raw.githubusercontent.com/" + REPO + "/"
    if not isinstance(value, str) or not re.fullmatch(re.escape(prefix) + r"[0-9a-f]{40}/[A-Za-z0-9_./-]+", value):
        raise ValueError("owned immutable public URL required")
    if any(part in (".", "..", "") for part in value[len(prefix):].split("/")):
        raise ValueError("immutable path escape")
    return value


def fetch(url, path, expected, ceiling=MAX_PACKAGE):
    bare_hash(expected)
    if path.is_file() and path.stat().st_size <= ceiling and file_sha(path) == expected:
        return
    allowed = url in {FROZEN["source_url"], FROZEN["licences_url"]} or url.startswith(RELEASE) or url == (
        "https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + FROZEN["kit_commit"])
    if not allowed:
        immutable_url(url)
    part = path.with_name(path.name + ".partial")
    start, total, h = time.monotonic(), 0, hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "Sentiment-owner-verification/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as source, part.open("wb") as out:
            if source.status != 200 or not source.geturl().startswith("https://"):
                raise ValueError("public download status or transport mismatch")
            while chunk := source.read(MAX_BODY):
                total += len(chunk)
                if total > ceiling or time.monotonic() - start > 600:
                    raise ValueError("download byte or time ceiling")
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != expected:
            raise ValueError("public download hash mismatch")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def package_parts(raw):
    if not 8 < len(raw) <= MAX_PACKAGE:
        raise ValueError("bounded safetensors package required")
    length = struct.unpack("<Q", raw[:8])[0]
    if not 0 < length <= MAX_BODY or 8 + length >= len(raw):
        raise ValueError("invalid package header length")
    header = strict_json(raw[8:8 + length])
    if not isinstance(header, dict) or not isinstance(header.get("__metadata__"), dict):
        # All malformed binary containers share the ValueError parser boundary.
        raise ValueError("package metadata missing")  # noqa: TRY004
    return header, raw[8 + length:]


def recover_package(raw, recovery_id):
    if sha(raw) != FROZEN["origin_sha256"] or recovery_id != FROZEN["recovery_id"]:
        raise ValueError("frozen original or recovery identity mismatch")
    header, payload = package_parts(raw)
    metadata = header["__metadata__"]
    if metadata.get("intent") != FAMILY or metadata.get("class") != "judgment":
        raise ValueError("original package family/class mismatch")
    for field in ("x-licence", "x-upstream-notice", "x-upstream-apache-2.0", "x-upstream-mit"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise ValueError("original embedded grants missing")
    if "LicenseRef-zkasuran-SAND-1.0" not in metadata["x-licence"]:
        raise ValueError("original SAND terms missing")
    if "x-registration-recovery" in metadata:
        raise ValueError("original already has recovery metadata")
    updated = {**header, "__metadata__": {**metadata, "x-registration-recovery": json.dumps({
        "origin_sha256": FROZEN["origin_sha256"], "recovery_id": recovery_id,
        "rejected_registration_id": 118, "wire_fix_commit": FROZEN["wire_fix_commit"],
        "change": "registration identity only; every tensor and original model field preserved",
        "SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0",
    }, sort_keys=True, separators=(",", ":"))}}
    encoded = json.dumps(updated, ensure_ascii=False, separators=(",", ":")).encode()
    recovered = struct.pack("<Q", len(encoded)) + encoded + payload
    verify_recovery(raw, recovered)
    return recovered


def verify_recovery(original, recovered):
    before, old_data = package_parts(original)
    after, new_data = package_parts(recovered)
    old, new = before["__metadata__"], after["__metadata__"]
    if old_data != new_data or {k: v for k, v in before.items() if k != "__metadata__"} != {
            k: v for k, v in after.items() if k != "__metadata__"}:
        raise ValueError("recovery changed tensor descriptors or payload bytes")
    if any(new.get(key) != value for key, value in old.items()) or set(new) - set(old) != {"x-registration-recovery"}:
        raise ValueError("recovery changed original model metadata or grants")
    record = strict_json(new["x-registration-recovery"])
    if record.get("origin_sha256") != FROZEN["origin_sha256"] or record.get("recovery_id") != FROZEN["recovery_id"]:
        raise ValueError("recovery lineage mismatch")
    return sha(old_data)


def finite_bits(value):
    if type(value) is not int or not 0 <= value < 1 << 32:
        raise ValueError("invalid FP32 score bits")
    decoded = struct.unpack("<f", struct.pack("<I", value))[0]
    if not math.isfinite(decoded):
        raise ValueError("nonfinite FP32 score")
    return decoded


def validate_ready(report):
    if not isinstance(report, dict):
        raise ValueError("ready report must be an object")  # noqa: TRY004
    if (report.get("schema_version") != 1 or report.get("status") != "REGISTRATION_READY" or
            report.get("family") != FAMILY or type(report.get("run_id")) is not int or report["run_id"] < 1 or
            not re.fullmatch(r"[0-9a-f]{40}", report.get("commit", "")) or report.get("serving_commit") != report["commit"]):
        raise ValueError("ready report identity mismatch")
    if report.get("owner", OWNER) != OWNER:
        raise ValueError("selected Telegraph owner mismatch")
    bare_hash(report.get("package_sha256"))
    bare_hash(report.get("package_keccak"))
    for key, value in (("foundation_sha256", FOUNDATION["foundation_lora_fp32.onnx"]),
                       ("engine_sha256", FOUNDATION["foundation_engine.wasm"]),
                       ("tokenizer_sha256", FOUNDATION["tokenizer.json"])):
        if report.get(key) != value:
            raise ValueError("ready runtime asset identity mismatch")
    source, quality, runtime = report.get("source", {}), report.get("quality", {}), report.get("runtime", {})
    if (source.get("normalized_overlap") != 0 or source.get("audit_used_for_selection") is not False or
            source.get("grants_verified") is not True or source.get("independent_groups", 0) < 32):
        raise ValueError("source audit boundary failed")
    bare_hash(source.get("source_manifest_sha256"))
    if (quality.get("passed") is not True or quality.get("audit_pairs", 0) < 64 or
            quality.get("all_supported_formats_tested") is not True or set(quality.get("formats", {})) != set(FORMATS)):
        raise ValueError("missing or failed audit formats")
    for metrics in quality["formats"].values():
        labels = metrics.get("per_label", {})
        values = [metrics.get("accuracy"), metrics.get("untrained_accuracy"), metrics.get("shortcut_accuracy"), *labels.values()]
        if set(labels) != LABELS or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("invalid actual audit metrics")
        if values[0] < .80 or min(labels.values()) < .70 or values[0] < values[1] + .05 or values[0] < values[2] + .10:
            raise ValueError("audit quality or shortcut gate failed")
    if (runtime.get("full_parser_passed") is not True or runtime.get("reference_engine_passed") is not True or
            runtime.get("repeat_bits_identical") is not True or runtime.get("negative_identity_controls", 0) < 3):
        raise ValueError("parser/reference-engine/identity gate failed")
    vectors = runtime.get("vectors", [])
    if not isinstance(vectors, list) or not 8 <= len(vectors) <= 100:
        raise ValueError("bounded actual reference vectors required")
    for vector in vectors:
        ids = vector.get("tokens")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 512 or any(type(i) is not int or not 0 <= i < 50368 for i in ids):
            raise ValueError("invalid fixed token vector")
        finite_bits(vector.get("score_bits"))
    return report
