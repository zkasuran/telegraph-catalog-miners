# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Local source/identity contracts. Never imports a neural library."""
import ast
import hashlib
import json
import math
import os
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = "LANGUAGE_TRANSLATION"
PACKAGE_SHA = "e02ce556730041da5f24196e74e37afd8a6df9cfab4e17a6045ed2672671cb42"
PACKAGE_KECCAK = "cc90629facd61d278ab745cf3b5778d78206a509d57fb39dc4e16adc68752410"
SOURCE_SHA = "099fa6f2c0f4eda56dd995ed5265e820b8c885bae58667e0e72f12db1f7e6f7b"
TRAIN_RUN = 44463705810
TRAIN_COMMIT = "312353893799fe90b52111ae37a079809f56fbc4"
KIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
KIT_SHA = "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243"
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"
FOUNDATION = {"foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
              "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
              "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_asset(name):
    path = HERE / "assets" / name
    expected = json.loads((HERE / "asset-pins.json").read_text())[name]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("immutable asset changed: " + name)
    return raw


def verify_assets():
    for name in json.loads((HERE / "asset-pins.json").read_text()):
        read_asset(name)
    if sha(HERE / "assets/candidate.safetensors") != PACKAGE_SHA:
        raise ValueError("candidate identity changed")
    source = json.loads(read_asset("source-manifest.json"))
    native = json.loads(read_asset("native-report.json"))
    if sha(HERE / "assets/source-manifest.json") != SOURCE_SHA or native["source_manifest_sha256"] != SOURCE_SHA:
        raise ValueError("frozen data identity changed")
    if int(native["run_id"]) != TRAIN_RUN or native["commit"] != TRAIN_COMMIT or not native["passed"]:
        raise ValueError("original native run binding failed")
    components, texts, identities = set(), set(), set()
    import unicodedata
    for split, count in (("train", 128), ("validation", 32), ("test", 64)):
        blocks = source["splits"][split]
        roots = [root for block in blocks for root in block["component_roots"]]
        ids = {sid for block in blocks for link in block["links"] for sid in link}
        normalized = {" ".join(unicodedata.normalize("NFKC", source["originals"][str(sid)]["text"]).casefold().split()) for sid in ids}
        if len(blocks) != count or len(roots) != len(set(roots)) or components.intersection(roots) or identities.intersection(ids) or texts.intersection(normalized):
            raise ValueError("group/identity/normalized overlap failed")
        components.update(roots)
        identities.update(ids)
        texts.update(normalized)
    if components.intersection(source["excluded_failed_components"]) or not source["source_review_before_split"]:
        raise ValueError("exposed failed sources or unreviewed freeze")
    for original in source["originals"].values():
        if original["license"] != "CC-BY-2.0-FR" or not original["credited_contributor"] or original["original_text_changed"]:
            raise ValueError("original grant/attribution failed")
    return source, native


def rows(style):
    source = json.loads(read_asset("source-manifest.json"))
    originals = source["originals"]
    result = []
    for block in source["splits"]["test"]:
        (ea, fa), (eb, fb) = block["links"]
        for src, good, bad, direction in ((ea, fa, fb, "eng-fra"), (eb, fb, fa, "eng-fra"), (fa, ea, eb, "fra-eng"), (fb, eb, ea, "fra-eng")):
            def answer(sid):
                text = originals[str(sid)]["text"]
                return json.dumps({"translation": text}, ensure_ascii=False, separators=(",", ":")) if style == "json" else "Translation: " + text if style == "prose" else text
            result.append({"prompt": json.dumps({"text": originals[str(src)]["text"], "target_language": "French" if direction == "eng-fra" else "English"}, ensure_ascii=False, separators=(",", ":")),
                           "chosen": answer(good), "rejected": answer(bad), "direction": direction, "group_id": block["id"]})
    return result


def metrics(items, bits):
    if len(items) != 256 or len(bits) != 512 or len({row["group_id"] for row in items}) != 64:
        raise ValueError("complete frozen grouped audit required")
    values = []
    for bit in bits:
        if type(bit) is not int or not 0 <= bit < 1 << 32:
            raise ValueError("invalid score bits")
        value = struct.unpack("<f", struct.pack("<I", bit))[0]
        if not math.isfinite(value):
            raise ValueError("nonfinite score bits")
        values.append(value)
    wins = [values[2*i] > values[2*i+1] for i in range(256)]
    labels = {label: sum(w for row, w in zip(items, wins) if row["direction"] == label) / 128 for label in ("eng-fra", "fra-eng")}
    return {"accuracy": sum(wins)/256, "per_label": labels, "audit_pairs": 256,
            "independent_groups": 64, "ties_are_failures": True}


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        raise RuntimeError("neural runtime is limited to the bounded public CPU workflow")


def helper_scope(extra):
    """The json key is the exact repaired dependency, not a changed helper."""
    import math as math_module
    import socket
    import subprocess
    import time
    import urllib.request
    from contextlib import contextmanager
    scope = {"json": json, "struct": struct, "math": math_module, "socket": socket, "os": os,
             "subprocess": subprocess, "time": time, "urllib": urllib, "contextmanager": contextmanager,
             "model_remote_only": remote_only, "RELEASE_HASHES": FOUNDATION, "MAX_RUNTIME_SECONDS": 6900,
             "MAX_LOG_BYTES": 16 << 20, "BATCH": 4, "FAMILY": FAMILY, **extra}
    tree = ast.parse(read_asset("runtime-helper.py.txt").decode())
    names = {"decode_bits", "check_response", "server"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("trusted helper function set changed")
    # SHA-verified function AST only. Module imports and main never execute.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-runtime-helper", "exec"), scope)  # noqa: S102
    return scope
