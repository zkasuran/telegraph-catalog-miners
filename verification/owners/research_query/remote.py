# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Owner's bounded CPU training and literal reference-engine qualification."""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from common import HERE, SHARED, capture_class, module, remote_only
from prepare import render, replay, check_splits
from report import ENGINE, FOUNDATION, TOKENIZER, accuracy, assemble, bits_float, formatted, literal_reply
from source import sha, write_json
from terms import preserve

AUDIT = {}
UNTRAINED = {}
RUNTIME = {}
SHORTCUTS = {}


def native_scores(model, tokenizer, trainer, rows):
    remote_only()
    import torch
    model.eval()
    scores = []
    with torch.no_grad():
        for row in rows:
            for side in ("chosen", "rejected"):
                encoded = tokenizer(render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                if encoded["input_ids"].shape[1] > 256:
                    raise ValueError("full supported-format answer exceeded export cap")
                scores.append(trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item())
    return scores


def record_audit(model, tokenizer, trainer, rows):
    remote_only()
    AUDIT["json"] = native_scores(model, tokenizer, trainer, formatted(rows, "json"))


def record_untrained(model, tokenizer, trainer, rows, measured_prose):
    remote_only()
    UNTRAINED["prose"] = {"accuracy": measured_prose["pairwise_accuracy"]}
    UNTRAINED["json"] = accuracy(rows, native_scores(model, tokenizer, trainer, formatted(rows, "json")))


def runtime_check(root, kit, rows, native):
    remote_only()
    from tokenizers import Tokenizer
    contract = json.loads((HERE / "contract.json").read_text())
    for name in ("qa-experiment/data.py", "qa-experiment/live_class.py"):
        if sha((SHARED / name).read_bytes()) != contract["shared_source_sha256"][name]:
            raise ValueError("runtime dependency source changed")
    sys.path.insert(0, str(SHARED / "qa-experiment"))
    transport = module("owner_runtime_transport", SHARED / "qa-experiment/runtime_check.py")
    sys.path.pop(0)
    transport.FAMILY, transport.CHILD = "RESEARCH_QUERY", "RESEARCH_QUERY_GENERAL"
    transport.check_response = literal_reply
    subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / "adapterexec"), "./cmd/adapterexec"],
                   cwd=kit, check=True, timeout=600)
    snapshot = capture_class(root)
    package = root / "RESEARCH_QUERY.safetensors"
    result = subprocess.run([str(root / "packagecheck"), "--intent", "RESEARCH_QUERY", "--child", "RESEARCH_QUERY_GENERAL",
                             "--class-snapshot", str(root / "class-snapshot.json"), "--snapshot-sha256", snapshot,
                             "--package", str(package), "--sha256", sha(package.read_bytes()), "--foundation", str(root / "foundation.json"),
                             "--foundation-sha256", sha((root / "foundation.json").read_bytes()), "--out", str(root)],
                            check=True, capture_output=True, timeout=60)
    manifest = json.loads(result.stdout)
    if manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("full current parser did not execute successfully")
    write_json(root / "package-manifest.json", manifest)
    import hashlib
    for name, expected in (("foundation_lora_fp32.onnx", FOUNDATION), ("foundation_engine.wasm", ENGINE), ("tokenizer.json", TOKENIZER)):
        digest = hashlib.sha256()
        with (root / "foundation" / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("actual foundation, engine or tokenizer bytes changed")
    tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    native_by_format = {"prose": native, "json": AUDIT["json"]}
    RUNTIME.update({"formats": {}, "vectors": [], "negative_identity_controls": 0, "passed": True,
                    "full_parser_passed": manifest["manifest"]["verification"]["stage1"]})
    frozen = json.loads((HERE / "data/frozen-controls.json").read_text())
    for name in ("prose", "json"):
        actual_rows = formatted(rows, name)
        vectors = [tokenizer.encode(render(row["prompt"], row[side])).ids for row in actual_rows for side in ("chosen", "rejected")]
        if len(vectors) != 128 or any(not 1 <= len(ids) <= 256 for ids in vectors):
            raise ValueError("full supported format was clipped or missing")
        with transport.server(root, manifest, name) as score:
            streams = [[bit for start in range(0, 128, 4) for bit in score(vectors[start:start + 4])] for _ in range(2)]
            for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
                try:
                    score(vectors[:1], {key: "0" * 64})
                except ValueError:
                    RUNTIME["negative_identity_controls"] += 1
                else:
                    raise ValueError("wrong identity escaped the literal wire gate")
        with transport.server(root, manifest, name + "-fresh") as score:
            fresh = [bit for start in range(0, 8, 4) for bit in score(vectors[start:start + 4])]
        if streams[0] != streams[1] or fresh != streams[0][:8]:
            raise ValueError("actual repeated or fresh-process score bits changed")
        agreement = transport.native_agreement(native_by_format[name], streams[0])
        measured = accuracy(rows, [bits_float(bit) for bit in streams[0]])
        RUNTIME["formats"][name] = measured
        SHORTCUTS[name] = replay(actual_rows, frozen, tokenizer)
        RUNTIME["passed"] = RUNTIME["passed"] and agreement["numerical"] and agreement["ordering"] and measured["accuracy"] >= .8 and min(measured["per_label"].values()) >= .7
        RUNTIME["vectors"].extend({"tokens": vector, "score_bits": bit} for vector, bit in zip(vectors[:8], streams[0][:8]))
        write_json(root / ("runtime-" + name + ".json"), {"metrics": measured, "native_agreement": agreement,
                   "score_bits": streams, "fresh_bits": fresh, "package_sha256": manifest["sha256"],
                   "actual_run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"]})
    write_json(root / "runtime-combined.json", RUNTIME)
    return RUNTIME["passed"]


def experiment(root):
    remote_only()
    import train_core as core
    original_reader = core.read_data
    def frozen_reader():
        contract, splits = original_reader()
        overlap = check_splits(splits)
        core.source_observed = max(overlap.values())
        training = []
        for row in splits["train"]:
            name = "prose" if int(sha(row["group_id"].encode()), 16) % 2 == 0 else "json"
            training.extend(formatted([row], name))
        splits["train"] = training
        splits["validation"] = formatted(splits["validation"], "prose") + formatted(splits["validation"], "json")
        return contract, splits
    core.read_data = frozen_reader
    core.record_owner_audit = record_audit
    core.record_owner_untrained = record_untrained
    core.runtime_check = runtime_check
    code = core.experiment(root)
    contract = json.loads((HERE / "contract.json").read_text())
    original = json.loads((root / "result.json").read_text())
    manifest = json.loads((root / "package-manifest.json").read_text())
    if code:
        return code
    ready = assemble(root, contract, manifest, RUNTIME, UNTRAINED, SHORTCUTS, original)
    write_json(root / "registration-ready.json", ready)
    return 0


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("only bounded runner storage is allowed")
    def stop(signum, frame):
        raise TimeoutError("bounded remote job stopped: " + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(300 * 60)
    try:
        return experiment(root)
    except Exception as exc:
        write_json(root / "failure.json", {"status": "INCOMPLETE", "error": type(exc).__name__ + ": " + str(exc),
                   "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"]})
        raise
    finally:
        preserve(root)


if __name__ == "__main__":
    raise SystemExit(main())
