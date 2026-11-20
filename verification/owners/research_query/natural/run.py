# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Broader research-answer model on original human preferences, remote CPU only."""
import argparse
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

from data import HERE, digest, read_splits, write_json
from gates import controls, formatted
from runtime_check import decode_bits, model_remote_only, native_agreement, server

NATIVE_JSON = []
UNTRAINED = {}
RUNTIME = {}


def native(model, tok, trainer, rows):
    model_remote_only()
    import torch
    values = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            for side in ("chosen", "rejected"):
                encoded = tok(trainer.render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                if encoded["input_ids"].shape[1] > 512:
                    raise ValueError("original research reply was cropped")
                values.append(trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item())
    return values


def measured(rows, values):
    if len(rows) != 64 or len(values) != 128 or len({row["group_id"] for row in rows}) != 64:
        raise ValueError("64 complete independent source preferences required")
    wins = [values[2 * i] > values[2 * i + 1] for i in range(64)]
    slices = {"chosen_longer": [], "chosen_shorter": []}
    for row, win in zip(rows, wins):
        slices["chosen_longer" if len(row["chosen"]) > len(row["rejected"]) else "chosen_shorter"].append(win)
    if any(len(values) != 32 for values in slices.values()):
        raise ValueError("balanced preferred-length source slices required")
    return {"accuracy": sum(wins) / 64, "per_label": {name: sum(values) / len(values) for name, values in slices.items()}}


def record_development(model, tok, trainer, root):
    model_remote_only()
    _, split = read_splits(HERE / "fixtures")
    result = {name: measured(split["validation"], native(model, tok, trainer, formatted(split["validation"], name))) for name in ("prose", "json")}
    accepted = all(value["accuracy"] >= .8 and min(value["per_label"].values()) >= .7 for value in result.values())
    write_json(root / "validation-acceptance.json", {"formats": result, "accepted": accepted, "selection_split": "validation", "audit_scored": False})
    return accepted


def record_audit(model, tok, trainer, rows):
    model_remote_only()
    NATIVE_JSON.extend(native(model, tok, trainer, formatted(rows, "json")))


def record_initial(model, tok, trainer, rows, prose):
    model_remote_only()
    UNTRAINED["prose"] = prose["pairwise_accuracy"]
    UNTRAINED["json"] = measured(rows, native(model, tok, trainer, formatted(rows, "json")))["accuracy"]


def runtime(root, manifest, native_prose):
    model_remote_only()
    from tokenizers import Tokenizer
    from runtime_check import RELEASE_HASHES
    for name, expected in RELEASE_HASHES.items():
        fingerprint = hashlib.sha256()
        with (root / "foundation" / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                fingerprint.update(chunk)
        if fingerprint.hexdigest() != expected:
            raise ValueError("actual released foundation/runtime/tokenizer bytes changed")
    _, split = read_splits(root / "data")
    rows = split["audit"]
    tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    frozen = json.loads((root / "data/frozen-controls.json").read_text())
    result = {"verdict": "PASS", "formats": {}, "vectors": [], "negative_identity_controls": 0}
    for name, native_values in (("prose", native_prose), ("json", NATIVE_JSON)):
        represented = formatted(rows, name)
        from data import render
        vectors = [tokenizer.encode(render(row["prompt"], row[side])).ids for row in represented for side in ("chosen", "rejected")]
        if any(not 1 <= len(tokens) <= 512 for tokens in vectors):
            raise ValueError("literal complete format exceeded node cap")
        with server(root, manifest, name) as score:
            streams = [[bit for start in range(0, 128, 4) for bit in score(vectors[start:start + 4])] for _ in range(2)]
            for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
                try:
                    score(vectors[:1], {key: "0" * 64})
                except ValueError:
                    result["negative_identity_controls"] += 1
                else:
                    raise ValueError("wrong source identity escaped literal runtime check")
        with server(root, manifest, name + "-fresh") as score:
            fresh = [bit for start in range(0, 8, 4) for bit in score(vectors[start:start + 4])]
        if streams[0] != streams[1] or fresh != streams[0][:8]:
            raise ValueError("reference bits changed across repeats or process")
        agreement = native_agreement(native_values, streams[0])
        metrics = measured(rows, [decode_bits(bit) for bit in streams[0]])
        metrics.update(untrained_accuracy=UNTRAINED[name], shortcut_accuracy=max(controls(represented, frozen, tokenizer).values()))
        if not agreement["numerical"] or not agreement["ordering"] or metrics["accuracy"] < .8 or min(metrics["per_label"].values()) < .7 or metrics["accuracy"] < metrics["untrained_accuracy"] + .05 or metrics["accuracy"] < metrics["shortcut_accuracy"] + .10 or metrics["shortcut_accuracy"] > .65:
            result["verdict"] = "FAIL"
        result["formats"][name] = metrics
        result["vectors"].extend({"tokens": tokens, "score_bits": bit} for tokens, bit in zip(vectors[:8], streams[0][:8]))
        write_json(root / ("runtime-natural-" + name + ".json"), {"metrics": metrics, "repeated_bits": streams, "fresh_bits": fresh, "native_agreement": agreement})
    RUNTIME.update(result)
    write_json(root / "runtime-natural-combined.json", result)
    return result


def main():
    model_remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("bounded runner storage required")
    def stop(signum, frame):
        raise TimeoutError("bounded natural preference job stopped")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(300 * 60)
    import run_base as base
    base.record_natural_development = record_development
    base.record_natural_audit = record_audit
    base.record_natural_initial = record_initial
    base.execute_runtime = runtime
    try:
        code = base.experiment(root)
        if code:
            return code
        manifest = json.loads((root / "package-manifest.json").read_text())
        source = json.loads((root / "data/source-reverification.json").read_text())
        if source["grants_verified"] is not True or source["checked_preference_pairs"] != 1220:
            raise ValueError("full source/rank verification not completed")
        from runtime_check import RELEASE_HASHES
        report = {"schema_version": 1, "status": "REGISTRATION_READY", "family": "RESEARCH_QUERY", "run_id": int(os.environ["GITHUB_RUN_ID"]),
                  "commit": os.environ["GITHUB_SHA"], "serving_commit": os.environ["GITHUB_SHA"], "package_sha256": manifest["sha256"],
                  "package_keccak": manifest["keccak256"].removeprefix("0x"), "package_bytes": manifest["bytes"],
                  "foundation_sha256": RELEASE_HASHES["foundation_lora_fp32.onnx"], "engine_sha256": RELEASE_HASHES["foundation_engine.wasm"],
                  "tokenizer_sha256": RELEASE_HASHES["tokenizer.json"],
                  "source": {"normalized_overlap": source["normalized_overlap"], "audit_used_for_selection": False,
                             "grants_verified": source["grants_verified"], "source_manifest_sha256": digest((HERE / "source-manifest.json").read_bytes()),
                             "independent_groups": 64, "scope": "original source human preference relevance/usefulness, not factual truth"},
                  "quality": {"passed": RUNTIME["verdict"] == "PASS", "audit_pairs": 64, "all_supported_formats_tested": True, "formats": RUNTIME["formats"]},
                  "runtime": {"full_parser_passed": manifest["manifest"]["verification"]["stage1"], "reference_engine_passed": RUNTIME["verdict"] == "PASS",
                              "repeat_bits_identical": True, "negative_identity_controls": RUNTIME["negative_identity_controls"], "vectors": RUNTIME["vectors"]}}
        write_json(root / "registration-ready.json", report)
        return 0
    finally:
        base.preserve(root)


if __name__ == "__main__":
    raise SystemExit(main())
