# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pinned public inputs. Neural assets are fetched only by the owned CPU workflow."""
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path

from common import FOUNDATION, FOUNDATION_RELEASE, KIT, KIT_SHA, check_classes, now, public_json, sha, write_json

HERE = Path(__file__).resolve().parent
FAMILY = "EVENT_OUTCOME_RESOLUTION"
CHILD = FAMILY + "_GENERAL"
WORKFLOW = "Owner EVENT_OUTCOME_RESOLUTION"


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or
            os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners" or
            os.environ.get("GITHUB_WORKFLOW") != WORKFLOW or
            os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"):
        raise RuntimeError("neural work requires the exact owned public CPU workflow")


def fetch(url, target, expected, ceiling, remote=True):
    if remote:
        remote_only()
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("pinned source digest required")
    partial = target.with_name(target.name + ".partial")
    import hashlib
    digest = hashlib.sha256()
    total = 0
    deadline = time.monotonic() + 600
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Event-settlement-owner/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as output:
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > ceiling or time.monotonic() > deadline:
                    raise ValueError("source download size/time ceiling")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("pinned source hash mismatch")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def capture_classes(out):
    envelope = {"schema": 1}
    for name, url in (("family", "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY),
                      ("child", "https://devnode.telegraphprotocol.com/engine/v1/intents/" + CHILD)):
        _, raw = public_json(url)
        envelope[name] = {"url": url, "raw": raw.decode()}
    envelope["fetched_at"] = now().isoformat()
    try:
        check_classes(envelope, {"family": FAMILY, "child": CHILD, "class": "judgment"})
    except ValueError as error:
        # Preserve only operational identities, not source examples or pool
        # rows. This failure is not a measured model-quality report.
        parent = json.loads(envelope["family"]["raw"]).get("resolve", {})
        child = json.loads(envelope["child"]["raw"])
        write_json(out.with_suffix(".failure.json"), {
            "schema_version": 1, "status": "SOURCE_CLASS_GATE_FAILED", "family": FAMILY,
            "child": CHILD, "fetched_at": envelope["fetched_at"], "error": str(error),
            "expected_class": "judgment", "observed_parent_class": parent.get("class"),
            "observed_child_class": child.get("class"),
            "observed_child_served_by": child.get("onchain", {}).get("served_by"),
            "response_sha256": {name: sha(envelope[name]["raw"].encode()) for name in ("family", "child")},
            "stage": "preflight" if out.name == "class-preflight.json" else "class_capture"})
        raise
    write_json(out, envelope)
    return envelope


def prepare(root, training=False):
    remote_only()
    root.mkdir(parents=True, exist_ok=True)
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, digest in FOUNDATION.items():
        target = foundation / name
        if not target.exists() or sha(target.read_bytes()) != digest:
            fetch(FOUNDATION_RELEASE + name, target, digest, 1 << 30)
    lock = json.loads((HERE / "model-sources.json").read_text())
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT,
          archive, KIT_SHA, 32 << 20)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > 5000 or sum(m.size for m in members) > 128 << 20:
            raise ValueError("kit extraction ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + KIT)
    spec = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
            "tokenizer_hash": FOUNDATION["tokenizer.json"], "foundation_version": "modernbert-base-lora16-v1",
            "r_max": 16, "num_layers": 22, "hidden_size": 768,
            "targets": {"attn.Wqkv": [768, 2304], "attn.Wo": [768, 768], "mlp.Wi": [768, 2304], "mlp.Wo": [1152, 768]}}
    write_json(root / "foundation.json", spec)
    base = root / "base"
    if training:
        base.mkdir(exist_ok=True)
        (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
        for name, digest in lock["base"]["files"].items():
            fetch("https://huggingface.co/" + lock["base"]["id"] + "/resolve/" + lock["base"]["revision"] + "/" + name,
                  base / name, digest, 1 << 30)
        os.environ["HF_HUB_OFFLINE"] = "1"
    return foundation, base, kit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-classes", required=True, type=Path)
    args = parser.parse_args()
    envelope = capture_classes(args.capture_classes)
    child = json.loads(envelope["child"]["raw"])
    print(json.dumps({"family": FAMILY, "class": child["class"], "request": child["onchain"]["request"],
                      "response": child["onchain"]["response"], "source_sha256": sha(args.capture_classes.read_bytes())}, indent=2))
