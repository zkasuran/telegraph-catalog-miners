# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Pinned shared tool reuse, with no edits to shared code or catalogs."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from source import KIT, read_url, sha, write_json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SHARED = HERE / "support"
KIT_SHA = "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243"
PREPARATION_SHA = "5b2ba64a11ff141ac96a20c491fc282f917e722183b2ab67c17e6a12c8ff519b"
FAMILY = "RESEARCH_QUERY"
CHILD = "RESEARCH_QUERY_GENERAL"


def module(name, path):
    contract_path = HERE / "contract.json"
    if contract_path.exists() and path.is_relative_to(SHARED):
        pins = json.loads(contract_path.read_text())["shared_source_sha256"]
        key = str(path.relative_to(SHARED))
        if key not in pins or sha(path.read_bytes()) != pins[key]:
            raise ValueError("shared source module identity changed: " + key)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners" or not os.environ.get("RUNNER_TEMP")):
        raise RuntimeError("neural work requires the controller's bounded public CPU GitHub dispatch")


def capture_class(out):
    out.mkdir(parents=True, exist_ok=True)
    urls = {"family": "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY,
            "child": "https://devnode.telegraphprotocol.com/engine/v1/intents/" + CHILD}
    snapshot = {"schema": 1, "fetched_at": datetime.now(timezone.utc).isoformat()}
    receipts = {}
    for key, url in urls.items():
        raw, receipt = read_url(url, 1 << 20)
        if receipt["final_url"] != url:
            raise ValueError("live class redirect")
        snapshot[key] = {"url": url, "raw": raw.decode()}
        receipts[key] = receipt
    parent = json.loads(snapshot["family"]["raw"])["resolve"]
    child = json.loads(snapshot["child"]["raw"])
    if (parent.get("intent") != FAMILY or parent.get("class") != "judgment"
            or child.get("class") != "judgment" or child.get("intent_id") != CHILD
            or child.get("evaluation", {}).get("kind") != "judgment"):
        raise ValueError("live class or identity mismatch")
    request = child.get("request", {})
    if request.get("served_by") != FAMILY or not any(p.get("name") == "question" and p.get("required") is True for p in request.get("params", [])):
        raise ValueError("live required research question schema changed")
    write_json(out / "class-snapshot.json", snapshot)
    write_json(out / "class-receipts.json", receipts)
    return sha((out / "class-snapshot.json").read_bytes())


def source_kit(root):
    root.mkdir(parents=True, exist_ok=True)
    fetcher = module("research_shared_fetch", SHARED / "training/data.py")
    archive = root / "kit.tar.gz"
    fetcher.fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT,
                  archive, KIT_SHA, 32 << 20)
    with tarfile.open(archive) as source:
        if sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError("source kit expansion ceiling")
        source.extractall(root, filter="data")
    return root / ("telegraph-wasm-baseline-" + KIT)


def prepare_foundation(root):
    remote_only()  # before any weight download or model-related import
    source = (SHARED / "training/run.py").read_bytes()
    if sha(source) != PREPARATION_SHA:
        raise ValueError("reviewed foundation preparation source changed")
    parsed = ast.parse(source)
    function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    fetcher = module("research_foundation_fetch", SHARED / "training/data.py")
    foundation_spec = module("research_foundation_spec", SHARED / "prepare.py")
    env = {"FOUNDATION": foundation_spec.FOUNDATION, "RELEASE": foundation_spec.RELEASE,
           "SOURCES": fetcher.SOURCES, "fetch": fetcher.fetch, "tarfile": tarfile, "sys": sys, "os": os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "reviewed_prepare", "exec"), env)
    return env["prepare"](root)


def compile_checker(root, kit, go):
    # Source specialization only, never a binary wrap. This preserves the full
    # upstream Stage 1 parser. The shared QA files remain unchanged on disk.
    contract = json.loads((HERE / "contract.json").read_text())
    for name in ("packagecheck.go", "packagecheck_test.go"):
        path = SHARED / "qa-experiment" / name
        raw = path.read_bytes()
        if sha(raw) != contract["shared_source_sha256"][name]:
            raise ValueError("reviewed source checker changed")
        text = raw.decode().replace("QUESTION_ANSWERING", FAMILY)
        (root / name).write_text(text)
    subprocess.run([go, "test", "-mod=readonly", "-v", str(root / "packagecheck.go"), str(root / "packagecheck_test.go")],
                   cwd=kit, check=True, timeout=600)
    subprocess.run([go, "build", "-mod=readonly", "-trimpath", "-o", str(root / "packagecheck"), str(root / "packagecheck.go")],
                   cwd=kit, check=True, timeout=600)
    snapshot_sha = capture_class(root)
    completed = subprocess.run([str(root / "packagecheck"), "--intent", FAMILY, "--child", CHILD,
                               "--class-snapshot", str(root / "class-snapshot.json"), "--snapshot-sha256", snapshot_sha, "--preflight"],
                               check=True, capture_output=True, timeout=60)
    result = json.loads(completed.stdout)
    if result.get("loaded") is not False or result.get("class") != "judgment":
        raise ValueError("full checker live-class preflight mismatch")
    write_json(root / "class-preflight.json", result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--go")
    args = parser.parse_args()
    if args.go:
        compile_checker(args.out.resolve(), source_kit(args.out.resolve()), args.go)
    else:
        print(capture_class(args.out))
