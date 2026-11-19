# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded public CPU training/qualification driver. Never registers or signs."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from bootstrap import compile_tools, dump, prepare_kit, prepare_models, source_gate
from live_class import capture
from scorer import remote_only
from source import HERE


def command(arguments, root, name, timeout):
    start = time.monotonic()
    with (root / (name + ".log")).open("w") as log:
        result = subprocess.run(arguments, stdout=log, stderr=log, timeout=timeout)
    dump(root / (name + "-process.json"), {"arguments": arguments, "returncode": result.returncode, "seconds": time.monotonic() - start})
    if result.returncode:
        raise RuntimeError(name + " stage failed; read the real preserved log")


def train(root):
    remote_only()
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HOME"] = str(root / "public-hf-cache")
    dump(root / "run-identity.json", {"family": "ACADEMIC_SEARCH", "run_id": int(os.environ["GITHUB_RUN_ID"]),
         "commit": os.environ["GITHUB_SHA"], "repository": os.environ["GITHUB_REPOSITORY"], "runner_os": os.environ["RUNNER_OS"]})
    command([sys.executable, "-W", "error", str(HERE / "verify.py")], root, "source-gate", 120)
    source_gate(root)
    kit = prepare_kit(root)
    compile_tools(root, kit)
    live = capture(root / "class")
    preflight = [str(root / "packagecheck"), "--intent", "ACADEMIC_SEARCH", "--child", "ACADEMIC_SEARCH_GENERAL",
                 "--class-snapshot", str(root / "class/class-snapshot.json"), "--snapshot-sha256", live["snapshot_sha256"], "--preflight"]
    command(preflight, root, "live-preflight", 60)
    foundation, base = prepare_models(root)
    command([sys.executable, str(HERE / "scorer.py"), "--frozen", str(HERE / "frozen"), "--base", str(base), "--kit", str(kit),
             "--tokenizer", str(foundation / "tokenizer.json"), "--out", str(root)], root, "training", 155 * 60)
    remaining = 315 * 60 - (time.monotonic() - started)
    if remaining < 60:
        raise TimeoutError("bounded job has no remaining independent-audit reserve")
    command([sys.executable, str(HERE / "audit.py"), "--root", str(root), "--kit", str(kit)], root, "independent-audit-runtime", remaining)
    dump(root / "result.json", {"status": "COMPLETED_REMOTE_QUALIFICATION", "family": "ACADEMIC_SEARCH", "elapsed_seconds": time.monotonic() - started,
         "registration_ready_written": (root / "registration-ready.json").is_file(), "registration_sent": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        train(args.root)
    except Exception as exc:
        args.root.mkdir(parents=True, exist_ok=True)
        dump(args.root / "result.json", {"status": "FAILED", "family": "ACADEMIC_SEARCH", "error_type": type(exc).__name__, "error": str(exc),
             "run_id": os.environ.get("GITHUB_RUN_ID"), "commit": os.environ.get("GITHUB_SHA"), "registration_sent": False})
        traceback.print_exc()
        raise SystemExit(1)
