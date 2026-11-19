# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bound subprocess groups and preserve incremental diagnostics on failure."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from core import HERE, remote_only, sha, write
from preserve_all import preserve


def phase(root, name, command, seconds, deadline):
    remaining = deadline - time.monotonic() - 120
    if remaining <= 0:
        raise TimeoutError("deadline reserve exhausted")
    cap = min(seconds, remaining)
    evidence = root / "evidence"
    write(evidence / "phase.json", {"phase": name, "state": "running", "cap_seconds": cap})
    with (evidence / (name + ".log")).open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            raise TimeoutError("bounded phase expired: " + name)
    write(evidence / "phase.json", {"phase": name, "state": "complete" if code == 0 else "failed", "exit_code": code})
    if code:
        raise RuntimeError("phase failed: " + name)


def main(root):
    remote_only()
    if root.exists():
        raise ValueError("diagnostic output must be fresh; no audit rerun")
    root.mkdir()
    (root / "evidence").mkdir()
    deadline = time.monotonic() + 17400
    write(root / "evidence" / "run-binding.json", {"run_id": os.environ["GITHUB_RUN_ID"],
          "commit": os.environ["GITHUB_SHA"], "workflow": os.environ["GITHUB_WORKFLOW"],
          "diagnostic_files": {p.name: sha(p.read_bytes()) for p in HERE.iterdir() if p.is_file()},
          "local_model_execution": False, "runtime": "standard public CPU runner"})
    try:
        # Reuse only the existing pinned foundation preparation function.
        phase(root, "prepare", [sys.executable, str(HERE / "setup_remote.py"), "--root", str(root)], 1200, deadline)
        phase(root, "freeze-data", [sys.executable, str(HERE / "freeze_data.py"), "--root", str(root)], 900, deadline)
        phase(root, "initial-control", [sys.executable, str(HERE / "stages.py"), "initial", "--root", str(root)], 1200, deadline)
        fit_deadline = min(deadline, time.monotonic() + 12000)
        for arm in ("C3", "H6"):
            phase(root, "fit-" + arm, [sys.executable, str(HERE / "fit.py"), "--root", str(root), "--arm", arm], 12000, fit_deadline)
        phase(root, "select", [sys.executable, str(HERE / "stages.py"), "select", "--root", str(root)], 120, deadline)
        phase(root, "final-audit", [sys.executable, str(HERE / "stages.py"), "final", "--root", str(root)], 1800, deadline)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        write(root / "evidence" / "operational-failure.json", {"kind": type(error).__name__, "message": str(error),
              "release_gate": "INCOMPLETE", "parser": "not attempted", "runtime": "not attempted", "node": "not attempted"})
        return 1
    finally:
        preserve(root)


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    sys.exit(main(parser.parse_args().root))
