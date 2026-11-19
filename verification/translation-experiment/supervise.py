# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One remote CPU process tree, fixed wall and measured RSS ceilings."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from checkpoint_terms import remote_guard
from source_probe import write


def rss(pid):
    total = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().split("\n"):
            if line.startswith("VmRSS:"):
                total = int(line.split()[1]) * 1024
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return total + sum(rss(int(child)) for child in children)
    except (FileNotFoundError, ProcessLookupError):
        return total


def main():
    remote_guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("supervisor output outside RUNNER_TEMP")
    root.mkdir(exist_ok=True)
    mem = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().split("\n") if line.startswith("MemTotal:"))
    ceiling = min(10 << 30, int(mem * 0.75))
    env = {k: v for k, v in os.environ.items() if k not in {"GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} and not k.startswith("AWS_")}
    command = [sys.executable, str(Path(__file__).with_name("remote_train.py")), "--root", str(root)]
    child = subprocess.Popen(command, env=env, start_new_session=True)
    started, peak, reason = time.monotonic(), 0, None
    while child.poll() is None:
        peak = max(peak, rss(child.pid))
        if peak > ceiling or time.monotonic() - started > 16800:
            reason = "process-tree RSS ceiling" if peak > ceiling else "fixed wall ceiling"
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            break
        time.sleep(1)
    write(root / "supervisor.json", {"returncode": child.returncode, "reason": reason, "elapsed_seconds": time.monotonic() - started,
          "peak_tree_rss_bytes": peak, "rss_ceiling_bytes": ceiling, "wall_seconds": 16800, "cpu_threads": 4,
          "status": "PASS" if child.returncode == 0 and reason is None else "FAIL"})
    return child.returncode if child.returncode and child.returncode > 0 else 1 if reason or child.returncode else 0


if __name__ == "__main__":
    raise SystemExit(main())
