# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One hosted CPU replay process tree, fixed wall and measured RSS limits."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from contracts import remote_only
from replay import write


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
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("output must remain in the hosted runner temporary directory")
    root.mkdir(parents=True, exist_ok=True)
    memory = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().split("\n") if line.startswith("MemTotal:"))
    ceiling = min(10 << 30, int(memory * 0.75))
    started, peak, reason = time.monotonic(), 0, None
    child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("replay.py")), "--root", str(root)], start_new_session=True)
    while child.poll() is None:
        peak = max(peak, rss(child.pid))
        if peak > ceiling or time.monotonic() - started > 9300:
            reason = "RSS ceiling" if peak > ceiling else "wall ceiling"
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            break
        time.sleep(1)
    write(root / "supervisor.json", {"status": "PASS" if child.returncode == 0 and reason is None else "FAIL",
          "returncode": child.returncode, "reason": reason, "peak_process_tree_rss": peak,
          "rss_ceiling": ceiling, "wall_ceiling_seconds": 9300, "new_training": False})
    return 0 if child.returncode == 0 and reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
