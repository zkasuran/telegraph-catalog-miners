# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only process-tree RSS and time ceiling. Does not import a model."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from remote import remote_only
from source import write


def timeout_for(profile, command, config):
    if profile == "train":
        return config["timeout_seconds"]
    if profile != "serve" or len(command) < 2 or Path(command[1]).resolve() != Path(__file__).parent.resolve() / "serve.py":
        raise ValueError("serving profile only permits the exact owned serving entry point")
    return 19800  # 30-minute preparation plus the exact five-hour public lease.


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", choices=("train", "serve"), default="train")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        raise ValueError("bounded child command required")
    import psutil
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    timeout = timeout_for(args.profile, args.command, config)
    ceiling = min(config["max_rss_bytes"], psutil.virtual_memory().total * 3 // 4)
    started = time.monotonic()
    child = subprocess.Popen(args.command, start_new_session=True)
    observed, reason = 0, None
    try:
        while child.poll() is None:
            process = psutil.Process(child.pid)
            rss = 0
            for p in [process, *process.children(recursive=True)]:
                try:
                    rss += p.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            observed = max(observed, rss)
            if rss > ceiling or time.monotonic() - started > timeout:
                reason = "RSS ceiling" if rss > ceiling else "time ceiling"
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                break
            time.sleep(2)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
        write(args.root / "supervisor.json", {"peak_rss_bytes": observed, "rss_ceiling_bytes": ceiling,
              "profile": args.profile, "timeout_seconds": timeout,
              "elapsed_seconds": time.monotonic() - started, "termination_reason": reason,
              "returncode": child.returncode, "run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"]})
    return 0 if child.returncode == 0 and not reason else child.returncode if child.returncode and child.returncode > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
