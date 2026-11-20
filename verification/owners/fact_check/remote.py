# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Remote-only owner entry point. Exact same frozen experiment, no local models."""
import argparse
import os
import signal
from pathlib import Path

from data import write_json
from preserve import preserve
from report import emit
from run import experiment, remote_only


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("runner-temporary output only")
    def stop(signum, frame):
        raise TimeoutError("bounded owner CPU job stopped")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(300 * 60)
    preserve(args.root)
    try:
        result = experiment(args.root)
        if result == 0:
            emit(args.root)
        return result
    except Exception as error:
        write_json(args.root / "owner-failure.json", {"status": "INCOMPLETE", "error": type(error).__name__ + ": " + str(error)})
        raise
    finally:
        preserve(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
