# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Keep one already verified runner process alive without reading any private key."""
import argparse
import os
from pathlib import Path
import time
from reference import remote_only


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root
    pid = int((root / "process.pid").read_text())
    if pid <= 1:
        raise ValueError("invalid exact serving process")
    deadline = time.monotonic() + 5 * 3600
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(2)


if __name__ == "__main__":
    main()
