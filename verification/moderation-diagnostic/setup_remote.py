# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Use the pinned shared preparation, without entering its training main."""
import argparse
from pathlib import Path

from core import load_shared, remote_only, write

if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    _, module, _ = load_shared()
    module.prepare(root)
    write(root / "evidence" / "foundation-binding.json", {"foundation": module.FOUNDATION,
          "base": module.SOURCES["base"], "kit": module.SOURCES["kit"], "hash_checks": "enforced by shared prepare"})
