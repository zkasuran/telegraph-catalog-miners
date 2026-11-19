# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Preserve successful/failed deltas with the same retained licence boundary."""
import argparse
import json
import zipfile
from pathlib import Path

from contracts import HERE, file_sha, remote_only, write_json


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root/"licensed-checkpoint.zip"
    manifest = []
    with zipfile.ZipFile(checkpoint, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("LICENSE", "NOTICE", "DATA-SOURCES.md", "CC-BY-4.0.txt", "licences.json"):
            archive.write(HERE/name, name)
        for directory in ("trained-adapter", "untrained-adapter"):
            path = root/directory
            if not path.exists():
                continue
            for name in ("adapter_model.safetensors", "adapter_config.json", "reward_head.pt"):
                source = path/name
                if source.is_file() and source.stat().st_size <= 64 << 20:
                    archive.write(source, directory+"/"+name)
                    manifest.append({"path": directory+"/"+name, "sha256": file_sha(source), "bytes": source.stat().st_size})
        archive.writestr("CHECKPOINT-NOTICE.json", json.dumps({"status": "UNQUALIFIED_CHECKPOINT", "files": manifest,
            "licence_boundary": "Our deltas SAND; retained Apache/MIT/CC BY grants remain attached. No human preference review or canonical result claimed."}))
    write_json(root/"checkpoint-manifest.json", {"files": manifest, "zip_sha256": file_sha(checkpoint)})


if __name__ == "__main__":
    main()
