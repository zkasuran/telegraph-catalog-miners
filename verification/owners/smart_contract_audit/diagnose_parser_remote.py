# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded real parser diagnostic on frozen bytes. No training or neural imports."""
import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

from policy import remote_only
from remote_prepare import fetch, HERE
from class_source import capture

EXPECTED = {
    "untrained.safetensors": "e9b120b2606917c33eeb944ff8254b516c03c34a0aac917b3dad46d08b62c335",
    "candidate.safetensors": "27618af3d5e188abc2f53ee69c1134979311060f04c6561a3b9d1210289b00a5",
}


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root
    for name, expected in EXPECTED.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("original frozen package changed")
    sources = json.loads((HERE / "sources.json").read_text())
    pin = sources["kit"]
    archive = root / "parser-kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + pin["revision"], archive, pin["sha256"], 32 << 20)
    with tarfile.open(archive) as tar:
        if sum(m.size for m in tar.getmembers()) > 128 << 20:
            raise ValueError("source archive ceiling")
        tar.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + pin["revision"])
    directory = kit / "cmd/owner-classcheck"
    directory.mkdir(exist_ok=True)
    (directory / "main.go").write_bytes((HERE / "classcheck.go").read_bytes())
    checker = root / "parser-check"
    subprocess.run(["go", "build", "-trimpath", "-o", str(checker), "./cmd/owner-classcheck"], cwd=kit, check=True, timeout=600)
    source = capture(root / "class-snapshot.json")
    foundation_sha = hashlib.sha256((root / "foundation.json").read_bytes()).hexdigest()
    report = {"schema_version": 1, "status": "PARSER_DIAGNOSTIC", "family": "SMART_CONTRACT_AUDIT",
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
              "training_origin": {"run_id": 44577682115, "commit": "0077ea0fab7c15cc0a026016fbf0e07591aec71c", "artifact_id": 14556039711},
              "training_or_audit_ran": False, "package_bytes_unchanged": True, "results": {}}
    for name in EXPECTED:
        args = [str(checker), "--package", str(root / name), "--sha256", EXPECTED[name],
                "--class-snapshot", source["path"], "--snapshot-sha256", source["sha256"],
                "--foundation", str(root / "foundation.json"), "--foundation-sha256", foundation_sha,
                "--out", str(root / "parser-output" / name)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        (root / (name + ".parser.stdout.json")).write_text(result.stdout)
        (root / (name + ".parser.stderr.log")).write_text(result.stderr)
        report["results"][name] = {"returncode": result.returncode, "stderr": result.stderr, "package_sha256": EXPECTED[name], "actual_class_source": source}
        (root / "parser-diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if any(r["returncode"] for r in report["results"].values()):
        raise RuntimeError("real pinned parser refused frozen package; exact diagnostics retained")


if __name__ == "__main__":
    main()
