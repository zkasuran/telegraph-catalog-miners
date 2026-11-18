#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Prepare one byte-exact evaluator registration integration test."""
import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import urllib.request
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT_COMMIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"
FOUNDATION = {
    "foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
    "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
    "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456",
}

def fetch(url, target, expected=None, ceiling=1 << 30):
    request = urllib.request.Request(url, headers={"User-Agent": "Telegraph-evaluator-integration/1.0"})
    for attempt in range(3):
        digest = hashlib.sha256()
        total = 0
        deadline = time.monotonic() + 600
        try:
            with urllib.request.urlopen(request, timeout=60) as source, target.open("wb") as out:
                while chunk := source.read(1 << 20):
                    total += len(chunk)
                    if total > ceiling or time.monotonic() > deadline:
                        raise RuntimeError("download exceeds its byte or time ceiling")
                    digest.update(chunk)
                    out.write(chunk)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    if expected and digest.hexdigest() != expected:
        raise RuntimeError("download hash mismatch: " + target.name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    inventory = json.loads((HERE / "champions.json").read_text())
    row = next((r for r in inventory["rows"] if r["family"] == args.family), None)
    if row is None:
        raise SystemExit("family is not in the exact champion inventory")
    root = Path(os.environ["RUNNER_TEMP"]) / "telegraph-evaluator"
    root.mkdir(exist_ok=True)
    model = root / "foundation"
    model.mkdir(exist_ok=True)
    for name, digest in FOUNDATION.items():
        fetch(RELEASE + name, model / name, digest)
    package = root / row["file"]
    fetch(row["url"], package, row["sha256"], 64 << 20)
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT_COMMIT, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + KIT_COMMIT)
    executable = root / "adapterexec"
    subprocess.run(["go", "build", "-trimpath", "-o", str(executable), "./cmd/adapterexec"], cwd=kit, check=True, timeout=600)
    private_key = root / "forward-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private_key), "-C", "telegraph-verify-" + args.family], check=True, timeout=15)
    private_key.chmod(0o600)
    known = root / "known_hosts"
    known.write_text("98.95.110.88 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMRYtmP2ESjH1ah2MI3eqUmEgtVYeyR81RAbr0MB/r0G\n")
    state = {"row": row, "root": str(root), "model": str(model), "package": str(package), "executable": str(executable), "private_key": str(private_key), "known_hosts": str(known)}
    (root / "state.json").write_text(json.dumps(state))
    lease = {"family": args.family, "slot": row["slot"], "run_id": os.environ["GITHUB_RUN_ID"], "batch": os.environ["VERIFY_BATCH"],
             "public_key": private_key.with_suffix(".pub").read_text().strip(), "adapter_hash": row["keccak"],
             "runnable_url": f"https://tga{row['slot']}.zkasuran.dev/runnable"}
    (root / "lease.json").write_text(json.dumps(lease, indent=2))
    print("Prepared one exact package, pinned foundation and restricted-forwarding public key")

if __name__ == "__main__":
    main()
