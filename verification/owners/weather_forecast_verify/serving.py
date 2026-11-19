# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Serve only a passed immutable package with an ephemeral restricted public lease."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from build import sha, write_json
from engine import executor
from lease_common import DEFAULT_OWNER, HOST, HOST_KEY, SSH_USER, immutable_url, lease_for
from protocol import score
from reference import compile_tools, fetch, foundation, remote_only, source_kit, validate_package
from ready_contract import validate_ready


def serve(root, payload):
    remote_only()
    immutable_url(payload["package_url"])
    immutable_url(payload["report_url"])
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "registration-ready.json"
    fetch(payload["report_url"], report_path, payload["report_sha256"], 1 << 20)
    ready = json.loads(report_path.read_text())
    validate_ready(ready)
    if ready["serving_commit"] != os.environ["GITHUB_SHA"] or ready["family"] != "WEATHER_FORECAST_VERIFY":
        raise ValueError("serving source commit or family differs from actual audited report")
    slot, port = int(payload["slot"]), int(payload["forward_port"])
    if not 1 <= slot <= 6 or port != 48600 + slot or not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", payload["runnable_url"]):
        raise ValueError("only the adapter-reserved free endpoint is allowed")
    package = root / "WEATHER_FORECAST_VERIFY.safetensors"
    fetch(payload["package_url"], package, ready["package_sha256"], 64 << 20)
    foundation(root)
    kit = source_kit(root)
    compile_tools(root, kit)
    # The original audit contains the parser's exact public layout as safe evidence.
    layout = ready["runtime"]["foundation_spec"]
    write_json(root / "foundation.json", layout)
    checked = validate_package(root, package, "serving-parser.json")
    if checked["keccak256"] != "0x" + ready["package_keccak"]:
        raise ValueError("served package differs from exact actual audit")
    identity = {"foundation_hash": ready["foundation_sha256"], "adapter_hash": ready["package_keccak"],
                "tokenizer_hash": ready["tokenizer_sha256"]}
    with executor(root, package, "serving") as url:
        vectors = ready["runtime"]["vectors"]
        for offset in range(0, len(vectors), 4):
            batch = vectors[offset:offset + 4]
            if score(url, [v["tokens"] for v in batch], identity) != [v["score_bits"] for v in batch]:
                raise ValueError("serving executor does not reproduce audited literal vectors")
        key = root / "forward-key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "weather-restricted-lease"],
                       check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        key.chmod(0o600)
        known = root / "known_hosts"
        known.write_text(HOST_KEY)
        proof_hash = sha(report_path.read_bytes())
        row = {"family": ready["family"], "owner": DEFAULT_OWNER, "slot": slot, "keccak": ready["package_keccak"],
               "sha256": ready["package_sha256"], "forward_port": port, "runnable_url": payload["runnable_url"],
               "existing_registration_id": 0,
               **{name + "_proof": {"sha256": proof_hash} for name in ("candidate", "runtime", "source")}}
        lease = lease_for(row, key.with_suffix(".pub").read_text(), "weather-owner-lease", os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
        write_json(root / "lease.json", lease)
        local_port = url.split(":")[2].split("/")[0]
        command = ["ssh", "-N", "-i", str(key), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                   "-o", "UserKnownHostsFile=" + str(known), "-o", "ExitOnForwardFailure=yes",
                   "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
                   "-R", f"127.0.0.1:{port}:127.0.0.1:{local_port}", SSH_USER + "@" + HOST]
        # The adapter installs only the uploaded restricted public lease, never this key.
        deadline = time.monotonic() + 5 * 3600
        while time.monotonic() < deadline:
            tunnel = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                while tunnel.poll() is None and time.monotonic() < deadline:
                    time.sleep(2)
            finally:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
                    tunnel.wait(timeout=5)
            time.sleep(10)


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    for name in ("package-url", "report-url", "report-sha256", "slot", "forward-port", "runnable-url"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    serve(args.root, vars(args))
