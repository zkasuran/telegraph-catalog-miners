# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Pinned public-CPU serving with only a restricted public lease uploaded."""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from data import write_json
from fetch_source import fetch
from foundation import prepare_kit
from foundation_pins import FOUNDATION, RELEASE
from lease_contract import lease_for
from runtime_check import compile_tools, model_remote_only, server

HOST = "98.95.110.88"
HOST_KEY = "98.95.110.88 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMRYtmP2ESjH1ah2MI3eqUmEgtVYeyR81RAbr0MB/r0G\n"
OWNER = "0x8b224783fe5b3c52b7db0cb9b1754f8812b75287"


def immutable_url(value):
    if not re.fullmatch(r"https://raw\.githubusercontent\.com/zkasuran/telegraph-catalog-miners/[0-9a-f]{40}/verification/owners/question_answering/[A-Za-z0-9_/.-]+", value) or ".." in value:
        raise ValueError("exact immutable owned public URL required")
    return value


def main():
    model_remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--package-url", required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--report-url", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--slot", required=True, type=int)
    parser.add_argument("--forward-port", required=True, type=int)
    parser.add_argument("--runnable-url", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()) or not 1 <= args.slot <= 64 or not 1024 <= args.forward_port <= 65535:
        raise ValueError("bounded runner storage/lease identity required")
    if not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", args.runnable_url):
        raise ValueError("existing proxy namespace required")
    root.mkdir(parents=True, exist_ok=True)
    for value in (args.package_sha256, args.report_sha256):
        if not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("complete exact SHA-256 required")
    fetch(immutable_url(args.report_url), root / "registration-ready.json", args.report_sha256, 1 << 20)
    report = json.loads((root / "registration-ready.json").read_text())
    if (report.get("status") != "REGISTRATION_READY" or report.get("family") != "QUESTION_ANSWERING"
            or report.get("serving_commit") != os.environ["GITHUB_SHA"] or report.get("package_sha256") != args.package_sha256):
        raise ValueError("serving source/report/package identity mismatch")
    fetch(immutable_url(args.package_url), root / "QUESTION_ANSWERING.safetensors", args.package_sha256, 64 << 20)
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, expected in FOUNDATION.items():
        fetch(RELEASE + name, foundation / name, expected)
    compile_tools(root, prepare_kit(root))
    manifest = {"keccak256": report["package_keccak"], "sha256": args.package_sha256}
    key = root / "ephemeral-forward-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True, timeout=30)
    key.chmod(0o600)
    known = root / "known-hosts"
    known.write_text(HOST_KEY)
    row = {"family": "QUESTION_ANSWERING", "owner": OWNER, "slot": args.slot, "forward_port": args.forward_port,
           "runnable_url": args.runnable_url, "keccak": report["package_keccak"], "sha256": args.package_sha256,
           "existing_registration_id": 0,
           **{name + "_proof": {"sha256": args.report_sha256} for name in ("candidate", "runtime", "source")}}
    lease = lease_for(row, key.with_suffix(".pub").read_text(), "qa-" + os.environ["GITHUB_RUN_ID"],
                      os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    # The standard runtime context verifies literal identities and exact scores.
    with server(root, manifest, "serving") as score:
        vectors = report["runtime"]["vectors"]
        observed = [bit for start in range(0, len(vectors), 4) for bit in score([v["tokens"] for v in vectors[start:start + 4]])]
        if observed != [v["score_bits"] for v in vectors]:
            raise ValueError("served reference bits differ from the qualified package")
        # Discover the exact executor listener only from its real bounded log.
        log = (root / "runtime-serving.log").read_text()
        match = re.search(r"adapterexec listening on 127\.0\.0\.1:([0-9]+)", log)
        if match is None:
            raise ValueError("reference listener not observed")
        local_port = match.group(1)
        write_json(root / "lease.json", lease)
        command = ["ssh", "-N", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                   "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(known),
                   "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "ConnectTimeout=15",
                   "-R", f"127.0.0.1:{args.forward_port}:127.0.0.1:{local_port}", "telegraph-verify@" + HOST]
        forwarding = None
        try:
            deadline = time.monotonic() + 5 * 3600 - 120
            while time.monotonic() < deadline:
                # The adapter installs the exact public lease after upload.
                # Connection retries use only this same restricted ephemeral key.
                if forwarding is None or forwarding.poll() is not None:
                    forwarding = subprocess.Popen(command)
                time.sleep(10)
        finally:
            if forwarding is not None:
                forwarding.terminate()
                forwarding.wait(timeout=20)


if __name__ == "__main__":
    main()
