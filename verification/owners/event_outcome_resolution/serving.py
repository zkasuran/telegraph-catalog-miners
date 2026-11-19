# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded free public CPU lease for the exact verified package and engine."""
import argparse
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from assets import FAMILY, HERE, fetch, prepare, remote_only
from common import DEFAULT_OWNER, HOST, HOST_KEY, WINDOW, lease_for, sha, strict_json, write_json
from contracts import validate_ready
from proxy import Backend, identity
from runtime import compile_runtime, executor, negative_controls, parse_package, score_vectors


def public_asset(url):
    prefix = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/"
    if not url.startswith(prefix) or not re.fullmatch(r"[0-9a-f]{40}/verification/owners/event_outcome_resolution/[A-Za-z0-9_./-]+", url[len(prefix):]) or ".." in url.split("/"):
        raise ValueError("owned immutable public asset URL required")
    return url


def forward_command(root, row, port):
    return ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"),
            "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(root / "known_hosts"),
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
            "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no",
            "-o", "RequestTTY=no", "-o", "LogLevel=ERROR", "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{port}",
            "telegraph-verify@" + HOST]


def run(root):
    remote_only()
    root.mkdir(parents=True, exist_ok=True)
    public_asset(os.environ["PACKAGE_URL"])
    public_asset(os.environ["REPORT_URL"])
    fetch(os.environ["REPORT_URL"], root / "registration-ready.json", os.environ["REPORT_SHA256"], 64 << 10)
    ready = validate_ready(strict_json((root / "registration-ready.json").read_bytes()))
    if (ready["serving_commit"] != os.environ["GITHUB_SHA"] or ready["package_sha256"] != os.environ["PACKAGE_SHA256"] or
            ready["package_keccak"] != os.environ["PACKAGE_KECCAK"]):
        raise ValueError("serving source/package/report identity mismatch")
    package = root / (FAMILY + ".safetensors")
    fetch(os.environ["PACKAGE_URL"], package, ready["package_sha256"], 64 << 20)
    if package.stat().st_size != ready["package_bytes"]:
        raise ValueError("serving package byte count mismatch")
    _, _, kit = prepare(root)
    compile_runtime(root, kit)
    parsed = parse_package(root, package, ready["package_sha256"])
    if parsed["keccak256"] != "0x" + ready["package_keccak"]:
        raise ValueError("serving full parser keccak mismatch")
    row = {"family": FAMILY, "owner": DEFAULT_OWNER, "slot": int(os.environ["SLOT"]),
           "forward_port": int(os.environ["FORWARD_PORT"]), "runnable_url": os.environ["RUNNABLE_URL"],
           "keccak": ready["package_keccak"], "sha256": ready["package_sha256"], "existing_registration_id": 0,
           **{name + "_proof": {"sha256": os.environ["REPORT_SHA256"]} for name in ("candidate", "runtime", "source")}}
    if not 1 <= row["slot"] <= 6 or row["forward_port"] != 48600 + row["slot"] or not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", row["runnable_url"]):
        raise ValueError("reserved free endpoint identity mismatch")
    with executor(root, package, parsed) as (runtime_row, backend, port):
        vectors = ready["runtime"]["vectors"]
        actual = score_vectors(backend, runtime_row, [v["tokens"] for v in vectors])
        if actual != [v["score_bits"] for v in vectors]:
            raise ValueError("serving engine differs from verified fixed vectors")
        negative_controls(port, runtime_row, backend)
        # Only this runner's fresh forwarding key is generated. It is never read
        # by an agent, uploaded, logged or used for signing.
        key = root / "forward-key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "event-owner-forward"],
                       check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        key.chmod(0o600)
        (root / "known_hosts").write_text(HOST_KEY)
        lease = lease_for(row, (root / "forward-key.pub").read_text().strip(), "event-outcome-resolution",
                          os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
        write_json(root / "lease.json", lease)
        deadline = time.monotonic() + WINDOW
        while time.monotonic() < deadline:
            forwarding = subprocess.Popen(forward_command(root, row, port), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                while forwarding.poll() is None and time.monotonic() < deadline:
                    time.sleep(2)
            finally:
                forwarding.terminate()
                try:
                    forwarding.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    forwarding.kill()
                    forwarding.wait(timeout=5)
            # The automatic adapter installs the restricted public key only after
            # reading the real lease artifact. Retry that same ephemeral tunnel.
            if time.monotonic() < deadline:
                time.sleep(5)
    write_json(root / "ended.json", {"status": "lease_window_ended"})


def monitor(root, lease_only):
    remote_only()
    deadline = time.monotonic() + (1200 if lease_only else WINDOW + 1200)
    while time.monotonic() < deadline:
        pidfile = root / "pid.txt"
        if pidfile.exists():
            pid = int(pidfile.read_text())
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                if not lease_only and (root / "ended.json").exists():
                    return
                raise ValueError("serving worker exited before its verified lease window")
        if lease_only and (root / "lease.json").exists():
            return
        if not lease_only and (root / "ended.json").exists():
            return
        time.sleep(2)
    raise ValueError("serving startup or bounded lease deadline exceeded")


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "lease", "wait"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.root)
    else:
        monitor(args.root, args.mode == "lease")
