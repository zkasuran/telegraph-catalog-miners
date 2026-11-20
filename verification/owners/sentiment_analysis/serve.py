# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Keep one exact verified public executor alive on a bounded free runner."""
import argparse
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from common import HOST_KEY, lease_for, public_key, validate_lease
from contracts import (
    FAMILY,
    MAX_BODY,
    NODE,
    OWNER,
    fetch,
    file_sha,
    immutable_url,
    remote_only,
    strict_json,
    validate_ready,
    write_json,
)
from remote import assets, classes, executor, identity_controls, score, stage_check


def prepare(root, package_url, package_sha, package_keccak, slot, forward_port, runnable_url, existing=0):
    remote_only()
    immutable_url(package_url)
    report = validate_ready(strict_json((root / "registration-ready.json").read_bytes()))
    if report["package_sha256"] != package_sha or report["package_keccak"] != package_keccak or report["serving_commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("serving bytes/report/source identity mismatch")
    if not 1 <= slot <= 6 or forward_port != 48600 + slot or runnable_url != (
            "https://tga1.zkasuran.dev/_s2o_sentiment_analysis_" + package_keccak[:8] + "/runnable"):
        raise ValueError("reserved endpoint identity mismatch")
    package = root / "SENTIMENT_ANALYSIS-recovery.safetensors"
    fetch(package_url, package, package_sha)
    assets(root)
    manifest = stage_check(root, package, package_sha, "serving-parser")
    if manifest["keccak256"].removeprefix("0x") != package_keccak:
        raise ValueError("served package actual Keccak mismatch")
    report_sha = file_sha(root / "registration-ready.json")
    if type(existing) is not int or not 0 <= existing < 1 << 63:
        raise ValueError("explicit existing registration identity required for continuation")
    epoch = os.environ.get("LEASE_EPOCH", "initial")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", epoch):
        raise ValueError("bounded public lease epoch required")
    row = {"family": FAMILY, "owner": OWNER, "slot": slot, "forward_port": forward_port, "runnable_url": runnable_url,
           "keccak": package_keccak, "sha256": package_sha, "existing_registration_id": existing}
    for kind in ("candidate", "runtime", "source"):
        row[kind + "_proof"] = {"sha256": report_sha}
    key = root / "forward-key"
    if key.exists():
        raise ValueError("do not replace a live runner key")
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "s2-owner-sentiment"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    key.chmod(0o600)
    (root / "known_hosts").write_text(HOST_KEY)
    lease = lease_for(row, public_key(key.with_suffix(".pub").read_text().strip()), "sentiment-" + str(report["run_id"]) + "-" + epoch,
                      os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    validate_lease(lease, row)
    write_json(root / "lease.json", lease)
    write_json(root / "serving-row.json", row)
    print("PUBLIC_RESTRICTED_LEASE_PREPARED")


def run(root):
    remote_only()
    row = strict_json((root / "serving-row.json").read_bytes())
    lease = strict_json((root / "lease.json").read_bytes())
    validate_lease(lease, row)
    report = validate_ready(strict_json((root / "registration-ready.json").read_bytes()))
    if file_sha(root / "registration-ready.json") != row["candidate_proof"]["sha256"]:
        raise ValueError("serving report binding changed")
    if file_sha(root / "SENTIMENT_ANALYSIS-recovery.safetensors") != row["sha256"]:
        raise ValueError("serving package changed")
    created = datetime.fromisoformat(lease["created_at"])
    end = datetime.fromisoformat(lease["expires_at"])
    duration = (end - datetime.now(timezone.utc)).total_seconds()
    timer = threading.Timer(duration, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    forward = None
    try:
        with executor(root, root / "SENTIMENT_ANALYSIS-recovery.safetensors", row["keccak"]) as (backend, _, server):
            classes(root)
            expected_vectors = [v["tokens"] for v in report["runtime"]["vectors"]]
            expected_bits = [v["score_bits"] for v in report["runtime"]["vectors"]]
            if score(backend, row, expected_vectors) != expected_bits or score(backend, row, expected_vectors) != expected_bits:
                raise ValueError("served actual reference bits differ from the audit")
            identity_controls(server, row)
            command = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"),
                       "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                       "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(root / "known_hosts"),
                       "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
                       "-o", "ServerAliveCountMax=3", "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{server.server_port}",
                       "telegraph-verify@98.95.110.88"]
            authorized = time.monotonic() + min(1200, duration)
            while time.monotonic() < authorized:
                forward = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           env={"PATH": os.environ["PATH"], "HOME": str(root)})
                time.sleep(3)
                if forward.poll() is None:
                    break
                forward.wait(timeout=5)
                time.sleep(5)
            else:
                raise RuntimeError("restricted lease authorization timeout")
            print("EXACT_REFERENCE_EXECUTOR_FORWARDED", flush=True)
            deadline = time.monotonic() + duration
            next_poll = 0
            while time.monotonic() < deadline:
                if server.fatal.is_set() or forward.poll() is not None:
                    raise RuntimeError("bounded proxy or forwarding stopped")
                if time.monotonic() >= next_poll:
                    url = NODE + "/api/wasm?intent=" + FAMILY + "&poll=" + str(time.time_ns())
                    try:
                        with urllib.request.urlopen(urllib.request.Request(url, headers={"Cache-Control": "no-cache", "User-Agent": "curl/8.0 Telegraph-owned-serving"}), timeout=25) as response:
                            raw = response.read(MAX_BODY + 1)
                        if len(raw) > MAX_BODY:
                            raise ValueError("node listing bound")
                        entries = strict_json(raw).get("intents", {}).get(FAMILY, {}).get("entries", [])
                        for entry in entries:
                            if (entry.get("intent_id") != FAMILY or entry.get("author_address", "").lower() != OWNER or
                                    entry.get("wasm_hash", "").removeprefix("0x") != row["keccak"]):
                                continue
                            if row["existing_registration_id"]:
                                if entry.get("registration_id") != row["existing_registration_id"]:
                                    continue
                            elif datetime.fromisoformat(entry["registered_at"].replace("Z", "+00:00")) < created:
                                continue
                            status = entry.get("activation_status")
                            write_json(root / "node-state.json", {"family": FAMILY, "owner": OWNER, "package_keccak": row["keccak"],
                                       "registration_id": entry["registration_id"], "status": status})
                            # Keep the exact active registration served until the lease ceiling.
                            if status in ("rejected", "superseded", "deregistered"):
                                return
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                    next_poll = time.monotonic() + 30
                time.sleep(1)
    finally:
        timer.cancel()
        if forward and forward.poll() is None:
            forward.terminate()
            try:
                forward.wait(timeout=10)
            except subprocess.TimeoutExpired:
                forward.kill()
                forward.wait(timeout=5)
        for name in ("forward-key", "forward-key.pub"):
            (root / name).unlink(missing_ok=True)


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package-url")
    parser.add_argument("--package-sha256")
    parser.add_argument("--package-keccak")
    parser.add_argument("--slot", type=int)
    parser.add_argument("--forward-port", type=int)
    parser.add_argument("--runnable-url")
    parser.add_argument("--existing", type=int, default=0)
    args = parser.parse_args()

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.phase == "prepare":
        prepare(args.root, args.package_url, args.package_sha256, args.package_keccak,
                args.slot, args.forward_port, args.runnable_url, args.existing)
    else:
        run(args.root)


if __name__ == "__main__":
    main()
