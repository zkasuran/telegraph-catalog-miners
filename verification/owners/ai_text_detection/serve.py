# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only bounded serving of an already verified immutable package.

No operator key is read. The runner creates one ephemeral restricted forwarding
key. Only its public lease is an upload artifact.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

from audit_runtime import executor
from common import FAMILY, HERE, dump, fetch, remote_only, sha
from proxy import Backend, BoundedServer
from reg_common import DEFAULT_OWNER, HOST, HOST_KEY, lease_for
from run import prepare
from wire import check_reply, request


def main():
    remote_only()
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--package-url", required=True)
    p.add_argument("--package-sha256", required=True)
    p.add_argument("--report-url", required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--forward-port", type=int, required=True)
    p.add_argument("--slot", type=int, required=True)
    p.add_argument("--runnable-url", required=True)
    args = p.parse_args()
    prefix = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/"
    for url in (args.package_url, args.report_url):
        if not re.fullmatch(re.escape(prefix) + r"[a-f0-9]{40}/verification/owners/ai_text_detection/[A-Za-z0-9_./-]+", url) or ".." in url.split("/"):
            raise ValueError("immutable owned public URL required")
    if not 1 <= args.slot <= 6 or args.forward_port != 48600 + args.slot or not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", args.runnable_url):
        raise ValueError("assigned free endpoint slot required")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    package = root / (FAMILY + ".safetensors")
    raw = fetch(args.report_url, root / "registration-ready.json", args.report_sha256, 256 << 10)
    report = json.loads(raw)
    if (report["status"] != "REGISTRATION_READY" or report["family"] != FAMILY or
            report["package_sha256"] != args.package_sha256 or report["serving_commit"] != os.environ["GITHUB_SHA"]):
        raise ValueError("passing report/serving source identity mismatch")
    fetch(args.package_url, package, args.package_sha256, 64 << 20)
    _, _, kit = prepare(root)
    subprocess.run(["go", "build", "-trimpath", "-o", str(root / "adapterexec"), "./cmd/adapterexec"],
                   cwd=kit, timeout=600, check=True)
    proof = {"url": args.report_url, "sha256": args.report_sha256}
    row = {"family": FAMILY, "owner": DEFAULT_OWNER, "slot": args.slot, "forward_port": args.forward_port,
           "runnable_url": args.runnable_url, "sha256": args.package_sha256, "keccak": report["package_keccak"],
           "existing_registration_id": 0, "candidate_proof": proof, "runtime_proof": proof, "source_proof": proof}
    with executor(root, package) as url:
        vectors = report["runtime"]["vectors"]
        identity = {"foundation_hash": report["foundation_sha256"], "adapter_hash": report["package_keccak"],
                    "tokenizer_hash": report["tokenizer_sha256"]}
        if check_reply(request(url, [v["tokens"] for v in vectors], identity), len(vectors), identity) != [v["score_bits"] for v in vectors]:
            raise ValueError("served delta differs from audited real reference vectors")
        backend_port = int(url.split(":")[2].split("/")[0])
        backend = Backend(backend_port)
        server = BoundedServer(("127.0.0.1", 0), row, backend)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        local_port = server.server_address[1]
        try:
            proxy_url = f"http://127.0.0.1:{local_port}/runnable"
            if check_reply(request(proxy_url, [v["tokens"] for v in vectors], identity), len(vectors), identity) != [v["score_bits"] for v in vectors]:
                raise ValueError("bounded serving proxy changes reference vectors")
            key = root / "ephemeral-forward-key"
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True, timeout=20)
            public = key.with_suffix(".pub").read_text().strip()
            lease = lease_for(row, public, "ai-origin-owner", os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
            dump(root / "lease.json", lease)
            (root / "known_hosts").write_text(HOST_KEY)
            # Activation can occur while the public lease upload step runs. Wait
            # briefly then let restricted SSH retries establish that same lease.
            deadline = time.monotonic() + 5 * 3600 - 120
            while time.monotonic() < deadline and not server.fatal.is_set():
                command = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(key), "-o", "IdentitiesOnly=yes",
                           "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                           "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(root / "known_hosts"),
                           "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes",
                           "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no",
                           "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no", "-o", "RequestTTY=no",
                           "-R", f"127.0.0.1:{args.forward_port}:127.0.0.1:{local_port}", "telegraph-verify@" + HOST]
                proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    proc.wait(timeout=min(600, max(1, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=15)
                time.sleep(10)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=10)


if __name__ == "__main__":
    main()
