# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only exact public executor and restricted five-hour forwarding lease."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
from common import DEFAULT_OWNER, HOST, HOST_KEY, lease_for, immutable_url, FOUNDATION
from reference import prepare, fetch
from source_gate import digest, write


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def start(root):
    package_sha = os.environ["PACKAGE_SHA256"]
    package_keccak = os.environ["PACKAGE_KECCAK"]
    package_url = immutable_url(os.environ["PACKAGE_URL"])
    report_url = immutable_url(os.environ["REPORT_URL"])
    report_sha = os.environ["REPORT_SHA256"]
    fetch(package_url, root / "package.safetensors", package_sha, 64 << 20)
    fetch(report_url, root / "registration-ready.json", report_sha, 1 << 20)
    ready = json.loads((root / "registration-ready.json").read_text())
    if (ready["status"] != "REGISTRATION_READY" or ready["family"] != "LLM_OUTPUT_EVALUATION"
            or ready["package_sha256"] != package_sha or ready["package_keccak"] != package_keccak
            or ready["serving_commit"] != os.environ["GITHUB_SHA"]):
        raise ValueError("serving cannot substitute a candidate or source commit")
    helper, foundation, base, kit = __import__('remote').reference(root)
    subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / "adapterexec"), "./cmd/adapterexec"], cwd=kit, check=True, timeout=600)
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
    row = {"family": ready["family"], "owner": DEFAULT_OWNER, "keccak": package_keccak,
           "sha256": package_sha, "slot": int(os.environ["ENDPOINT_SLOT"]),
           "forward_port": int(os.environ["FORWARD_PORT"]), "runnable_url": os.environ["RUNNABLE_URL"],
           "existing_registration_id": 0, "candidate_proof": {"sha256": report_sha},
           "runtime_proof": {"sha256": report_sha}, "source_proof": {"sha256": report_sha}}
    backend, proxy_port = free_port(), free_port()
    executable = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / "package.safetensors"), "--listen", f"127.0.0.1:{backend}"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    write(root / "serve-row.json", row)
    proxy = subprocess.Popen([os.sys.executable, str(Path(__file__).parent / "serve_proxy.py"), "--port", str(proxy_port), "--backend", str(backend), "--row", str(root / "serve-row.json")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    key = root / "forward-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "telegraph-owner-llm"], check=True, timeout=15)
    key.chmod(0o600)
    (root / "known_hosts").write_text(HOST_KEY)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if executable.poll() is not None or proxy.poll() is not None:
            raise ValueError("exact serving executor failed to start")
        try:
            with socket.create_connection(("127.0.0.1", backend), timeout=1):
                with socket.create_connection(("127.0.0.1", proxy_port), timeout=1):
                    break
        except OSError:
            time.sleep(1)
    else:
        raise ValueError("serving readiness timeout")
    from proxy import identity, input_root
    vectors = ready["runtime"]["vectors"]
    body = {"batch": [{"tokens": v["tokens"]} for v in vectors], **identity(row)}
    body["input_root"] = input_root(body["batch"])
    request = urllib.request.Request(f"http://127.0.0.1:{proxy_port}/runnable", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        value = json.loads(response.read(1 << 20))
    if value["results"] != [{"score": v["score_bits"]} for v in vectors] or value["adapter_hash"] != package_keccak:
        raise ValueError("serving bits do not match original registration-ready audit")
    lease = lease_for(row, (root / "forward-key.pub").read_text().strip(), "llm-owner", os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    write(root / "lease.json", lease)
    write(root / "serving-state.json", {"proxy_port": proxy_port, "executor_pid": executable.pid, "proxy_pid": proxy.pid})
    # No private key or process state is uploaded. Only lease.json is public.


def hold(root):
    state = json.loads((root / "serving-state.json").read_text())
    row = json.loads((root / "serve-row.json").read_text())
    deadline = time.monotonic() + 4.8 * 3600
    ssh = None
    try:
        while time.monotonic() < deadline:
            for name in ("executor_pid", "proxy_pid"):
                os.kill(state[name], 0)
            if ssh is None or ssh.poll() is not None:
                args = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"),
                        "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(root / "known_hosts"),
                        "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
                        "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
                        "-o", "RequestTTY=no", "-o", "PermitLocalCommand=no", "-o", "LogLevel=ERROR",
                        "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{state['proxy_port']}", "telegraph-verify@" + HOST]
                ssh = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(10)
    finally:
        if ssh is not None:
            ssh.terminate()
            ssh.wait(timeout=20)
        for name in ("executor_pid", "proxy_pid"):
            try:
                os.kill(state[name], 15)
            except ProcessLookupError:
                pass


def main(root):
    stage = os.environ.get("SERVING_STAGE", "start")
    if stage == "start":
        start(root)
    elif stage == "hold":
        hold(root)
    else:
        raise ValueError("unsupported serving stage")
