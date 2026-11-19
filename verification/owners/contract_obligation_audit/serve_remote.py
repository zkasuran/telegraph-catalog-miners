# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One exact package on a bounded free runner, with only a public restricted lease."""
import argparse
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from contracts import FAMILY, FOUNDATION, file_sha, quality, remote_only, strict_json, write_json
from data import download
from executor import ReferenceExecutor, Server
from lease_contract import lease_for
from prepare_remote import compile_runtime, full_parser, prepare

OWNER = "0x8b224783fe5b3c52b7db0cb9b1754f8812b75287"
HOST = "98.95.110.88"
HOST_KEY = "98.95.110.88 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMRYtmP2ESjH1ah2MI3eqUmEgtVYeyR81RAbr0MB/r0G\n"


def immutable_url(value):
    if not isinstance(value, str) or not re.fullmatch(r"https://raw\.githubusercontent\.com/zkasuran/telegraph-catalog-miners/[a-f0-9]{40}/verification/owners/contract_obligation_audit/[A-Za-z0-9_./-]+", value) or ".." in value:
        raise ValueError("package/proof URL must be in this exact immutable public namespace")
    return value


def prepare_lease(root):
    remote_only()
    package_url = immutable_url(os.environ["PACKAGE_URL"])
    report_url = immutable_url(os.environ["REPORT_URL"])
    for key in ("PACKAGE_SHA256", "PACKAGE_KECCAK", "REPORT_SHA256"):
        if not re.fullmatch(r"[a-f0-9]{64}", os.environ[key]):
            raise ValueError("exact lowercase package/report hashes required")
    root.mkdir(parents=True, exist_ok=True)
    package = root/(FAMILY+".safetensors")
    report_path = root/"registration-ready.json"
    download(package_url, package, os.environ["PACKAGE_SHA256"], 64 << 20)
    download(report_url, report_path, os.environ["REPORT_SHA256"], 1 << 20)
    report = strict_json(report_path.read_bytes())
    if (report["status"] != "REGISTRATION_READY" or report["family"] != FAMILY or report["serving_commit"] != os.environ["GITHUB_SHA"]
            or report["package_sha256"] != os.environ["PACKAGE_SHA256"] or report["package_keccak"] != os.environ["PACKAGE_KECCAK"]
            or not quality(report["quality"]["formats"])["passed"] or report["source"]["independent_groups"] < 32
            or report["source"]["audit_used_for_selection"] is not False or report["source"]["normalized_overlap"] != 0):
        raise ValueError("actual complete audit report is not ready for this serving code")
    _, base, kit = prepare(root)
    compile_runtime(root, kit)
    cfg = strict_json((base/"config.json").read_bytes())
    hidden, intermediate = cfg["hidden_size"], cfg["intermediate_size"]
    write_json(root/"foundation.json", {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
        "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": FOUNDATION["tokenizer.json"],
        "r_max": 16, "num_layers": cfg["num_hidden_layers"], "hidden_size": hidden,
        "targets": {"attn.Wqkv": [hidden, 3*hidden], "attn.Wo": [hidden, hidden], "mlp.Wi": [hidden, 2*intermediate], "mlp.Wo": [intermediate, hidden]}})
    parsed = full_parser(root, package, root/"foundation.json")
    if parsed["keccak256"].removeprefix("0x") != report["package_keccak"]:
        raise ValueError("served parser package identity changed")
    slot, port = int(os.environ["SLOT"]), int(os.environ["FORWARD_PORT"])
    runnable = os.environ["RUNNABLE_URL"]
    if not 1 <= slot <= 6 or port != 48600+slot or not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", runnable):
        raise ValueError("only a supplied existing-proxy free reservation is allowed")
    row = {"family": FAMILY, "owner": OWNER, "slot": slot, "forward_port": port, "runnable_url": runnable,
           "keccak": report["package_keccak"], "sha256": report["package_sha256"], "existing_registration_id": 0}
    for kind in ("candidate", "runtime", "source"):
        row[kind+"_proof"] = {"sha256": os.environ["REPORT_SHA256"]}
    key_path = root/"forward-key"
    if key_path.exists():
        raise ValueError("existing ephemeral lease must be resumed, not replaced")
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key_path), "-C", "contract-obligation-forward"], check=True, timeout=15)
    key_path.chmod(0o600)
    (root/"known_hosts").write_text(HOST_KEY)
    lease = lease_for(row, key_path.with_suffix(".pub").read_text().strip(), "contract-owner", os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    public = root/"public-lease"
    public.mkdir(exist_ok=True)
    write_json(public/"lease.json", lease)
    write_json(root/"serving-state.json", {"row": row, "package": str(package), "report_sha256": file_sha(report_path)})
    print("Restricted public lease prepared. No forwarding or signer private key is uploaded.")


def hold(root):
    remote_only()
    state = strict_json((root/"serving-state.json").read_bytes())
    row = state["row"]
    report = strict_json((root/"registration-ready.json").read_bytes())
    package = Path(state["package"])
    if file_sha(package) != row["sha256"] or file_sha(root/"registration-ready.json") != state["report_sha256"]:
        raise ValueError("immutable serving state changed")
    backend = ReferenceExecutor(root, package, row["keccak"])
    server = Server(("127.0.0.1", 0), backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    identity = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "adapter_hash": row["keccak"], "tokenizer_hash": FOUNDATION["tokenizer.json"]}
    forwarding = None
    try:
        actual = backend.score({**identity, "batch": [{"tokens": item["tokens"]} for item in report["runtime"]["vectors"]]})
        if [item["score"] for item in actual["results"]] != [item["score_bits"] for item in report["runtime"]["vectors"]]:
            raise ValueError("serving executor differs from actual frozen reference vectors")
        command = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root/"forward-key"),
            "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile="+str(root/"known_hosts"), "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no",
            "-o", "ForwardX11=no", "-o", "RequestTTY=no", "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{server.server_port}", "telegraph-verify@"+HOST]
        expires = time.monotonic()+5*3600-300
        with (root/"forward.log").open("w") as log:
            while time.monotonic() < expires:
                if backend.process.poll() is not None:
                    raise RuntimeError("serving reference engine stopped")
                if forwarding is None or forwarding.poll() is not None:
                    forwarding = subprocess.Popen(command, stdout=log, stderr=log)
                time.sleep(5)
            if forwarding is not None and forwarding.poll() is None:
                forwarding.terminate()
                forwarding.wait(timeout=10)
    finally:
        if forwarding is not None and forwarding.poll() is None:
            forwarding.terminate()
            try:
                forwarding.wait(timeout=10)
            except subprocess.TimeoutExpired:
                forwarding.kill()
                forwarding.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        backend.close()


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "hold"), required=True)
    args = parser.parse_args()
    (prepare_lease if args.phase == "prepare" else hold)(args.root.resolve())


if __name__ == "__main__":
    main()
